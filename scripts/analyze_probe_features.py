"""Probe candidates の特徴量抽出と offline logistic regression 評価。

results.jsonl から per-domain-per-query data point を抽出し、
6 種の特徴量を計算して LogisticRegression (L1) で training / evaluation を行う。
Phase 1 の目標は AUC >= 0.85 の達成。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "self_confidence",
    "max_other_confidence",
    "margin",
    "is_top1",
    "confidence_spread",
    "num_above_threshold",
]
CONFIDENCE_THRESHOLD = 0.5


def extract_features(
    probe_candidates: list[dict],
    selected_domain: str,
    expected_domains: list[str],
    domains: list[str],
) -> tuple[np.ndarray, list[int], list[str]]:
    """per-domain-per-query の data point を作成し、特徴量・ラベル・ドメインを返す。

    各ドメインにつき 1 data point を生成（domains の要素数 x query = N points）。
    ラベルは「そのドメインが selected_domain かつ expected_domains に含まれる」場合に 1。

    Parameters
    ----------
    probe_candidates : list[dict]
        [{"node_id": ..., "domain": ..., "confidence": ...}, ...]
    selected_domain : str
        実際に選択されたドメイン。
    expected_domains : list[str]
        正解のドメインリスト。
    domains : list[str]
        このメッシュ構成が持つ全ドメイン（4 でも 10 でも可）。呼び出し元が
        results.jsonl から動的に導出する（メッシュのドメイン数をこのスクリプトに
        ハードコードしない）。

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
        特徴量行列。
    labels : list[int]
        ラベル（1=正解, 0=misroute or not-selected）。
    domains_seen : list[str]
        各行のドメイン名。
    """
    confidences = {c["domain"]: c["confidence"] for c in probe_candidates}
    all_conf = [c["confidence"] for c in probe_candidates]
    max_conf = max(all_conf) if all_conf else 0.0
    std_conf = float(np.std(all_conf)) if len(all_conf) > 1 else 0.0
    num_above = sum(1 for c in all_conf if c > CONFIDENCE_THRESHOLD)

    X_rows: list[list[float]] = []
    labels: list[int] = []
    domains_seen: list[str] = []

    for domain in domains:
        if domain not in confidences:
            continue
        self_conf = confidences[domain]
        other_confs = [c for d, c in confidences.items() if d != domain]
        max_other = max(other_confs) if other_confs else 0.0
        margin = self_conf - max_other
        is_top1 = 1.0 if self_conf == max_conf and domain == selected_domain else 0.0

        is_correct = domain == selected_domain and domain in expected_domains
        label = 1 if is_correct else 0

        X_rows.append([self_conf, max_other, margin, is_top1, std_conf, num_above])
        labels.append(label)
        domains_seen.append(domain)

    return np.array(X_rows), labels, domains_seen


def train_and_evaluate(
    X: np.ndarray,
    y: list[int],
    domains: list[str],
) -> dict:
    """LogisticRegression (L1) を訓練し、評価指標を返す。

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        特徴量行列。
    y : list[int]
        ラベル。
    domains : list[str]
        各行のドメイン名。

    Returns
    -------
    results : dict
        AUC, Precision, Recall, F1, confusion matrix, feature coefficients, per-domain metrics.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        l1_ratio=1.0,
        solver="saga",
        C=1.0,
        max_iter=10000,
        random_state=42,
    )
    model.fit(X_scaled, y)

    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y, y_pred_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="binary", zero_division=0
    )

    cm = confusion_matrix(y, y_pred).flatten()
    tn, fp, fn, tp = cm[0], cm[1], cm[2], cm[3]

    coefficients = dict(zip(FEATURE_NAMES, model.coef_[0]))

    # per-domain metrics (iterates the domains actually present in this
    # results.jsonl, not a hardcoded mesh size)
    per_domain: dict[str, dict[str, float]] = {}
    for d in sorted(set(domains)):
        mask = [i for i, dom in enumerate(domains) if dom == d]
        if not mask:
            continue
        y_d = [y[i] for i in mask]
        yp_d = [y_pred[i] for i in mask]
        if sum(y_d) == 0 and sum(yp_d) == 0:
            per_domain[d] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            continue
        p, r, f, _ = precision_recall_fscore_support(y_d, yp_d, average="binary", zero_division=0)
        per_domain[d] = {"precision": float(p), "recall": float(r), "f1": float(f)}

    return {
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "coefficients": coefficients,
        "per_domain": per_domain,
    }


def format_results(
    total: int,
    positive: int,
    negative: int,
    metrics: dict,
) -> str:
    """評価結果を指定フォーマットで文字列化して返す。"""
    lines: list[str] = []
    lines.append("=== Probe Feature Analysis (Phase 1: Offline) ===")
    lines.append(f"Total samples: {total}")
    lines.append(f"Positive samples: {positive} (correctly routed)")
    lines.append(f"Negative samples: {negative} (misrouted or not selected)")
    lines.append("")

    lines.append("=== Model Performance ===")
    lines.append(f"AUC: {metrics['auc']:.3f}")
    lines.append(f"Precision: {metrics['precision']:.3f}")
    lines.append(f"Recall: {metrics['recall']:.3f}")
    lines.append(f"F1: {metrics['f1']:.3f}")
    lines.append("")

    lines.append("=== Confusion Matrix ===")
    lines.append(f"[[{metrics['tn']}, {metrics['fp']}],")
    lines.append(f" [{metrics['fn']}, {metrics['tp']}]]")
    lines.append("")

    lines.append("=== Feature Coefficients ===")
    sorted_coeffs = sorted(metrics["coefficients"].items(), key=lambda x: abs(x[1]), reverse=True)
    for name, coeff in sorted_coeffs:
        lines.append(f"  {name}: {coeff:+.4f}")
    lines.append("")

    lines.append("=== Per-domain Results ===")
    for d in sorted(metrics["per_domain"]):
        pd = metrics["per_domain"][d]
        lines.append(
            f"  {d}: precision={pd['precision']:.3f}, recall={pd['recall']:.3f}, f1={pd['f1']:.3f}"
        )
    lines.append("")

    lines.append("=== Success Criteria Check ===")
    auc_pass = metrics["auc"] >= 0.85
    status = "PASS" if auc_pass else "FAIL"
    lines.append(f"AUC >= 0.85: {status}")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point。results.jsonl を読み、offline evaluation を実行して出力する。"""
    parser = argparse.ArgumentParser(
        description="Probe candidates の特徴量を抽出し、offline logistic regression で評価"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="results.jsonl のパス（例: results/20260721_222225/results.jsonl）",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: {results_path} not found.", file=sys.stderr)
        sys.exit(1)

    # Load all records
    records: list[dict] = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Derive the mesh's domain set from the data itself (whatever number of
    # nodes actually produced this results.jsonl), rather than hardcoding a
    # domain count/list here.
    mesh_domains = sorted({c["domain"] for rec in records for c in rec.get("probe_candidates", [])})

    # Extract features from each record
    X_rows: list[list[float]] = []
    all_labels: list[int] = []
    all_domains: list[str] = []

    for rec in records:
        probe_candidates = rec.get("probe_candidates", [])
        selected_domain = rec.get("selected_domain", "")
        expected_domains = rec.get("expected_domains", [])

        X, labels, domains = extract_features(
            probe_candidates, selected_domain, expected_domains, mesh_domains
        )
        X_rows.extend(X.tolist())
        all_labels.extend(labels)
        all_domains.extend(domains)

    X = np.array(X_rows)
    y = all_labels

    total = len(y)
    positive = sum(y)
    negative = total - positive

    metrics = train_and_evaluate(X, y, all_domains)
    output = format_results(total, positive, negative, metrics)
    print(output)


if __name__ == "__main__":
    main()
