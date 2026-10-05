"""Iter109 Step 0: 補間と温度を適用した後の確率に w_y = π_t(y)/π_s(y) を掛けたときの eval の決定を replay で見積もる．

使い方: リポジトリ直下で次を実行する（開発ホストの CPU だけ．wafl500〜509 は使わない）．
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        uv run python .claude/research/_iter109_replay_prior.py

π_s: 訓練の外側 5-fold（seed 104．Iter105 と同じ分割）の OOF 補間分布に本番と同じ温度を当てた確率の平均．
π_t: ラベルなしの eval の全クエリ（3,750 行）に対する Saerens らの EM（Neural Computation 14(1):21-41, 2002）の推定値．
     収束まで回し，調整する値は持たない．eval の実際の分布との差は診断のためだけに記録し，選択には使わない．
比較の範囲: 基準線 results/20261004_225553 の共通行（confidence が非 null．dispatch_failed を除く．
            _iter107_main_summary.py と同じ条件）で，「補正 − 無補正」の replay 同士を比べる．
進む条件: Δtop1 >= +0.5pt かつ top1 の正味の行数がノイズの床（B200: ±2 行）を超えること．
"""

import json
import sys
from collections import Counter

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

REPO = "/mnt/data-raid/ktakahashi/workspace/expert-mesh"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/.claude/research")
from _iter105_replay_temp import (  # noqa: E402
    apply_temperature,
    c1_tests,
    compound_only,
    decision_agreement,
    load_cache,
    run_outer_fold,
    simulate_rows,
    single_only,
    summarize,
)
from metrics import compute_mcnemar_test, compute_precision_recall_per_domain  # noqa: E402

BASELINE_RESULTS = f"{REPO}/results/20261004_225553/results.jsonl"
OUTPUT_PATH = f"{REPO}/.claude/research/_iter109_replay_prior.json"
CV_SEED = 104
N_OUTER_FOLDS = 5
# Iter105 で採用した温度（選び直さない．artifact の値がこれと一致することを確かめる）
EXPECTED_TEMPERATURE = 0.9426
TEMPERATURE_TOLERANCE = 5e-5
# EM の停止条件: 事前分布の各成分の変化の最大値がこれ未満になったら収束とみなす
EM_TOLERANCE = 1e-10
# 収束しない場合の打ち切り（打ち切ったら converged=False として記録する）
EM_MAX_ITER = 100000
# 進む条件（journal の Iteration 109 の Step 0）
MIN_DELTA_TOP1 = 0.005
NOISE_FLOOR_ROWS = 2


def estimate_target_prior_by_saerens_em(
    proba: np.ndarray, prior_source: np.ndarray
) -> tuple[np.ndarray, int, bool]:
    """Saerens らの EM で，ラベルなしの行の事前分布を推定する（推定値，反復回数，収束の有無を返す）．"""
    prior = prior_source.copy()
    for iteration in range(1, EM_MAX_ITER + 1):
        adjusted = proba * (prior / prior_source)
        adjusted /= adjusted.sum(axis=1, keepdims=True)
        new_prior = adjusted.mean(axis=0)
        if np.max(np.abs(new_prior - prior)) < EM_TOLERANCE:
            return new_prior, iteration, True
        prior = new_prior
    return prior, EM_MAX_ITER, False


def reweight_by_prior_ratio(proba: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """確率にクラスごとの重みを掛けて，行ごとに正規化し直す．"""
    scaled = proba * weights
    return scaled / scaled.sum(axis=1, keepdims=True)


def describe_changed_rows(before: list[dict], after: list[dict]) -> dict:
    """argmax が変わった行の数，向き（正誤の遷移と組），複合行の送出集合の変化を数える．"""
    transitions: Counter = Counter()
    pairs: Counter = Counter()
    pairs_single: Counter = Counter()
    compound_set_changed = 0
    compound_k_change: Counter = Counter()
    for old, new in zip(before, after):
        assert old["id"] == new["id"]
        is_compound = len(old["expected_domains"]) > 1
        if is_compound and set(old["dispatched_domains"]) != set(new["dispatched_domains"]):
            compound_set_changed += 1
            compound_k_change[f"{len(old['dispatched_domains'])}->{len(new['dispatched_domains'])}"] += 1
        if old["selected_domain"] == new["selected_domain"]:
            continue
        old_ok = old["selected_domain"] in old["expected_domains"]
        new_ok = new["selected_domain"] in new["expected_domains"]
        transitions[f"{'correct' if old_ok else 'wrong'}->{'correct' if new_ok else 'wrong'}"] += 1
        pairs[f"{old['selected_domain']}->{new['selected_domain']}"] += 1
        if not is_compound:
            truth = old["expected_domains"][0]
            pairs_single[f"[{truth}] {old['selected_domain']}->{new['selected_domain']}"] += 1
    net_rows = transitions["wrong->correct"] - transitions["correct->wrong"]
    return {
        "n_argmax_changed": int(sum(transitions.values())),
        "transitions": dict(transitions),
        "net_top1_rows": int(net_rows),
        "pairs": dict(pairs.most_common()),
        "pairs_single_with_truth": dict(pairs_single.most_common()),
        "n_compound_dispatched_set_changed": compound_set_changed,
        "compound_k_change": dict(compound_k_change),
    }


def recall_differences(new_rows: list[dict], old_rows: list[dict], classes: list[str]) -> dict:
    """ドメインごとの recall の新旧と差を metrics.compute_precision_recall_per_domain で求める．"""
    new_pr = compute_precision_recall_per_domain(new_rows)
    old_pr = compute_precision_recall_per_domain(old_rows)
    return {
        d: {"recall_none": old_pr[d]["recall"], "recall_corrected": new_pr[d]["recall"],
            "diff": new_pr[d]["recall"] - old_pr[d]["recall"]}
        for d in classes
    }


def main() -> None:
    """π_s と π_t を求め，eval の決定を「補正 − 無補正」で比べて JSON に保存する．"""
    train_x, _, train_domains = load_cache("train")
    eval_x, eval_ids, _ = load_cache("eval")
    artifact = joblib.load(f"{REPO}/models/domain_classifier.joblib")
    assert type(artifact).__name__ == "KnnInterpolatedClassifier"
    temperature = float(artifact.temperature)
    assert abs(temperature - EXPECTED_TEMPERATURE) < TEMPERATURE_TOLERANCE
    classes = list(artifact.classes_)
    assert classes == sorted(set(train_domains))
    labels = np.array(train_domains)

    # 1. π_s: OOF の補間分布に本番と同じ温度を当てた確率の平均
    folds = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=CV_SEED).split(train_x, labels)
    outputs = Parallel(n_jobs=N_OUTER_FOLDS)(
        delayed(run_outer_fold)(train_x, labels, tr, va, classes) for tr, va in folds
    )
    oof = np.zeros((len(labels), len(classes)))
    for val_idx, mix in outputs:
        oof[val_idx] = mix
    oof_t = apply_temperature(oof, temperature)
    prior_source = oof_t.mean(axis=0)
    train_counts = Counter(train_domains)

    # 2. π_t: artifact の predict_proba は補間と温度を適用済みなので，そのまま EM に渡す（3,750 行すべて）
    eval_proba = np.asarray(artifact.predict_proba(eval_x), dtype=np.float64)
    prior_target, n_iter, converged = estimate_target_prior_by_saerens_em(eval_proba, prior_source)
    weights = prior_target / prior_source
    corrected = reweight_by_prior_ratio(eval_proba, weights)

    # 3. eval の決定を共通行（基準線の confidence が非 null）の上で replay する
    with open(BASELINE_RESULTS, encoding="utf-8") as f:
        baseline_rows = [json.loads(line) for line in f if line.strip()]
    assert sorted(r["id"] for r in baseline_rows) == sorted(eval_ids)
    base_by_id = {r["id"]: r for r in baseline_rows}
    keep = np.array([base_by_id[i]["confidence"] is not None for i in eval_ids])
    common_ids = [i for i, k in zip(eval_ids, keep) if k]
    common_baseline = [base_by_id[i] for i in common_ids]
    sim_none = simulate_rows(baseline_rows, common_ids, eval_proba[keep], classes)
    sim_corrected = simulate_rows(baseline_rows, common_ids, corrected[keep], classes)
    summary_none = summarize(sim_none)
    summary_corrected = summarize(sim_corrected)
    changes = describe_changed_rows(sim_none, sim_corrected)

    # 診断のためだけ（選択には使わない）: eval の実際の分布
    single_counts = Counter(r["expected_domains"][0] for r in single_only(baseline_rows))
    n_single = sum(single_counts.values())
    compound_mass: Counter = Counter()
    for r in compound_only(baseline_rows):
        for d in r["expected_domains"]:
            compound_mass[d] += 1.0 / len(r["expected_domains"])
    all_mass = {c: single_counts[c] + compound_mass[c] for c in classes}
    n_all = sum(all_mass.values())
    mean_pred_eval = eval_proba.mean(axis=0)

    delta = {
        key: summary_corrected[key] - summary_none[key]
        for key in ("top1", "top1_single", "top1_compound", "ece",
                    "compound_domain_set_recall", "compound_mean_dispatched_count", "mean_k_all")
    }
    c1 = c1_tests(sim_corrected, sim_none, classes)
    result = {
        "config": {
            "temperature": temperature, "cv_seed": CV_SEED, "n_outer_folds": N_OUTER_FOLDS,
            "n_train": int(len(labels)), "n_eval_em": int(len(eval_ids)), "n_common": int(len(common_ids)),
            "classes": classes,
        },
        "prior_source": dict(zip(classes, prior_source.tolist())),
        "oof_top1": float(np.mean(oof_t.argmax(axis=1) == np.array([classes.index(d) for d in train_domains]))),
        "train_count_share": {c: train_counts[c] / len(labels) for c in classes},
        "mean_pred_eval_uncorrected": dict(zip(classes, mean_pred_eval.tolist())),
        "prior_target_em": dict(zip(classes, prior_target.tolist())),
        "weights": dict(zip(classes, weights.tolist())),
        "em": {"n_iter": n_iter, "converged": converged, "tolerance": EM_TOLERANCE},
        "diagnostic_not_for_selection": {
            "actual_single_row_share": {c: single_counts[c] / n_single for c in classes},
            "actual_all_rows_share_compound_split_evenly": {c: all_mass[c] / n_all for c in classes},
            "pi_t_minus_actual_all": {c: float(prior_target[i] - all_mass[c] / n_all) for i, c in enumerate(classes)},
        },
        "agreement_none_vs_actual": decision_agreement(sim_none, common_baseline),
        "summary_actual_common": summarize(common_baseline),
        "summary_none": summary_none,
        "summary_corrected": summary_corrected,
        "delta": delta,
        "changes": changes,
        "mcnemar_all": compute_mcnemar_test(sim_corrected, sim_none),
        "mcnemar_single": compute_mcnemar_test(single_only(sim_corrected), single_only(sim_none)),
        "mcnemar_compound": compute_mcnemar_test(compound_only(sim_corrected), compound_only(sim_none)),
        "recall_diff": recall_differences(sim_corrected, sim_none, classes),
        "c1_significant": c1["significant"],
        "c1_min_p": c1["min_p"],
        "passed": bool(delta["top1"] >= MIN_DELTA_TOP1 and changes["net_top1_rows"] > NOISE_FLOOR_ROWS),
        "pass_rule": "delta_top1 >= +0.005 and net_top1_rows > 2 (common rows)",
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print("wrote", OUTPUT_PATH)
    for key in ("prior_source", "prior_target_em", "weights", "em", "delta", "passed", "agreement_none_vs_actual"):
        print(key, json.dumps(result[key], ensure_ascii=False))
    print("changes", json.dumps({k: v for k, v in changes.items() if k not in ("pairs",)}, ensure_ascii=False))
    print("recall_diff", json.dumps(result["recall_diff"], ensure_ascii=False))
    print("mcnemar_all", json.dumps(result["mcnemar_all"]))
    print("c1_significant", result["c1_significant"])


if __name__ == "__main__":
    main()
