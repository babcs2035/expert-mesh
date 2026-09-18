#!/usr/bin/env python3
"""Independent metric computation for Iter52 analysis.

Reads iter44_boundary_tuning_calibrated_predictions.jsonl (baseline) and
iter52_threshold0.02_predictions.jsonl / iter52_threshold0.05_predictions.jsonl (after),
and computes ALL metrics from scratch.
"""
import json
import math
import sys
from collections import defaultdict

# ---- Load data ----
def load_predictions(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

baseline = load_predictions("results/iter44_boundary_tuning_calibrated_predictions.jsonl")
iter52a = load_predictions("results/iter52_threshold0.02_predictions.jsonl")
iter52b = load_predictions("results/iter52_threshold0.05_predictions.jsonl")

assert len(baseline) == len(iter52a) == len(iter52b) == 1600, \
    f"Row count mismatch: {len(baseline)}, {len(iter52a)}, {len(iter52b)}"

# Index by id for paired comparison
baseline_map = {r["id"]: r for r in baseline}
iter52a_map = {r["id"]: r for r in iter52a}
iter52b_map = {r["id"]: r for r in iter52b}

# ---- Helper functions ----
def is_correct(r):
    return r["selected_domain"] in r["expected_domains"]

def get_domain_correct(r, domain):
    return 1 if (r["selected_domain"] == domain and domain in r["expected_domains"]) else 0

def wilson_ci(success, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = success / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    margin = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n) / denom
    return (max(0, center - margin), min(1, center + margin))

def mcnemar_2x2(a_only, b_only):
    if (a_only + b_only) == 0:
        return {"discordant_a_only": a_only, "discordant_b_only": b_only,
                "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(a_only - b_only) - 1)**2 / (a_only + b_only)
    z_val = math.sqrt(chi2) if chi2 > 0 else 0
    t = 1.0 / (1.0 + 0.2316419 * z_val) if z_val > 0 else 1.0
    d = 0.3989422804014327
    p_one_tail = d * math.exp(-z_val*z_val/2) * (t*(0.319381530 + t*(-0.356563782 + t*(1.781477937 + t*(-1.821255978 + t*1.330274429)))))
    p_value = 2 * p_one_tail
    return {"discordant_a_only": a_only, "discordant_b_only": b_only,
            "chi2": round(chi2, 6), "p_value": round(p_value, 10)}

def bh_correction(p_values):
    indexed = [(p, i) for i, p in enumerate(p_values)]
    indexed.sort(key=lambda x: x[0])
    m = len(p_values)
    results = [None] * m
    for rank, (p, orig_idx) in enumerate(indexed, 1):
        q = p * m / rank
        q = min(q, 1.0)
        results[orig_idx] = (p, q)
    for i in range(m - 2, -1, -1):
        orig_idx = indexed[i+1][1]
        prev_idx = indexed[i][1]
        if results[prev_idx][1] > results[orig_idx][1]:
            results[prev_idx] = (results[prev_idx][0], results[orig_idx][1])
    return results

def compute_ece(predictions, n_bins=10):
    bins = [[] for _ in range(n_bins)]
    for r in predictions:
        conf = r["confidence"]
        if conf is None:
            continue
        correct = 1.0 if is_correct(r) else 0.0
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((conf, correct))
    n = len(predictions)
    ece = 0.0
    for bin_data in bins:
        if len(bin_data) == 0:
            continue
        avg_conf = sum(c for c, _ in bin_data) / len(bin_data)
        avg_acc = sum(a for _, a in bin_data) / len(bin_data)
        ece += len(bin_data) / n * abs(avg_conf - avg_acc)
    return ece

def compute_argmax_flip_rate(before, after):
    flipped = 0
    for rb, ra in zip(before, after):
        if rb["selected_domain"] != ra["selected_domain"]:
            flipped += 1
    return flipped / len(before), flipped

def compute_flip_transitions(before, after):
    transitions = defaultdict(int)
    for rb, ra in zip(before, after):
        if rb["selected_domain"] != ra["selected_domain"]:
            key = f"{rb['selected_domain']} -> {ra['selected_domain']}"
            transitions[key] += 1
    return transitions

def compute_edu_med_flips(before, after):
    edu_ctw = 0
    edu_wtc = 0
    edu_flips_from = defaultdict(int)
    edu_flips_to = defaultdict(int)
    for rb, ra in zip(before, after):
        if "education" in rb["expected_domains"]:
            if rb["selected_domain"] == "education" and ra["selected_domain"] != "education":
                edu_ctw += 1
                edu_flips_from[ra["selected_domain"]] += 1
            elif rb["selected_domain"] != "education" and ra["selected_domain"] == "education":
                edu_wtc += 1
                edu_flips_to[rb["selected_domain"]] += 1
    med_ctw = 0
    med_wtc = 0
    med_flips_from = defaultdict(int)
    med_flips_to = defaultdict(int)
    for rb, ra in zip(before, after):
        if "medical" in rb["expected_domains"]:
            if rb["selected_domain"] == "medical" and ra["selected_domain"] != "medical":
                med_ctw += 1
                med_flips_from[ra["selected_domain"]] += 1
            elif rb["selected_domain"] != "medical" and ra["selected_domain"] == "medical":
                med_wtc += 1
                med_flips_to[rb["selected_domain"]] += 1
    return {
        "edu_ctw": edu_ctw, "edu_wtc": edu_wtc,
        "edu_flips_from": dict(edu_flips_from), "edu_flips_to": dict(edu_flips_to),
        "med_ctw": med_ctw, "med_wtc": med_wtc,
        "med_flips_from": dict(med_flips_from), "med_flips_to": dict(med_flips_to),
    }

def fisher_exact_2x2(a, b, c, d):
    n = a + b + c + d
    if n == 0:
        return 1.0
    def hypergeom_p(k, N, K, n):
        if k < 0 or k > min(K, n) or k > n:
            return 0.0
        log_p = (math.lgamma(K+1) + math.lgamma(n+1) + math.lgamma(N-K+1) + math.lgamma(N-n+1)
                 - math.lgamma(N+1) - math.lgamma(k+1) - math.lgamma(K-k+1)
                 - math.lgamma(n-k+1) - math.lgamma(N-K-n+k+1))
        return math.exp(log_p)
    row_sums = [a+b, c+d]
    col_sums = [a+c, b+d]
    total = a+b+c+d
    min_k = max(0, row_sums[0] + col_sums[0] - total)
    max_k = min(row_sums[0], col_sums[0])
    obs_prob = hypergeom_p(a, total, col_sums[0], row_sums[0])
    two_tail_p = 0.0
    for k in range(min_k, max_k + 1):
        prob = hypergeom_p(k, total, col_sums[0], row_sums[0])
        if prob <= obs_prob + 1e-15:
            two_tail_p += prob
    return min(two_tail_p, 1.0)

# =====================================================================
# Domains list
# =====================================================================
domains = ["business_economics", "computer_science", "education", "general",
           "history_culture", "legal", "mathematics", "medical",
           "natural_science", "social_science"]

# =====================================================================
# Per-domain expected counts
# =====================================================================
domain_expected_count = defaultdict(int)
for r in baseline:
    for d in r["expected_domains"]:
        domain_expected_count[d] += 1

# =====================================================================
# Run analysis for both Iter52a and Iter52b
# =====================================================================
for label, after in [("Iter52a (threshold=0.02)", iter52a), ("Iter52b (threshold=0.05)", iter52b)]:
    print("=" * 70)
    print(f"COMPARISON: Baseline (Iter44) vs {label}")
    print("=" * 70)

    # 1. top1_accuracy
    before_correct = sum(1 for r in baseline if is_correct(r))
    after_correct = sum(1 for r in after if is_correct(r))
    before_acc = before_correct / len(baseline)
    after_acc = after_correct / len(after)
    print(f"\n  top1_accuracy: {before_acc:.4f} -> {after_acc:.4f} (delta={after_acc-before_acc:+.4f})")
    print(f"  Before: {before_correct}/{len(baseline)}")
    print(f"  After:  {after_correct}/{len(after)}")

    top1_before = [is_correct(r) for r in baseline]
    top1_after = [is_correct(r) for r in after]
    a_only = sum(1 for i in range(len(baseline)) if top1_before[i] and not top1_after[i])
    b_only = sum(1 for i in range(len(baseline)) if not top1_before[i] and top1_after[i])
    mcnemar_top1 = mcnemar_2x2(a_only, b_only)
    print(f"  McNemar: a_only={a_only}, b_only={b_only}, chi2={mcnemar_top1['chi2']:.4f}, p={mcnemar_top1['p_value']:.10f}")

    # 2. education_recall
    edu_before = sum(get_domain_correct(r, "education") for r in baseline)
    edu_after = sum(get_domain_correct(r, "education") for r in after)
    edu_total = domain_expected_count["education"]
    edu_before_r = edu_before / edu_total
    edu_after_r = edu_after / edu_total
    print(f"\n  education_recall: {edu_before_r:.4f} -> {edu_after_r:.4f} (delta={edu_after_r-edu_before_r:+.4f})")
    edu_before_ci = wilson_ci(edu_before, edu_total)
    edu_after_ci = wilson_ci(edu_after, edu_total)
    print(f"  Wilson 95% CI Before: [{edu_before_ci[0]:.4f}, {edu_before_ci[1]:.4f}]")
    print(f"  Wilson 95% CI After:  [{edu_after_ci[0]:.4f}, {edu_after_ci[1]:.4f}]")
    edu_bc = [get_domain_correct(r, "education") for r in baseline]
    edu_ac = [get_domain_correct(r, "education") for r in after]
    edu_a_only = sum(1 for i in range(len(baseline)) if edu_bc[i] and not edu_ac[i])
    edu_b_only = sum(1 for i in range(len(baseline)) if not edu_bc[i] and edu_ac[i])
    mcnemar_edu = mcnemar_2x2(edu_a_only, edu_b_only)
    print(f"  McNemar: a_only={edu_a_only}, b_only={edu_b_only}, p={mcnemar_edu['p_value']:.10f}")

    # 3. medical_recall
    med_before = sum(get_domain_correct(r, "medical") for r in baseline)
    med_after = sum(get_domain_correct(r, "medical") for r in after)
    med_total = domain_expected_count["medical"]
    med_before_r = med_before / med_total
    med_after_r = med_after / med_total
    print(f"\n  medical_recall: {med_before_r:.4f} -> {med_after_r:.4f} (delta={med_after_r-med_before_r:+.4f})")
    med_before_ci = wilson_ci(med_before, med_total)
    med_after_ci = wilson_ci(med_after, med_total)
    print(f"  Wilson 95% CI Before: [{med_before_ci[0]:.4f}, {med_before_ci[1]:.4f}]")
    print(f"  Wilson 95% CI After:  [{med_after_ci[0]:.4f}, {med_after_ci[1]:.4f}]")
    med_bc = [get_domain_correct(r, "medical") for r in baseline]
    med_ac = [get_domain_correct(r, "medical") for r in after]
    med_a_only = sum(1 for i in range(len(baseline)) if med_bc[i] and not med_ac[i])
    med_b_only = sum(1 for i in range(len(baseline)) if not med_bc[i] and med_ac[i])
    mcnemar_med = mcnemar_2x2(med_a_only, med_b_only)
    print(f"  McNemar: a_only={med_a_only}, b_only={med_b_only}, p={mcnemar_med['p_value']:.10f}")

    # 4. argmax flip rate
    flip_rate, flip_count = compute_argmax_flip_rate(baseline, after)
    print(f"\n  argmax_flip_rate: {flip_rate:.4f} ({flip_count}/{len(baseline)})")
    print(f"  Single-lever check (<15%): {'PASS' if flip_rate < 0.15 else 'FAIL'}")

    # 5. ECE
    ece_before = compute_ece(baseline)
    ece_after = compute_ece(after)
    print(f"\n  ECE: {ece_before:.6f} -> {ece_after:.6f} (delta={ece_after-ece_before:+.6f})")

    # 6. Per-domain precision/recall with Wilson CIs
    print(f"\n  {'Domain':20s} {'Recall(B->A)':>14s} {'Prec(B->A)':>14s} {'Recall CI A':>16s}")
    per_domain = {}
    for domain in domains:
        n = domain_expected_count[domain]
        r_bc = sum(get_domain_correct(r, domain) for r in baseline)
        r_ac = sum(get_domain_correct(r, domain) for r in after)
        s_b = sum(1 for r in baseline if r["selected_domain"] == domain)
        s_a = sum(1 for r in after if r["selected_domain"] == domain)
        p_bc = r_bc  # precision numerator = recall numerator (same rows)
        p_ac = r_ac
        rec_b = r_bc / n if n > 0 else 0
        rec_a = r_ac / n if n > 0 else 0
        prec_b = p_bc / s_b if s_b > 0 else 0
        prec_a = p_ac / s_a if s_a > 0 else 0
        rec_ci_b = wilson_ci(r_bc, n)
        rec_ci_a = wilson_ci(r_ac, n)
        per_domain[domain] = {
            "recall_before": rec_b, "recall_after": rec_a, "recall_delta": rec_a - rec_b,
            "recall_ci_before": rec_ci_b, "recall_ci_after": rec_ci_a,
            "precision_before": prec_b, "precision_after": prec_a, "precision_delta": prec_a - prec_b,
            "recall_n": n,
            "precision_selected_before": s_b, "precision_selected_after": s_a,
            "recall_correct_before": r_bc, "recall_correct_after": r_ac,
        }
        print(f"    {domain:20s} {rec_b:.4f}->{rec_a:.4f} {prec_b:.4f}->{prec_a:.4f} [{rec_ci_a[0]:.4f},{rec_ci_a[1]:.4f}]")

    # 7. Per-domain McNemar tests
    print(f"\n  Per-domain recall McNemar:")
    per_domain_mcnemar = {}
    for domain in domains:
        bc = [get_domain_correct(r, domain) for r in baseline]
        ac = [get_domain_correct(r, domain) for r in after]
        ao = sum(1 for i in range(len(baseline)) if bc[i] and not ac[i])
        bo = sum(1 for i in range(len(baseline)) if not bc[i] and ac[i])
        mc = mcnemar_2x2(ao, bo)
        per_domain_mcnemar[domain] = mc
        sig = "*" if mc["p_value"] < 0.05 else ""
        print(f"    {domain:20s}: a_only={ao:3d}, b_only={bo:3d}, chi2={mc['chi2']:.4f}, p={mc['p_value']:.6f} {sig}")

    # 8. BH correction for 20 per-domain metrics
    p_values = []
    metric_names = []
    for domain in domains:
        recall_mc = per_domain_mcnemar[domain]
        p_values.append(recall_mc["p_value"])
        metric_names.append(f"{domain}_recall")
        pd = per_domain[domain]
        a = pd["precision_selected_before"]
        b = pd["precision_selected_before"] - pd["recall_correct_before"]
        c = pd["precision_selected_after"]
        d = pd["precision_selected_after"] - pd["recall_correct_after"]
        if a + b == 0 or c + d == 0:
            p_prec = 1.0
        else:
            p_prec = fisher_exact_2x2(pd["recall_correct_before"], b, pd["recall_correct_after"], d)
        p_values.append(p_prec)
        metric_names.append(f"{domain}_precision")

    bh_results = bh_correction(p_values)
    regressions = []
    improvements = []
    print(f"\n  BH-corrected 20 per-domain metrics:")
    print(f"  {'Metric':25s} {'p':>12s} {'q':>12s} {'Dir':>10s} {'Sig':>6s}")
    print(f"  {'-'*65}")
    for i, domain in enumerate(domains):
        recall_idx = i * 2
        precision_idx = i * 2 + 1
        for idx, suffix in [(recall_idx, "recall"), (precision_idx, "precision")]:
            p, q = bh_results[idx]
            delta = per_domain[domain][f"recall_delta" if suffix == "recall" else "precision_delta"]
            direction = "improvement" if delta > 0 else "regression"
            sig = "YES" if q < 0.05 else "no"
            print(f"    {metric_names[idx]:25s} {p:>12.6e} {q:>12.6e} {direction:>10s} {sig:>6s}")
            if q < 0.05:
                if delta < 0:
                    regressions.append((metric_names[idx], p, q, delta))
                else:
                    improvements.append((metric_names[idx], p, q, delta))
    print(f"\n  BH-significant regressions: {len(regressions)}")
    for name, p, q, d in regressions:
        print(f"    {name}: p={p:.6e}, q={q:.6e}, delta={d:+.4f}")
    print(f"  BH-significant improvements: {len(improvements)}")
    for name, p, q, d in improvements:
        print(f"    {name}: p={p:.6e}, q={q:.6e}, delta={d:+.4f}")

    # 9. Flip direction analysis
    print(f"\n  Flip transitions (top 15):")
    transitions = compute_flip_transitions(baseline, after)
    for trans, count in sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"    {trans:50s}: {count}")

    # 10. Education/Medical flip analysis
    flip_info = compute_edu_med_flips(baseline, after)
    print(f"\n  Education flips: correct->wrong={flip_info['edu_ctw']}, wrong->correct={flip_info['edu_wtc']}")
    if flip_info['edu_flips_from']:
        print(f"    Wrong->Correct sources: {dict(sorted(flip_info['edu_flips_from'].items(), key=lambda x: x[1], reverse=True))}")
    if flip_info['edu_flips_to']:
        print(f"    Correct->Wrong destinations: {dict(sorted(flip_info['edu_flips_to'].items(), key=lambda x: x[1], reverse=True))}")
    print(f"  Medical flips: correct->wrong={flip_info['med_ctw']}, wrong->correct={flip_info['med_wtc']}")
    if flip_info['med_flips_from']:
        print(f"    Wrong->Correct sources: {dict(sorted(flip_info['med_flips_from'].items(), key=lambda x: x[1], reverse=True))}")
    if flip_info['med_flips_to']:
        print(f"    Correct->Wrong destinations: {dict(sorted(flip_info['med_flips_to'].items(), key=lambda x: x[1], reverse=True))}")

    # 11. Success criteria check
    print(f"\n  {'='*50}")
    print(f"  SUCCESS CRITERIA CHECK")
    print(f"  {'='*50}")
    edu_gt_medical = edu_after_r > 0.5112
    bh_reg_zero = len(regressions) == 0
    flip_lt15 = flip_rate < 0.15
    mcnemar_top1_p = mcnemar_top1["p_value"]
    mcnemar_p_ok = mcnemar_top1_p >= 0.05
    print(f"    education_recall > 0.5112: {'PASS' if edu_gt_medical else 'FAIL'} ({edu_after_r:.4f})")
    print(f"    BH-significant regressions == 0: {'PASS' if bh_reg_zero else 'FAIL'} ({len(regressions)} found)")
    print(f"    argmax_flip_rate < 15%: {'PASS' if flip_lt15 else 'FAIL'} ({flip_rate:.4f})")
    print(f"    McNemar top1 p >= 0.05: {'PASS' if mcnemar_p_ok else 'FAIL'} (p={mcnemar_top1_p:.6f})")
    all_pass = edu_gt_medical and bh_reg_zero and flip_lt15 and mcnemar_p_ok
    print(f"    OVERALL: {'PASS (all 4 conditions met) - ADOPTED' if all_pass else 'FAIL'}")

    print(f"\n  SUMMARY:")
    print(f"    top1_accuracy:         {before_acc:.4f} -> {after_acc:.4f} ({after_acc-before_acc:+.4f})")
    print(f"    education_recall:      {edu_before_r:.4f} -> {edu_after_r:.4f} ({edu_after_r-edu_before_r:+.4f})")
    print(f"    medical_recall:        {med_before_r:.4f} -> {med_after_r:.4f} ({med_after_r-med_before_r:+.4f})")
    print(f"    argmax_flip_rate:      {flip_rate:.4f} ({flip_count}/{len(baseline)})")
    print(f"    ECE:                   {ece_before:.6f} -> {ece_after:.6f} ({ece_after-ece_before:+.6f})")
    print(f"    BH-significant regressions: {len(regressions)}")
    print(f"    BH-significant improvements: {len(improvements)}")
    print(f"    McNemar top1 p-value:  {mcnemar_top1['p_value']:.10f}")
    print(f"    McNemar edu p-value:   {mcnemar_edu['p_value']:.10f}")
    print(f"    McNemar med p-value:   {mcnemar_med['p_value']:.10f}")
