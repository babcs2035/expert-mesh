#!/usr/bin/env python3
"""Independent metric computation for Iter43 analysis.

Reads iter31_calibrated_predictions.jsonl (baseline) and
iter43_projection_head_calibrated_predictions.jsonl (after),
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

before = load_predictions("results/iter31_calibrated_predictions.jsonl")
after = load_predictions("results/iter43_projection_head_calibrated_predictions.jsonl")

assert len(before) == len(after) == 1600, f"Row count mismatch: {len(before)}, {len(after)}"

# Index by id for paired comparison
before_map = {r["id"]: r for r in before}
after_map = {r["id"]: r for r in after}

# ---- Helper functions ----
def is_correct(r):
    """Check if selected_domain is in expected_domains."""
    return r["selected_domain"] in r["expected_domains"]

def is_correct_domain(r, domain):
    """Check if the selected domain matches the given domain."""
    return r["selected_domain"] == domain

def get_domain_correct(r, domain):
    """For domain recall: 1 if selected_domain == domain and domain in expected_domains."""
    return 1 if (r["selected_domain"] == domain and domain in r["expected_domains"]) else 0

def get_domain_precision(r, domain):
    """For domain precision: 1 if selected_domain == domain and domain in expected_domains."""
    return 1 if (r["selected_domain"] == domain and domain in r["expected_domains"]) else 0

def wilson_ci(success, n, z=1.96):
    """Wilson score interval."""
    if n == 0:
        return (0.0, 0.0)
    p = success / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    margin = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n) / denom
    return (max(0, center - margin), min(1, center + margin))

def mcnemar_2x2(a_only, b_only):
    """McNemar test (chi-squared approximation)."""
    if (a_only + b_only) == 0:
        return {"discordant_a_only": a_only, "discordant_b_only": b_only,
                "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(a_only - b_only) - 1)**2 / (a_only + b_only)  # Yates correction
    # Approximate p-value from chi-squared(1)
    p_value = math.exp(-chi2 / 2)  # Rough approximation for chi2(1)
    # Better: use complementary error function approximation
    # For chi2(1), p = 2 * (1 - Phi(sqrt(chi2)))
    z_val = math.sqrt(chi2) if chi2 > 0 else 0
    # Approximation of 1 - Phi(z) for z > 0
    t = 1.0 / (1.0 + 0.2316419 * z_val) if z_val > 0 else 1.0
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    p_one_tail = d * math.exp(-z_val*z_val/2) * (t*(0.319381530 + t*(-0.356563782 + t*(1.781477937 + t*(-1.821255978 + t*1.330274429)))))
    p_value = 2 * p_one_tail
    return {"discordant_a_only": a_only, "discordant_b_only": b_only,
            "chi2": round(chi2, 6), "p_value": round(p_value, 10)}

def bh_correction(p_values):
    """Benjamini-Hochberg correction. Returns list of (original_idx, p, q)."""
    indexed = [(p, i) for i, p in enumerate(p_values)]
    indexed.sort(key=lambda x: x[0])
    m = len(p_values)
    results = [None] * m
    for rank, (p, orig_idx) in enumerate(indexed, 1):
        q = p * m / rank
        q = min(q, 1.0)
        # Step-up: ensure monotonicity from bottom
        results[orig_idx] = (p, q)
    # Apply monotonicity constraint (from largest q to smallest)
    for i in range(m - 2, -1, -1):
        orig_idx = indexed[i+1][1]
        prev_idx = indexed[i][1]
        if results[prev_idx][1] > results[orig_idx][1]:
            results[prev_idx] = (results[prev_idx][0], results[orig_idx][1])
    return results

# =====================================================================
# 1. top1_accuracy (compound-included)
# =====================================================================
print("=" * 70)
print("1. TOP1_ACCURACY (compound-included, 1600 questions)")
print("=" * 70)

before_correct = sum(1 for r in before if is_correct(r))
after_correct = sum(1 for r in after if is_correct(r))

before_acc = before_correct / len(before)
after_acc = after_correct / len(after)

print(f"  Before (Iter31): {before_correct}/{len(before)} = {before_acc:.4f}")
print(f"  After  (Iter43): {after_correct}/{len(after)} = {after_acc:.4f}")
print(f"  Delta: {after_acc - before_acc:+.4f}")

# McNemar test for top1_accuracy
top1_before = [is_correct(r) for r in before]
top1_after = [is_correct(r) for r in after]

a_only = sum(1 for i in range(len(before)) if top1_before[i] and not top1_after[i])
b_only = sum(1 for i in range(len(before)) if not top1_before[i] and top1_after[i])

mcnemar_top1 = mcnemar_2x2(a_only, b_only)
print(f"\n  McNemar test:")
print(f"    A only (before correct, after wrong): {a_only}")
print(f"    B only (before wrong, after correct): {b_only}")
print(f"    chi2: {mcnemar_top1['chi2']:.6f}")
print(f"    p-value: {mcnemar_top1['p_value']:.10f}")
print(f"    Significant at alpha=0.05: {'YES' if mcnemar_top1['p_value'] < 0.05 else 'NO'}")

# =====================================================================
# 2. education_recall (compound-included)
# =====================================================================
print("\n" + "=" * 70)
print("2. EDUCATION_RECALL (compound-included)")
print("=" * 70)

edu_before = sum(get_domain_correct(r, "education") for r in before)
edu_after = sum(get_domain_correct(r, "education") for r in after)
edu_total = sum(1 for r in before if "education" in r["expected_domains"])

edu_before_r = edu_before / edu_total
edu_after_r = edu_after / edu_total

print(f"  Total education questions: {edu_total}")
print(f"  Before: {edu_before}/{edu_total} = {edu_before_r:.4f}")
print(f"  After:  {edu_after}/{edu_total} = {edu_after_r:.4f}")
print(f"  Delta: {edu_after_r - edu_before_r:+.4f}")

# Wilson CI
edu_before_ci = wilson_ci(edu_before, edu_total)
edu_after_ci = wilson_ci(edu_after, edu_total)
print(f"  Wilson 95% CI Before: [{edu_before_ci[0]:.4f}, {edu_before_ci[1]:.4f}]")
print(f"  Wilson 95% CI After:  [{edu_after_ci[0]:.4f}, {edu_after_ci[1]:.4f}]")

# McNemar for education recall
edu_before_correct = [get_domain_correct(r, "education") for r in before]
edu_after_correct = [get_domain_correct(r, "education") for r in after]

edu_a_only = sum(1 for i in range(len(before)) if edu_before_correct[i] and not edu_after_correct[i])
edu_b_only = sum(1 for i in range(len(before)) if not edu_before_correct[i] and edu_after_correct[i])

mcnemar_edu = mcnemar_2x2(edu_a_only, edu_b_only)
print(f"\n  McNemar test:")
print(f"    A only (before correct, after wrong): {edu_a_only}")
print(f"    B only (before wrong, after correct): {edu_b_only}")
print(f"    chi2: {mcnemar_edu['chi2']:.6f}")
print(f"    p-value: {mcnemar_edu['p_value']:.10f}")

# =====================================================================
# 3. medical_recall (compound-included)
# =====================================================================
print("\n" + "=" * 70)
print("3. MEDICAL_RECALL (compound-included)")
print("=" * 70)

med_before = sum(get_domain_correct(r, "medical") for r in before)
med_after = sum(get_domain_correct(r, "medical") for r in after)
med_total = sum(1 for r in before if "medical" in r["expected_domains"])

med_before_r = med_before / med_total
med_after_r = med_after / med_total

print(f"  Total medical questions: {med_total}")
print(f"  Before: {med_before}/{med_total} = {med_before_r:.4f}")
print(f"  After:  {med_after}/{med_total} = {med_after_r:.4f}")
print(f"  Delta: {med_after_r - med_before_r:+.4f}")

med_before_ci = wilson_ci(med_before, med_total)
med_after_ci = wilson_ci(med_after, med_total)
print(f"  Wilson 95% CI Before: [{med_before_ci[0]:.4f}, {med_before_ci[1]:.4f}]")
print(f"  Wilson 95% CI After:  [{med_after_ci[0]:.4f}, {med_after_ci[1]:.4f}]")

# McNemar for medical recall
med_before_correct = [get_domain_correct(r, "medical") for r in before]
med_after_correct = [get_domain_correct(r, "medical") for r in after]

med_a_only = sum(1 for i in range(len(before)) if med_before_correct[i] and not med_after_correct[i])
med_b_only = sum(1 for i in range(len(before)) if not med_before_correct[i] and med_after_correct[i])

mcnemar_med = mcnemar_2x2(med_a_only, med_b_only)
print(f"\n  McNemar test:")
print(f"    A only (before correct, after wrong): {med_a_only}")
print(f"    B only (before wrong, after correct): {med_b_only}")
print(f"    chi2: {mcnemar_med['chi2']:.6f}")
print(f"    p-value: {mcnemar_med['p_value']:.10f}")

# =====================================================================
# 4. argmax flip rate (row-level argmax change)
# =====================================================================
print("\n" + "=" * 70)
print("4. ARGMAX FLIP RATE")
print("=" * 70)

flipped = 0
flipped_details = []
for i, (rb, ra) in enumerate(zip(before, after)):
    if rb["selected_domain"] != ra["selected_domain"]:
        flipped += 1
        flipped_details.append({
            "id": rb["id"],
            "before": rb["selected_domain"],
            "after": ra["selected_domain"],
            "before_conf": rb["confidence"],
            "after_conf": ra["confidence"]
        })

flip_rate = flipped / len(before)
print(f"  Flipped rows: {flipped}/{len(before)} = {flip_rate:.4f} ({flip_rate*100:.2f}%)")
print(f"  Threshold (<15%): {'PASS' if flip_rate < 0.15 else 'FAIL'}")

# =====================================================================
# 5. Probability change analysis
# =====================================================================
print("\n" + "=" * 70)
print("5. PROBABILITY CHANGE ANALYSIS")
print("=" * 70)

max_deltas = []
mean_max_deltas = []
changes_over_01 = 0
changes_over_05 = 0
changes_over_10 = 0

for rb, ra in zip(before, after):
    # Max delta across all domains
    deltas = {}
    for domain in rb["probabilities"]:
        d = abs(ra["probabilities"][domain] - rb["probabilities"][domain])
        deltas[domain] = d
    max_delta = max(deltas.values())
    max_deltas.append(max_delta)

    # Which domain had the max delta
    max_domain = max(deltas, key=deltas.get)
    mean_max_deltas.append((max_domain, max_delta))

    if max_delta > 0.1:
        changes_over_01 += 1
    if max_delta > 0.5:
        changes_over_05 += 1
    if max_delta > 1.0:
        changes_over_10 += 1

mean_max_delta = sum(max_deltas) / len(max_deltas)
max_max_delta = max(max_deltas)

print(f"  Mean max delta: {mean_max_delta:.4f}")
print(f"  Max max delta:  {max_max_delta:.4f}")
print(f"  Rows with max delta > 0.1: {changes_over_01}/{len(before)} ({changes_over_01/len(before)*100:.1f}%)")
print(f"  Rows with max delta > 0.5: {changes_over_05}/{len(before)} ({changes_over_05/len(before)*100:.1f}%)")
print(f"  Rows with max delta > 1.0: {changes_over_10}/{len(before)} ({changes_over_10/len(before)*100:.1f}%)")

# Top 20 domains with largest mean delta
domain_mean_deltas = defaultdict(list)
for rb, ra in zip(before, after):
    for domain in rb["probabilities"]:
        d = abs(ra["probabilities"][domain] - rb["probabilities"][domain])
        domain_mean_deltas[domain].append(d)

domain_mean_delta_sorted = sorted(
    [(d, sum(vals)/len(vals)) for d, vals in domain_mean_deltas.items()],
    key=lambda x: x[1], reverse=True
)
print("\n  Domain mean delta ranking:")
for domain, mean_d in domain_mean_delta_sorted:
    print(f"    {domain:20s}: mean_delta={mean_d:.4f}")

# =====================================================================
# 6. Per-domain precision/recall with Wilson CIs
# =====================================================================
print("\n" + "=" * 70)
print("6. PER-DOMAIN PRECISION/RECALL (compound-included)")
print("=" * 70)

domains = ["business_economics", "computer_science", "education", "general",
           "history_culture", "legal", "mathematics", "medical",
           "natural_science", "social_science"]

# Count how many questions each domain appears as expected
domain_expected_count = defaultdict(int)
for r in before:
    for d in r["expected_domains"]:
        domain_expected_count[d] += 1

per_domain = {}
for domain in domains:
    # Recall: selected_domain == domain AND domain in expected_domains
    r_before_correct = sum(get_domain_correct(r, domain) for r in before)
    r_after_correct = sum(get_domain_correct(r, domain) for r in after)
    n = domain_expected_count[domain]

    # Precision: among rows where selected_domain == domain, how many are correct
    selected_before = sum(1 for r in before if r["selected_domain"] == domain)
    selected_after = sum(1 for r in after if r["selected_domain"] == domain)
    p_before_correct = sum(get_domain_precision(r, domain) for r in before)
    p_after_correct = sum(get_domain_precision(r, domain) for r in after)

    recall_before = r_before_correct / n if n > 0 else 0
    recall_after = r_after_correct / n if n > 0 else 0
    precision_before = p_before_correct / selected_before if selected_before > 0 else 0
    precision_after = p_after_correct / selected_after if selected_after > 0 else 0

    recall_ci_before = wilson_ci(r_before_correct, n)
    recall_ci_after = wilson_ci(r_after_correct, n)
    precision_ci_before = wilson_ci(p_before_correct, selected_before)
    precision_ci_after = wilson_ci(p_after_correct, selected_after)

    per_domain[domain] = {
        "recall_before": recall_before,
        "recall_after": recall_after,
        "recall_delta": recall_after - recall_before,
        "recall_n": n,
        "recall_correct_before": r_before_correct,
        "recall_correct_after": r_after_correct,
        "recall_ci_before": recall_ci_before,
        "recall_ci_after": recall_ci_after,
        "precision_before": precision_before,
        "precision_after": precision_after,
        "precision_delta": precision_after - precision_before,
        "precision_selected_before": selected_before,
        "precision_selected_after": selected_after,
        "precision_ci_before": precision_ci_before,
        "precision_ci_after": precision_ci_after,
    }

    ci_lower_dropped = recall_ci_after[0] < recall_ci_before[0]

    print(f"\n  {domain}:")
    print(f"    Recall: {recall_before:.4f} -> {recall_after:.4f} (delta={recall_after-recall_before:+.4f})")
    print(f"    Recall CI: [{recall_ci_before[0]:.4f}, {recall_ci_before[1]:.4f}] -> [{recall_ci_after[0]:.4f}, {recall_ci_after[1]:.4f}]")
    print(f"    Precision: {precision_before:.4f} -> {precision_after:.4f} (delta={precision_after-precision_before:+.4f})")
    print(f"    Precision CI: [{precision_ci_before[0]:.4f}, {precision_ci_before[1]:.4f}] -> [{precision_ci_after[0]:.4f}, {precision_ci_after[1]:.4f}]")
    print(f"    CI lower bound dropped: {'YES' if ci_lower_dropped else 'NO'}")

# =====================================================================
# 7. McNemar tests for per-domain recall
# =====================================================================
print("\n" + "=" * 70)
print("7. PER-DOMAIN RECALL McNemar TESTS")
print("=" * 70)

per_domain_mcnemar = {}
for domain in domains:
    before_correct = [get_domain_correct(r, domain) for r in before]
    after_correct = [get_domain_correct(r, domain) for r in after]

    a_only = sum(1 for i in range(len(before)) if before_correct[i] and not after_correct[i])
    b_only = sum(1 for i in range(len(before)) if not before_correct[i] and after_correct[i])

    mc = mcnemar_2x2(a_only, b_only)
    per_domain_mcnemar[domain] = mc

    print(f"  {domain:20s}: a_only={a_only:3d}, b_only={b_only:3d}, chi2={mc['chi2']:.4f}, p={mc['p_value']:.6f}")

# =====================================================================
# 8. BH correction for 20 per-domain metrics
# =====================================================================
print("\n" + "=" * 70)
print("8. BH CORRECTION FOR 20 PER-DOMAIN METRICS")
print("=" * 70)

# For each domain, collect recall p-value (from McNemar) and precision p-value (from Fisher)
# Recall: use McNemar p-values
# Precision: use Fisher exact test (two-sample, unpaired)
# We need to compute Fisher exact test for precision

from itertools import combinations

def fisher_exact_2x2(a, b, c, d):
    """Fisher exact test for 2x2 table:
    | a  b |
    | c  d |
    Returns two-tailed p-value.
    """
    n = a + b + c + d
    if n == 0:
        return 1.0

    # Use the hypergeometric distribution
    def hypergeom_p(k, N, K, n):
        """P(X=k) for hypergeometric(N, K, n)."""
        if k < 0 or k > min(K, n) or k > n or k < N - K - n + n:
            return 0.0
        # Use log gamma for numerical stability
        log_p = (math.lgamma(K+1) + math.lgamma(n+1) + math.lgamma(N-K+1) + math.lgamma(N-n+1)
                 - math.lgamma(N+1) - math.lgamma(k+1) - math.lgamma(K-k+1)
                 - math.lgamma(n-k+1) - math.lgamma(N-K-n+k+1))
        return math.exp(log_p)

    # Table:
    # | correct_before  wrong_before |
    # | correct_after   wrong_after  |
    # a = correct_before & correct_after
    # b = correct_before & wrong_after
    # c = wrong_before & correct_after
    # d = wrong_before & wrong_after
    # But for precision, we need unpaired two-sample:
    # Group 1 (before): selected_before rows, p_before_correct correct
    # Group 2 (after): selected_after rows, p_after_correct correct
    # This is a standard two-proportion test, not paired.
    # Use Fisher exact on:
    # | p_before_correct  selected_before - p_before_correct |
    # | p_after_correct   selected_after - p_after_correct   |
    table = [[a, b], [c, d]]

    # Compute exact p-value by summing all tables with <= probability
    row_sums = [a+b, c+d]
    col_sums = [a+c, b+d]
    total = a+b+c+d

    # Generate all possible tables with same marginals
    min_k = max(0, row_sums[0] + col_sums[0] - total)
    max_k = min(row_sums[0], col_sums[0])

    obs_prob = hypergeom_p(a, total, col_sums[0], row_sums[0])

    two_tail_p = 0.0
    for k in range(min_k, max_k + 1):
        prob = hypergeom_p(k, total, col_sums[0], row_sums[0])
        if prob <= obs_prob + 1e-15:
            two_tail_p += prob

    return min(two_tail_p, 1.0)

# Collect all 20 p-values
p_values = []
metric_names = []

for domain in domains:
    # Recall McNemar p-value
    recall_mc = per_domain_mcnemar[domain]
    p_values.append(recall_mc["p_value"])
    metric_names.append(f"{domain}_recall")

    # Precision: Fisher exact (two-sample, unpaired)
    pd = per_domain[domain]
    a = pd["precision_selected_before"]  # selected_before
    b = pd["precision_selected_before"] - pd["recall_correct_before"]  # selected but not correct
    c = pd["precision_selected_after"]
    d = pd["precision_selected_after"] - pd["recall_correct_after"]

    # Wait, for precision we need:
    # Before: selected_before rows, precision_selected_before * selected_before correct
    # After: selected_after rows, precision_selected_after * selected_after correct
    # Fisher: | correct_before  wrong_before |
    #         | correct_after   wrong_after  |
    prec_before_correct = pd["recall_correct_before"]  # same as precision correct
    prec_before_wrong = pd["precision_selected_before"] - prec_before_correct
    prec_after_correct = pd["recall_correct_after"]
    prec_after_wrong = pd["precision_selected_after"] - prec_after_correct

    if prec_before_correct + prec_before_wrong == 0 or prec_after_correct + prec_after_wrong == 0:
        p_prec = 1.0
    else:
        p_prec = fisher_exact_2x2(
            prec_before_correct, prec_before_wrong,
            prec_after_correct, prec_after_wrong
        )

    p_values.append(p_prec)
    metric_names.append(f"{domain}_precision")

# Apply BH correction
bh_results = bh_correction(p_values)

print(f"\n  {'Metric':25s} {'p':>12s} {'q':>12s} {'Direction':>12s} {'Significant':>12s}")
print(f"  {'-'*70}")

regressions = []
improvements = []

for i, domain in enumerate(domains):
    recall_idx = i * 2
    precision_idx = i * 2 + 1

    # Recall
    p_rec, q_rec = bh_results[recall_idx]
    delta_rec = per_domain[domain]["recall_delta"]
    rec_dir = "improvement" if delta_rec > 0 else "regression"
    rec_sig = "YES" if q_rec < 0.05 else "no"
    print(f"  {metric_names[recall_idx]:25s} {p_rec:>12.6e} {q_rec:>12.6e} {rec_dir:>12s} {rec_sig:>12s}")

    if q_rec < 0.05:
        if delta_rec < 0:
            regressions.append((metric_names[recall_idx], p_rec, q_rec, delta_rec))
        else:
            improvements.append((metric_names[recall_idx], p_rec, q_rec, delta_rec))

    # Precision
    p_prec, q_prec = bh_results[precision_idx]
    delta_prec = per_domain[domain]["precision_delta"]
    prec_dir = "improvement" if delta_prec > 0 else "regression"
    prec_sig = "YES" if q_prec < 0.05 else "no"
    print(f"  {metric_names[precision_idx]:25s} {p_prec:>12.6e} {q_prec:>12.6e} {prec_dir:>12s} {prec_sig:>12s}")

    if q_prec < 0.05:
        if delta_prec < 0:
            regressions.append((metric_names[precision_idx], p_prec, q_prec, delta_prec))
        else:
            improvements.append((metric_names[precision_idx], p_prec, q_prec, delta_prec))

print(f"\n  BH-significant regressions: {len(regressions)}")
for name, p, q, d in regressions:
    print(f"    {name}: p={p:.6e}, q={q:.6e}, delta={d:+.4f}")

print(f"\n  BH-significant improvements: {len(improvements)}")
for name, p, q, d in improvements:
    print(f"    {name}: p={p:.6e}, q={q:.6e}, delta={d:+.4f}")

# =====================================================================
# 9. Flip direction analysis (where do flipped predictions go?)
# =====================================================================
print("\n" + "=" * 70)
print("9. FLIP DIRECTION ANALYSIS")
print("=" * 70)

# For flipped rows, track: before domain -> after domain
flip_transitions = defaultdict(int)
for rb, ra in zip(before, after):
    if rb["selected_domain"] != ra["selected_domain"]:
        key = f"{rb['selected_domain']} -> {ra['selected_domain']}"
        flip_transitions[key] += 1

# Top 20 transitions
sorted_transitions = sorted(flip_transitions.items(), key=lambda x: x[1], reverse=True)
print(f"  Top 20 flip transitions ({len(sorted_transitions)} total):")
for trans, count in sorted_transitions[:20]:
    print(f"    {trans:50s}: {count}")

# Education flip analysis
edu_flips_to = defaultdict(int)
edu_flips_from = defaultdict(int)
edu_flip_correct_to_wrong = 0
edu_flip_wrong_to_correct = 0

for rb, ra in zip(before, after):
    if "education" in rb["expected_domains"]:
        if rb["selected_domain"] == "education" and ra["selected_domain"] != "education":
            edu_flip_correct_to_wrong += 1
            edu_flips_from[ra["selected_domain"]] += 1
        elif rb["selected_domain"] != "education" and ra["selected_domain"] == "education":
            edu_flip_wrong_to_correct += 1
            edu_flips_to[rb["selected_domain"]] += 1

print(f"\n  Education flip analysis:")
print(f"    Correct -> Wrong: {edu_flip_correct_to_wrong}")
print(f"    Wrong -> Correct: {edu_flip_wrong_to_correct}")
if edu_flip_correct_to_wrong > 0:
    print(f"    Where do wrong->correct flips come from:")
    for d, c in sorted(edu_flips_from.items(), key=lambda x: x[1], reverse=True):
        print(f"      -> {d}: {c}")
if edu_flip_wrong_to_correct > 0:
    print(f"    Where does correct->wrong flip go:")
    for d, c in sorted(edu_flips_to.items(), key=lambda x: x[1], reverse=True):
        print(f"      {d}: {c}")

# Medical flip analysis
med_flip_correct_to_wrong = 0
med_flip_wrong_to_correct = 0
med_flips_from = defaultdict(int)
med_flips_to = defaultdict(int)

for rb, ra in zip(before, after):
    if "medical" in rb["expected_domains"]:
        if rb["selected_domain"] == "medical" and ra["selected_domain"] != "medical":
            med_flip_correct_to_wrong += 1
            med_flips_from[ra["selected_domain"]] += 1
        elif rb["selected_domain"] != "medical" and ra["selected_domain"] == "medical":
            med_flip_wrong_to_correct += 1
            med_flips_to[rb["selected_domain"]] += 1

print(f"\n  Medical flip analysis:")
print(f"    Correct -> Wrong: {med_flip_correct_to_wrong}")
print(f"    Wrong -> Correct: {med_flip_wrong_to_correct}")
if med_flip_correct_to_wrong > 0:
    print(f"    Where do wrong->correct flips come from:")
    for d, c in sorted(med_flips_from.items(), key=lambda x: x[1], reverse=True):
        print(f"      -> {d}: {c}")
if med_flip_wrong_to_correct > 0:
    print(f"    Where does correct->wrong flip go:")
    for d, c in sorted(med_flips_to.items(), key=lambda x: x[1], reverse=True):
        print(f"      {d}: {c}")

# =====================================================================
# 10. ECE computation
# =====================================================================
print("\n" + "=" * 70)
print("10. ECE (Expected Calibration Error)")
print("=" * 70)

def compute_ece(predictions, n_bins=10):
    """Compute ECE with equal-width bins."""
    bins = [[] for _ in range(n_bins)]
    for r in predictions:
        conf = r["confidence"]
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

ece_before = compute_ece(before)
ece_after = compute_ece(after)

print(f"  Before (Iter31): {ece_before:.6f}")
print(f"  After  (Iter43): {ece_after:.6f}")
print(f"  Delta: {ece_after - ece_before:+.6f}")

# =====================================================================
# 11. Summary
# =====================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  top1_accuracy:         {before_acc:.4f} -> {after_acc:.4f} ({after_acc-before_acc:+.4f})")
print(f"  education_recall:      {edu_before_r:.4f} -> {edu_after_r:.4f} ({edu_after_r-edu_before_r:+.4f})")
print(f"  medical_recall:        {med_before_r:.4f} -> {med_after_r:.4f} ({med_after_r-med_before_r:+.4f})")
print(f"  argmax_flip_rate:      {flip_rate:.4f} ({flip_rate*100:.2f}%)")
print(f"  ECE:                   {ece_before:.6f} -> {ece_after:.6f} ({ece_after-ece_before:+.6f})")
print(f"  BH-significant regressions: {len(regressions)}")
print(f"  BH-significant improvements: {len(improvements)}")
print(f"  McNemar top1 p-value:  {mcnemar_top1['p_value']:.10f}")
print(f"  McNemar edu p-value:   {mcnemar_edu['p_value']:.10f}")
print(f"  McNemar med p-value:   {mcnemar_med['p_value']:.10f}")
print(f"  Mean max delta:        {mean_max_delta:.4f}")
print(f"  Max max delta:         {max_max_delta:.4f}")
print(f"  Rows with delta > 0.1: {changes_over_01}/{len(before)} ({changes_over_01/len(before)*100:.1f}%)")

# Single-lever check
print(f"\n  SINGLE-LEVER CHECK:")
print(f"    argmax_flip_rate < 15%: {'PASS' if flip_rate < 0.15 else 'FAIL'}")
print(f"    education_recall > medical_recall baseline (0.5112): {'PASS' if edu_after_r > 0.5112 else 'FAIL'}")
print(f"    BH-significant regressions == 0: {'PASS' if len(regressions) == 0 else 'FAIL'}")
print(f"    McNemar top1 p >= 0.05: {'PASS' if mcnemar_top1['p_value'] >= 0.05 else 'FAIL'}")
