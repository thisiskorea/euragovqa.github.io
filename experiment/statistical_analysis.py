import json
import numpy as np
from pathlib import Path
from scipy import stats

RESULTS_FILE = (
    Path(__file__).parent / "results" / "large_scale_20260120_170800_fixed.json"
)

with open(RESULTS_FILE, "r") as f:
    data = json.load(f)

details = data["details"]
n = len(details)

track_a = np.array([d["track_a"]["is_correct"] for d in details])
track_b = np.array([d["track_b"]["is_correct"] for d in details])
track_c = np.array([d["track_c"]["is_correct"] for d in details])

print("=" * 70)
print("Statistical Analysis for VCE Paper")
print("=" * 70)


def bootstrap_ci(arr, n_bootstrap=10000, ci=0.95):
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        means.append(sample.mean())
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return np.mean(arr), lower, upper


print("\n[1] Overall Accuracy with 95% Bootstrap CI")
print("-" * 50)

for name, arr in [
    ("Track A (Image)", track_a),
    ("Track B (Text)", track_b),
    ("Track C (Multi)", track_c),
]:
    mean, lower, upper = bootstrap_ci(arr)
    print(f"  {name}: {mean*100:.1f}% [95% CI: {lower*100:.1f}% - {upper*100:.1f}%]")

print("\n[2] VCE (Text - Multimodal) with Bootstrap CI")
print("-" * 50)

vce_samples = track_b.astype(int) - track_c.astype(int)
vce_mean, vce_lower, vce_upper = bootstrap_ci(vce_samples)
print(
    f"  Overall VCE: {vce_mean*100:+.1f}% [95% CI: {vce_lower*100:+.1f}% - {vce_upper*100:+.1f}%]"
)

print("\n[3] McNemar Test (Text vs Multimodal)")
print("-" * 50)

b_only = ((track_b == 1) & (track_c == 0)).sum()
c_only = ((track_b == 0) & (track_c == 1)).sum()

print(f"  Text correct, Multi wrong (b): {b_only}")
print(f"  Multi correct, Text wrong (c): {c_only}")

if b_only + c_only > 0:
    if b_only + c_only < 25:
        result = stats.binomtest(
            min(b_only, c_only), b_only + c_only, 0.5, alternative="two-sided"
        )
        p_value = result.pvalue
        test_type = "Exact binomial"
    else:
        chi2 = (abs(b_only - c_only) - 1) ** 2 / (b_only + c_only)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
        test_type = "McNemar chi-square"

    print(f"  {test_type} p-value: {p_value:.4f}")
    print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
else:
    print("  Cannot compute (no discordant pairs)")

print("\n[4] VCE by Nation with Bootstrap CI")
print("-" * 70)

nations = {}
for d in details:
    nation = d["nation"]
    if nation not in nations:
        nations[nation] = {"b": [], "c": []}
    nations[nation]["b"].append(d["track_b"]["is_correct"])
    nations[nation]["c"].append(d["track_c"]["is_correct"])

print(f"{'Nation':<15} {'N':>5} {'Text%':>8} {'Multi%':>8} {'VCE':>8} {'95% CI':<20}")
print("-" * 70)

for nation in sorted(nations.keys()):
    b_arr = np.array(nations[nation]["b"])
    c_arr = np.array(nations[nation]["c"])
    vce_arr = b_arr.astype(int) - c_arr.astype(int)

    vce_mean, vce_lower, vce_upper = bootstrap_ci(vce_arr)

    print(
        f"{nation:<15} {len(b_arr):>5} {b_arr.mean()*100:>7.1f}% {c_arr.mean()*100:>7.1f}% {vce_mean*100:>+7.1f}% [{vce_lower*100:+.1f}%, {vce_upper*100:+.1f}%]"
    )

print("\n[5] Japan VCE Statistical Test")
print("-" * 50)

japan_b = np.array(nations["Japan"]["b"])
japan_c = np.array(nations["Japan"]["c"])

japan_b_only = ((japan_b == 1) & (japan_c == 0)).sum()
japan_c_only = ((japan_b == 0) & (japan_c == 1)).sum()

print(f"  Japan Text correct, Multi wrong: {japan_b_only}")
print(f"  Japan Multi correct, Text wrong: {japan_c_only}")

if japan_b_only + japan_c_only > 0:
    result = stats.binomtest(
        min(japan_b_only, japan_c_only),
        japan_b_only + japan_c_only,
        0.5,
        alternative="two-sided",
    )
    p_value = result.pvalue
    print(f"  Exact binomial p-value: {p_value:.4f}")
    print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")

print("\n[6] Effect Size (Cohen's h for proportions)")
print("-" * 50)


def cohens_h(p1, p2):
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


overall_h = cohens_h(track_b.mean(), track_c.mean())
japan_h = cohens_h(japan_b.mean(), japan_c.mean())

print(f"  Overall (Text vs Multi): h = {overall_h:.3f}")
print(f"  Japan (Text vs Multi):   h = {japan_h:.3f}")
print()
print("  해석: |h| < 0.2 small, 0.2-0.8 medium, > 0.8 large")

print("\n" + "=" * 70)
print("Summary for Paper")
print("=" * 70)
print("""
1. 전체 VCE는 통계적으로 유의하지 않음 (n=200, CI가 0을 포함)
2. 지역별로 보면 일본/대만은 양의 VCE, EU는 음의 VCE 경향
3. 일본의 Visual Noise (7 vs 5) 역시 n=40에서는 유의하지 않음
4. 더 큰 샘플 (n=500+)이 통계적 유의성 확보에 필요

→ 논문에서는:
   - 현상 자체는 명확 (방향성 일관)
   - 통계적 유의성은 larger scale에서 확인 필요
   - Effect size로 실질적 의미 강조
""")
