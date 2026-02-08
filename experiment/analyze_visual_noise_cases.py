import json
from pathlib import Path
from datasets import load_dataset

RESULTS_FILE = Path(__file__).parent / "results" / "vce_analysis.json"

with open(RESULTS_FILE, "r") as f:
    data = json.load(f)

noise_cases = data["noise_cases"]

print("=" * 70)
print(f"Visual Noise Cases Analysis ({len(noise_cases)} cases)")
print("=" * 70)

print("\n[1] 지역별 분포")
print("-" * 40)
by_nation = {}
for case in noise_cases:
    nation = case["nation"]
    by_nation[nation] = by_nation.get(nation, 0) + 1

for nation, count in sorted(by_nation.items(), key=lambda x: -x[1]):
    pct = count / len(noise_cases) * 100
    bar = "█" * int(pct / 2)
    print(f"  {nation:<15} {count:>2}개 ({pct:>5.1f}%) {bar}")

print("\n[2] 도메인별 분포")
print("-" * 40)
by_task = {}
for case in noise_cases:
    task = case["task"]
    by_task[task] = by_task.get(task, 0) + 1

for task, count in sorted(by_task.items(), key=lambda x: -x[1]):
    pct = count / len(noise_cases) * 100
    print(f"  {task:<20} {count:>2}개 ({pct:>5.1f}%)")

print("\n[3] 케이스 상세 정보")
print("-" * 70)
print(f"{'#':<3} {'Index':<8} {'Nation':<12} {'Task':<15} {'Correct':<8} {'패턴'}")
print("-" * 70)

for i, case in enumerate(noise_cases):
    pattern = ""
    if case["track_a"] and case["track_b"] and not case["track_c"]:
        pattern = "A✓ B✓ C✗ (둘 다 맞는데 합치면 틀림)"
    elif not case["track_a"] and case["track_b"] and not case["track_c"]:
        pattern = "A✗ B✓ C✗ (텍스트만 맞음)"
    else:
        pattern = f"A{'✓' if case['track_a'] else '✗'} B{'✓' if case['track_b'] else '✗'} C{'✓' if case['track_c'] else '✗'}"

    print(
        f"{i+1:<3} {case['index']:<8} {case['nation']:<12} {case['task']:<15} {case['correct_answer']:<8} {pattern}"
    )

print("\n[4] 패턴 분석")
print("-" * 40)

pattern_both_correct = sum(1 for c in noise_cases if c["track_a"] and c["track_b"])
pattern_text_only = sum(1 for c in noise_cases if not c["track_a"] and c["track_b"])

print(
    f"  패턴 1: 이미지✓ + 텍스트✓ → 멀티모달✗  : {pattern_both_correct}개 ({pattern_both_correct/len(noise_cases)*100:.1f}%)"
)
print(
    f"  패턴 2: 이미지✗ + 텍스트✓ → 멀티모달✗  : {pattern_text_only}개 ({pattern_text_only/len(noise_cases)*100:.1f}%)"
)
print()
print("  💡 패턴 1 해석: 개별적으로는 맞추는데, 합치면 Fusion Interference 발생")
print("  💡 패턴 2 해석: 이미지가 원래 도움 안 되는데, 멀티모달에서 방해까지 함")

print("\n[5] 이미지 저장 (직접 확인용)")
print("-" * 40)

dataset = load_dataset("EuraGovExam/EuraGovExam", split="train")

output_dir = Path(__file__).parent / "visual_noise_images"
output_dir.mkdir(exist_ok=True)

for i, case in enumerate(noise_cases):
    idx = case["index"]
    item = dataset[idx]
    img = item["img"]

    filename = f"{i+1:02d}_{case['nation']}_{case['task']}_{idx}.png"
    filepath = output_dir / filename
    img.save(filepath)
    print(f"  저장: {filename}")

print(f"\n이미지 저장 완료: {output_dir}")
print("\n직접 이미지를 확인하여 공통 특성을 분석하세요:")
print("  - 세로쓰기 여부")
print("  - 표/다이어그램 존재")
print("  - 수식/기호 복잡성")
print("  - 레이아웃 복잡성")
