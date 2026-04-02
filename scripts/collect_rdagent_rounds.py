#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Iterable


def extract_strings(path: Path, min_len: int = 4) -> str:
    data = path.read_bytes()
    chunks: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        if buf:
            s = ''.join(buf).strip('\n\r\t ')
            if len(s) >= min_len:
                chunks.append(s)
            buf = []

    for b in data:
        if 32 <= b <= 126 or b in (9, 10, 13):
            buf.append(chr(b))
        else:
            flush()
    flush()
    return '\n'.join(chunks)


def first_match(pattern: str, text: str, default: str = '') -> str:
    m = re.search(pattern, text, flags=re.S | re.M | re.I)
    return m.group(1).strip() if m else default


def find_first_file(patterns: Iterable[str], base: Path) -> Path | None:
    for pattern in patterns:
        matches = sorted(base.glob(pattern))
        if matches:
            return matches[0]
    return None


def find_last_file(patterns: Iterable[str], base: Path) -> Path | None:
    for pattern in patterns:
        matches = sorted(base.glob(pattern))
        if matches:
            return matches[-1]
    return None


def parse_round(loop_dir: Path) -> dict:
    hypo_file = find_first_file(["direct_exp_gen/hypothesis generation/*/*.pkl"], loop_dir)
    exp_file = find_last_file(["direct_exp_gen/experiment generation/*/*.pkl"], loop_dir)
    feedback_file = find_last_file(["feedback/feedback/*/*.pkl"], loop_dir)
    runner_file = find_last_file(["running/runner result/*/*.pkl"], loop_dir)

    hypo_text = extract_strings(hypo_file) if hypo_file else ''
    feedback_text = extract_strings(feedback_file) if feedback_file else ''
    runner_text = extract_strings(runner_file) if runner_file else ''
    exp_text = extract_strings(exp_file) if exp_file else ''

    # Fallback: if no expected pickle files found, try to concatenate any .pkl under the loop
    if not (hypo_file or feedback_file or runner_file or exp_file):
        all_pkls = sorted(loop_dir.rglob('*.pkl'))
        if all_pkls:
            combined = []
            for p in all_pkls:
                try:
                    combined.append(extract_strings(p))
                except Exception:
                    # ignore unreadable files
                    continue
            combined_text = '\n\n'.join([c for c in combined if c])
            if combined_text:
                hypo_text = combined_text
                feedback_text = combined_text
                runner_text = combined_text
                exp_text = combined_text

    hypothesis = first_match(r"(?:^|\n)hypothesis\n(.*?)\nreason\n", hypo_text)
    reason = first_match(r"(?:^|\n)reason\n(.*?)\nconcise_reason\n", hypo_text)
    # Try JSON-like extraction if plain-format not found
    if not hypothesis:
        hypothesis = first_match(r'"hypothesis"\s*:\s*"([\s\S]*?)"', hypo_text)
    if not reason:
        reason = first_match(r'"reason"\s*:\s*"([\s\S]*?)"', hypo_text)
    decision = first_match(r"(?:^|\n)decision\n(.*?)\n(?:eda_improvement|reason)\n", feedback_text)
    feedback_reason = first_match(r"(?:^|\n)reason\n(.*?)\n(?:exception|code_change_summary|observations)\n", feedback_text)
    observations = first_match(r"(?:^|\n)observations\n(.*?)\nhypothesis_evaluation\n", feedback_text)
    hypothesis_evaluation = first_match(r"(?:^|\n)hypothesis_evaluation\n(.*?)\nnew_hypothesis\n", feedback_text)
    new_hypothesis = first_match(r"(?:^|\n)new_hypothesis\n(.*?)\n(?:acceptable|$)", feedback_text)
    # JSON-like extraction from feedback text
    if not decision:
        decision = first_match(r'"decision"\s*:\s*"([\s\S]*?)"', feedback_text)
    if not feedback_reason:
        feedback_reason = first_match(r'"reason"\s*:\s*"([\s\S]*?)"', feedback_text)
    if not observations:
        observations = first_match(r'"observations"\s*:\s*"([\s\S]*?)"', feedback_text)
    if not new_hypothesis:
        new_hypothesis = first_match(r'"new_hypothesis"\s*:\s*"([\s\S]*?)"', feedback_text)
    factor_names = sorted(set(re.findall(r"(?:^|\n)factor_name\n([A-Za-z0-9_.$-]+)", runner_text)))

    return {
        "loop": loop_dir.name,
        "hypothesis": hypothesis,
        "reason": reason,
        "decision": decision,
        "feedback_reason": feedback_reason,
        "observations": observations,
        "hypothesis_evaluation": hypothesis_evaluation,
        "new_hypothesis": new_hypothesis,
        "factor_names": factor_names,
        "files": {
            "hypothesis_pickle": str(hypo_file) if hypo_file else "",
            "experiment_pickle": str(exp_file) if exp_file else "",
            "feedback_pickle": str(feedback_file) if feedback_file else "",
            "runner_pickle": str(runner_file) if runner_file else "",
        },
        "raw": {
            "hypothesis_text": hypo_text,
            "feedback_text": feedback_text,
            "runner_text": runner_text,
            "experiment_text": exp_text,
        },
    }


def write_round(round_dir: Path, info: dict) -> None:
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "summary.json").write_text(json.dumps({k: v for k, v in info.items() if k != "raw"}, ensure_ascii=False, indent=2), encoding="utf-8")
    (round_dir / "hypothesis_strings.txt").write_text(info["raw"]["hypothesis_text"], encoding="utf-8")
    (round_dir / "feedback_strings.txt").write_text(info["raw"]["feedback_text"], encoding="utf-8")
    (round_dir / "runner_strings.txt").write_text(info["raw"]["runner_text"], encoding="utf-8")

    md = []
    md.append(f"# {info['loop']} 分析结论\n")
    md.append(f"- 决策: {info['decision'] or 'N/A'}")
    md.append(f"- 生成因子数: {len(info['factor_names'])}")
    if info['factor_names']:
        md.append(f"- 因子列表: {', '.join(info['factor_names'])}")
    md.append("")
    md.append("## 本轮假设")
    md.append(info['hypothesis'] or 'N/A')
    md.append("")
    md.append("## 生成理由")
    md.append(info['reason'] or 'N/A')
    md.append("")
    md.append("## 结果观察")
    md.append(info['observations'] or 'N/A')
    md.append("")
    md.append("## 假设评估")
    md.append(info['hypothesis_evaluation'] or 'N/A')
    md.append("")
    md.append("## 反馈原因")
    md.append(info['feedback_reason'] or 'N/A')
    md.append("")
    md.append("## 下一轮优化方向")
    md.append(info['new_hypothesis'] or 'N/A')
    md.append("")
    md.append("## 原始文件")
    for k, v in info['files'].items():
        md.append(f"- {k}: {v}")
    (round_dir / "analysis_report.md").write_text('\n'.join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', required=True, type=Path)
    parser.add_argument('--out-base', required=True, type=Path)
    parser.add_argument('--clean', action='store_true')
    args = parser.parse_args()

    if args.clean and args.out_base.exists():
        shutil.rmtree(args.out_base)
    args.out_base.mkdir(parents=True, exist_ok=True)

    loop_dirs = sorted([p for p in args.log_dir.iterdir() if p.is_dir() and re.fullmatch(r'Loop_\d+', p.name)], key=lambda p: int(p.name.split('_')[1]))
    leaderboard = ["# rd-agent 多轮因子挖掘汇总", ""]
    leaderboard.append(f"- log_dir: {args.log_dir}")
    leaderboard.append(f"- rounds_found: {len(loop_dirs)}")
    leaderboard.append("")
    leaderboard.append("| Round | Decision | Factors | Hypothesis | Next |")
    leaderboard.append("|---|---|---:|---|---|")

    for idx, loop_dir in enumerate(loop_dirs, start=1):
        info = parse_round(loop_dir)
        round_dir = args.out_base / f"round_{idx:02d}"
        write_round(round_dir, info)
        hypo_short = (info['hypothesis'] or 'N/A').replace('|', ' ')[:80]
        next_short = (info['new_hypothesis'] or 'N/A').replace('|', ' ')[:80]
        leaderboard.append(f"| {idx:02d} | {info['decision'] or 'N/A'} | {len(info['factor_names'])} | {hypo_short} | {next_short} |")

    (args.out_base / 'leaderboard.md').write_text('\n'.join(leaderboard), encoding='utf-8')
    print(f"Collected {len(loop_dirs)} rounds into {args.out_base}")


if __name__ == '__main__':
    main()
