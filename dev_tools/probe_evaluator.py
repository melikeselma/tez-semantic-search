"""Compute precision@K and OOD-handling correctness from regression_probe.json.

Runs against the JSON dumps produced by dev_tools/regression_probe.py and
emits a per-probe / per-(profile,method) summary that can be pasted into the
thesis report instead of relying on visual inspection of the live UI.

Metrics:
    precision@1 / precision@3 — does the expected substring appear in the
        title or ref of the top result(s)?
    weak_match_recall — for queries marked expected="weak", how many of the
        returned results carry the weak_match flag (a higher number means the
        system is honest about the corpus not covering the topic).
    ood_correct — for an expected="weak" probe, did the top result get
        weak_match=True? Treats the system as correctly refusing to claim
        relevance.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def hit_top_k(rows: list[dict], expected_substr: str, k: int) -> bool:
    if not expected_substr or not rows:
        return False
    needle = expected_substr.lower()
    for row in rows[:k]:
        ref = (row.get("ref") or "").lower()
        title = (row.get("title") or "").lower()
        if needle in ref or needle in title:
            return True
    return False


def evaluate(report: dict) -> dict:
    by_combo: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    per_probe = {}

    for probe_id, block in report.items():
        expected = (block.get("expected") or "").lower()
        is_weak_probe = expected == "weak"
        runs = block.get("runs") or {}
        per_probe[probe_id] = {
            "query": block.get("query"),
            "expected": expected,
            "rows": {},
        }
        for combo, rows in runs.items():
            top1 = hit_top_k(rows, expected, 1) if not is_weak_probe else None
            top3 = hit_top_k(rows, expected, 3) if not is_weak_probe else None
            weak_count = sum(1 for r in rows if r.get("weak_match"))
            ood_correct = bool(rows and rows[0].get("weak_match")) if is_weak_probe else None
            per_probe[probe_id]["rows"][combo] = {
                "top_count": len(rows),
                "top1": top1,
                "top3": top3,
                "weak_count": weak_count,
                "ood_correct": ood_correct,
                "first_ref": rows[0]["ref"] if rows else None,
                "first_title": rows[0]["title"] if rows else None,
                "first_score": rows[0].get("score") if rows else None,
            }
            stats = by_combo[combo]
            if not is_weak_probe:
                stats["p1_total"].append(int(bool(top1)))
                stats["p3_total"].append(int(bool(top3)))
            else:
                stats["ood_total"].append(int(bool(ood_correct)))
                stats["weak_recall"].append(weak_count / max(len(rows), 1))

    summary = {}
    for combo, stats in by_combo.items():
        summary[combo] = {
            "p_at_1": round(_safe_mean(stats.get("p1_total", [])), 3),
            "p_at_3": round(_safe_mean(stats.get("p3_total", [])), 3),
            "ood_correct": round(_safe_mean(stats.get("ood_total", [])), 3),
            "weak_recall": round(_safe_mean(stats.get("weak_recall", [])), 3),
            "n_in_corpus": len(stats.get("p1_total", [])),
            "n_ood": len(stats.get("ood_total", [])),
        }
    return {"summary": summary, "per_probe": per_probe}


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def format_summary(eval_result: dict) -> str:
    lines = []
    summary = eval_result["summary"]
    lines.append("=== Aggregate per (profile|method|cross-encoder) ===")
    header = f"{'combo':45} {'P@1':>6} {'P@3':>6} {'OOD':>6} {'WeakR':>6} {'#in':>4} {'#ood':>5}"
    lines.append(header)
    lines.append("-" * len(header))
    for combo in sorted(summary):
        row = summary[combo]
        lines.append(
            f"{combo:45} {row['p_at_1']:>6.2f} {row['p_at_3']:>6.2f} "
            f"{row['ood_correct']:>6.2f} {row['weak_recall']:>6.2f} "
            f"{row['n_in_corpus']:>4} {row['n_ood']:>5}"
        )
    lines.append("")
    lines.append("=== Per-probe top-1 across combos ===")
    for probe_id, block in eval_result["per_probe"].items():
        lines.append(f"[{probe_id}] expected~{block['expected']!r}  query={block['query']}")
        for combo, run in block["rows"].items():
            badge = ""
            if block["expected"] == "weak":
                badge = "OOD-OK" if run.get("ood_correct") else "OOD-MISS"
            else:
                badge = "P1-HIT" if run.get("top1") else ("P3-HIT" if run.get("top3") else "MISS")
            score = run.get("first_score") or 0.0
            ref = (run.get("first_ref") or "?")[:36]
            title = (run.get("first_title") or "?")[:34]
            lines.append(f"   {combo:45} {badge:8} {score:6.2f}  {ref:36} {title}")
        lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    report = load_report(args.report)
    result = evaluate(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(format_summary(result))


if __name__ == "__main__":
    main()
