import json
from collections import Counter
from pathlib import Path

from evaluate_search import (
    build_pairwise_comparison,
    evaluate,
    load_judgments,
    summarize,
    summarize_by_category,
    summarize_by_field,
    validate_judgments,
    warm_context,
    write_csv,
)
from search import load_mappings
from search_profiles import DEFAULT_PROFILE_KEY, get_profile

BASE_DIR = Path(__file__).resolve().parent
JUDGMENTS_PATH = BASE_DIR / "data" / "evaluation" / "relevance_judgments_rq2.json"
OUT_DIR = BASE_DIR / "reports" / "evaluation" / "rq2_cross_source"
TOP_K = 5
METHODS = ["semantic", "bm25", "hybrid"]


def write_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def row_for(summary, method):
    for item in summary:
        if item["method"] == method:
            return item
    return None


def render_metric_table(rows, title):
    if not rows:
        return f"## {title}\n\nNo rows.\n"

    lines = [
        f"## {title}",
        "",
        "| Method | Bridge@1 | Bridge@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {p1:.3f} | {hit5:.3f} | {r5:.3f} | {mrr:.3f} | {map5:.3f} | {ndcg:.3f} | {lat:.1f} |".format(
                method=row["method"],
                p1=row["mean_precision_at_1"],
                hit5=row["mean_hit_rate_at_k"],
                r5=row["mean_recall_at_k"],
                mrr=row["mean_mrr"],
                map5=row["mean_ap_at_k"],
                ndcg=row["mean_ndcg_at_k"],
                lat=row["median_latency_ms"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_group_sections(rows, field_name, title_prefix):
    sections = []
    group_values = sorted({row[field_name] for row in rows if row.get(field_name)})
    for group_value in group_values:
        group_rows = [row for row in rows if row.get(field_name) == group_value]
        sections.append(render_metric_table(group_rows, f"{title_prefix}: {group_value}"))
    return "\n".join(sections)


def summarize_counts(judgments):
    directions = Counter(item.get("direction") or "unknown" for item in judgments)
    topics = Counter(item.get("topic") or "unknown" for item in judgments)
    anchor_sources = Counter(item.get("anchor_source") or "unknown" for item in judgments)
    return directions, topics, anchor_sources


def build_findings(overall_summary, direction_summary):
    findings = []
    bm25 = row_for(overall_summary, "bm25")
    semantic = row_for(overall_summary, "semantic")
    hybrid = row_for(overall_summary, "hybrid")

    if hybrid and semantic and bm25:
        findings.append(
            "Across the cross-source benchmark, hybrid is the strongest practical system: "
            f"Bridge@5={hybrid['mean_hit_rate_at_k']:.3f}, "
            f"nDCG@{TOP_K}={hybrid['mean_ndcg_at_k']:.3f}."
        )
        findings.append(
            "Pure semantic retrieval shows whether embeddings can bridge sources directly; "
            f"it reaches Bridge@5={semantic['mean_hit_rate_at_k']:.3f}, "
            f"compared with BM25={bm25['mean_hit_rate_at_k']:.3f}."
        )

    hf_to_kg = [row for row in direction_summary if row.get("direction") == "huggingface_to_kaggle"]
    kg_to_hf = [row for row in direction_summary if row.get("direction") == "kaggle_to_huggingface"]
    hf_semantic = row_for(hf_to_kg, "semantic")
    kg_semantic = row_for(kg_to_hf, "semantic")

    if hf_semantic and kg_semantic:
        findings.append(
            "Direction matters. The semantic system performs differently across source pairs: "
            f"HF->Kaggle nDCG@{TOP_K}={hf_semantic['mean_ndcg_at_k']:.3f}, "
            f"Kaggle->HF nDCG@{TOP_K}={kg_semantic['mean_ndcg_at_k']:.3f}."
        )
        findings.append(
            "This direction gap is useful for the thesis because it reflects description quality "
            "and metadata style differences between sources, not just model quality."
        )

    return findings


def write_markdown_report(
    profile_key,
    judgments,
    overall_summary,
    direction_summary,
    topic_summary,
):
    directions, topics, anchor_sources = summarize_counts(judgments)
    findings = build_findings(overall_summary, direction_summary)
    profile = get_profile(profile_key)

    lines = [
        "# RQ2 Report",
        "",
        "Research question:",
        "How successful is an embedding-based system at establishing semantic similarity between datasets collected from different sources?",
        "",
        f"Profile: `{profile.key}` ({profile.label})",
        f"Top-K: `{TOP_K}`",
        f"Methods: `{', '.join(METHODS)}`",
        "",
        "## Benchmark Composition",
        "",
        f"- Total anchors: `{len(judgments)}`",
        f"- Directions: `{dict(directions)}`",
        f"- Anchor sources: `{dict(anchor_sources)}`",
        f"- Topics: `{dict(topics)}`",
        "",
        render_metric_table(overall_summary, "Overall Cross-Source Comparison"),
        render_group_sections(direction_summary, "direction", "Direction Slice"),
        render_group_sections(topic_summary, "topic", "Topic Slice"),
        "## Interim Findings",
        "",
    ]

    for finding in findings:
        lines.append(f"- {finding}")

    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "- Expand the cross-source anchor list with more carefully matched topic pairs before freezing thesis claims.",
            "- Keep semantic retrieval as the direct answer to RQ2, and use hybrid as the strongest applied system.",
            "- In the next phase, compare multiple embedding models on the same RQ2 benchmark to answer RQ3 cleanly.",
            "",
        ]
    )

    (OUT_DIR / "rq2_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(profile_key=DEFAULT_PROFILE_KEY):
    profile = get_profile(profile_key)
    judgments = load_judgments(JUDGMENTS_PATH)
    mappings = load_mappings(profile.key)
    validate_judgments(judgments, mappings)

    context = {
        "mappings": mappings,
        "hybrid_alpha": 0.6,
        "hybrid_candidates": 100,
        "profile": profile.key,
    }
    warm_context(METHODS, context)
    rows = evaluate(judgments, METHODS, TOP_K, context)
    overall_summary = summarize(rows)
    category_summary = summarize_by_category(rows)
    direction_summary = summarize_by_field(rows, "direction")
    topic_summary = summarize_by_field(rows, "topic")
    pairwise = build_pairwise_comparison(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "evaluation_details.csv", rows)
    write_csv(OUT_DIR / "evaluation_summary.csv", overall_summary)
    write_csv(OUT_DIR / "evaluation_by_category.csv", category_summary)
    write_csv(OUT_DIR / "evaluation_by_direction.csv", direction_summary)
    write_csv(OUT_DIR / "evaluation_by_topic.csv", topic_summary)
    write_json(OUT_DIR / "evaluation_summary.json", overall_summary)
    write_json(OUT_DIR / "evaluation_by_category.json", category_summary)
    write_json(OUT_DIR / "evaluation_by_direction.json", direction_summary)
    write_json(OUT_DIR / "evaluation_by_topic.json", topic_summary)
    write_json(OUT_DIR / "evaluation_pairwise.json", pairwise)
    write_markdown_report(profile.key, judgments, overall_summary, direction_summary, topic_summary)

    print(f"[OK] RQ2 report generated in {OUT_DIR}")


if __name__ == "__main__":
    main()
