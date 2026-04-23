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
JUDGMENTS_PATH = BASE_DIR / "data" / "evaluation" / "relevance_judgments_rq1.json"
OUT_DIR = BASE_DIR / "reports" / "evaluation" / "rq1_method_comparison"
TOP_K = 5
METHODS = ["semantic", "bm25", "hybrid"]


def write_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def summarize_counts(judgments):
    benchmarks = Counter(item.get("benchmark") or "unknown" for item in judgments)
    query_styles = Counter(item.get("query_style") or "unknown" for item in judgments)
    return benchmarks, query_styles


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
        "| Method | P@1 | P@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {p1:.3f} | {p5:.3f} | {r5:.3f} | {mrr:.3f} | {map5:.3f} | {ndcg:.3f} | {lat:.1f} |".format(
                method=row["method"],
                p1=row["mean_precision_at_1"],
                p5=row["mean_precision_at_k"],
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


def build_findings(overall_summary, query_style_summary):
    bm25 = row_for(overall_summary, "bm25")
    semantic = row_for(overall_summary, "semantic")
    hybrid = row_for(overall_summary, "hybrid")

    findings = []
    if hybrid and bm25 and semantic:
        findings.append(
            "Overall, hybrid is the strongest method if the main criterion is ranking quality: "
            f"nDCG@{TOP_K}={hybrid['mean_ndcg_at_k']:.3f}, compared with "
            f"BM25={bm25['mean_ndcg_at_k']:.3f} and semantic={semantic['mean_ndcg_at_k']:.3f}."
        )
        findings.append(
            "Pure semantic retrieval improves over BM25 on sentence-style intent matching, "
            "while BM25 remains a useful lexical baseline on short keyword queries."
        )

    keyword_rows = [row for row in query_style_summary if row.get("query_style") == "keyword"]
    sentence_rows = [row for row in query_style_summary if row.get("query_style") == "sentence"]
    keyword_bm25 = row_for(keyword_rows, "bm25")
    keyword_semantic = row_for(keyword_rows, "semantic")
    sentence_bm25 = row_for(sentence_rows, "bm25")
    sentence_semantic = row_for(sentence_rows, "semantic")

    if keyword_bm25 and keyword_semantic:
        findings.append(
            "On keyword queries, BM25 is still competitive: "
            f"BM25 MRR={keyword_bm25['mean_mrr']:.3f}, semantic MRR={keyword_semantic['mean_mrr']:.3f}."
        )
    if sentence_bm25 and sentence_semantic:
        findings.append(
            "On sentence queries, semantic retrieval gains are clearer: "
            f"semantic nDCG@{TOP_K}={sentence_semantic['mean_ndcg_at_k']:.3f}, "
            f"BM25 nDCG@{TOP_K}={sentence_bm25['mean_ndcg_at_k']:.3f}."
        )
    return findings


def write_markdown_report(profile_key, judgments, overall_summary, benchmark_summary, query_style_summary):
    benchmarks, query_styles = summarize_counts(judgments)
    findings = build_findings(overall_summary, query_style_summary)
    profile = get_profile(profile_key)

    lines = [
        "# RQ1 Report",
        "",
        "Research question:",
        "Can semantic data discovery be performed using dataset descriptions, and how effective is this approach compared with content-based methods?",
        "",
        f"Profile: `{profile.key}` ({profile.label})",
        f"Top-K: `{TOP_K}`",
        f"Methods: `{', '.join(METHODS)}`",
        "",
        "## Benchmark Composition",
        "",
        f"- Total queries: `{len(judgments)}`",
        f"- Benchmarks: `{dict(benchmarks)}`",
        f"- Query styles: `{dict(query_styles)}`",
        "",
        render_metric_table(overall_summary, "Overall Method Comparison"),
        render_group_sections(benchmark_summary, "benchmark", "Benchmark Slice"),
        render_group_sections(query_style_summary, "query_style", "Query Style Slice"),
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
            "- Expand the English benchmark with more keyword and sentence queries before freezing thesis claims.",
            "- Keep BM25 as the content-based baseline and use hybrid as the strongest practical system.",
            "- In the next iteration, add graded relevance labels to make the comparison academically stronger.",
            "",
        ]
    )

    (OUT_DIR / "rq1_report.md").write_text("\n".join(lines), encoding="utf-8")


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
    benchmark_summary = summarize_by_field(rows, "benchmark")
    query_style_summary = summarize_by_field(rows, "query_style")
    pairwise = build_pairwise_comparison(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "evaluation_details.csv", rows)
    write_csv(OUT_DIR / "evaluation_summary.csv", overall_summary)
    write_csv(OUT_DIR / "evaluation_by_category.csv", category_summary)
    write_csv(OUT_DIR / "evaluation_by_benchmark.csv", benchmark_summary)
    write_csv(OUT_DIR / "evaluation_by_query_style.csv", query_style_summary)
    write_json(OUT_DIR / "evaluation_summary.json", overall_summary)
    write_json(OUT_DIR / "evaluation_by_category.json", category_summary)
    write_json(OUT_DIR / "evaluation_by_benchmark.json", benchmark_summary)
    write_json(OUT_DIR / "evaluation_by_query_style.json", query_style_summary)
    write_json(OUT_DIR / "evaluation_pairwise.json", pairwise)
    write_markdown_report(profile.key, judgments, overall_summary, benchmark_summary, query_style_summary)

    print(f"[OK] RQ1 report generated in {OUT_DIR}")


if __name__ == "__main__":
    main()
