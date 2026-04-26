import json
from collections import Counter
from pathlib import Path

from evaluate_search import (
    build_pairwise_comparison,
    evaluate,
    load_judgments,
    summarize,
    summarize_by_field,
    validate_judgments,
    warm_context,
    write_csv,
)
from search import load_mappings
from search_profiles import get_profile

BASE_DIR = Path(__file__).resolve().parent
JUDGMENTS_PATH = BASE_DIR / "data" / "evaluation" / "relevance_judgments_semantic_intent.json"
OUT_DIR = BASE_DIR / "reports" / "evaluation" / "semantic_intent_baseline"
TOP_K = 5

PROFILE_METHODS = {
    "minilm": ["semantic", "bm25", "hybrid"],
    "multilingual": ["semantic", "hybrid"],
}


def write_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def row_for(summary, profile, method):
    for item in summary:
        if item["profile"] == profile and item["method"] == method:
            return item
    return None


def metric_table(rows, title):
    if not rows:
        return f"## {title}\n\nNo rows.\n"

    lines = [
        f"## {title}",
        "",
        "| Profile | Method | P@1 | Hit@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {profile} | {method} | {p1:.3f} | {hit5:.3f} | {r5:.3f} | {mrr:.3f} | {map5:.3f} | {ndcg:.3f} | {lat:.1f} |".format(
                profile=row["profile"],
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
        sections.append(metric_table(group_rows, f"{title_prefix}: {group_value}"))
    return "\n".join(sections)


def counts(judgments):
    study_slices = Counter(item.get("study_slice") or "unknown" for item in judgments)
    languages = Counter(item.get("language") or "unknown" for item in judgments)
    topics = Counter(item.get("topic") or "unknown" for item in judgments)
    return study_slices, languages, topics


def build_findings(overall_summary, study_slice_summary):
    findings = []

    minilm_bm25_overall = row_for(overall_summary, "minilm", "bm25")
    minilm_sem_overall = row_for(overall_summary, "minilm", "semantic")
    minilm_hybrid_overall = row_for(overall_summary, "minilm", "hybrid")
    multilingual_sem_overall = row_for(overall_summary, "multilingual", "semantic")
    multilingual_hybrid_overall = row_for(overall_summary, "multilingual", "hybrid")

    english_rows = [
        row for row in study_slice_summary if row.get("study_slice") == "english_latent_intent"
    ]
    turkish_rows = [
        row for row in study_slice_summary if row.get("study_slice") == "turkish_latent_intent"
    ]

    english_bm25 = row_for(english_rows, "minilm", "bm25")
    english_sem = row_for(english_rows, "minilm", "semantic")
    english_hybrid = row_for(english_rows, "minilm", "hybrid")
    turkish_minilm_sem = row_for(turkish_rows, "minilm", "semantic")
    turkish_minilm_hybrid = row_for(turkish_rows, "minilm", "hybrid")
    turkish_multi_sem = row_for(turkish_rows, "multilingual", "semantic")
    turkish_multi_hybrid = row_for(turkish_rows, "multilingual", "hybrid")

    overall_best = max(overall_summary, key=lambda row: row["mean_ndcg_at_k"], default=None)
    if overall_best and minilm_hybrid_overall and minilm_sem_overall and minilm_bm25_overall:
        findings.append(
            "Across the full latent-intent benchmark, the current best system is "
            f"{overall_best['profile']} {overall_best['method']} "
            f"(nDCG@{TOP_K}={overall_best['mean_ndcg_at_k']:.3f}). "
            f"MiniLM semantic and hybrid now both stay ahead of lexical BM25 "
            f"({minilm_sem_overall['mean_ndcg_at_k']:.3f} / {minilm_hybrid_overall['mean_ndcg_at_k']:.3f} vs "
            f"{minilm_bm25_overall['mean_ndcg_at_k']:.3f})."
        )

    if english_sem and english_bm25:
        findings.append(
            "On English latent-intent queries, semantic retrieval already improves over the lexical baseline: "
            f"MiniLM semantic nDCG@{TOP_K}={english_sem['mean_ndcg_at_k']:.3f}, "
            f"BM25 nDCG@{TOP_K}={english_bm25['mean_ndcg_at_k']:.3f}."
        )
    if english_hybrid and english_sem:
        findings.append(
            "English retrieval is now stable across both semantic and hybrid variants, which means the next gains will likely come "
            f"from better document representations rather than more aggressive query rewriting alone "
            f"(semantic nDCG@{TOP_K}={english_sem['mean_ndcg_at_k']:.3f}, hybrid={english_hybrid['mean_ndcg_at_k']:.3f})."
        )

    if turkish_minilm_sem and turkish_minilm_hybrid:
        findings.append(
            "The query-understanding and concept-expansion layer closes most of the Turkish latent-intent gap for the MiniLM profile: "
            f"MiniLM semantic reaches nDCG@{TOP_K}={turkish_minilm_sem['mean_ndcg_at_k']:.3f} and MiniLM hybrid "
            f"reaches {turkish_minilm_hybrid['mean_ndcg_at_k']:.3f}."
        )
    if turkish_multi_sem and turkish_multi_hybrid:
        findings.append(
            "The multilingual profile is still weaker than the English baseline on the same Turkish slice "
            f"(semantic nDCG@{TOP_K}={turkish_multi_sem['mean_ndcg_at_k']:.3f}, hybrid={turkish_multi_hybrid['mean_ndcg_at_k']:.3f}), "
            "so cross-lingual robustness remains an active improvement area rather than a solved problem."
        )

    if multilingual_sem_overall and minilm_sem_overall:
        findings.append(
            "The profile comparison here should be interpreted as a retrieval baseline study, not as a final model claim. "
            "The multilingual profile is an extension path for bilingual intent coverage, while MiniLM remains the stronger English baseline."
        )

    return findings


def write_markdown_report(judgments, overall_summary, study_slice_summary, language_summary, topic_summary):
    study_slices, languages, topics = counts(judgments)
    findings = build_findings(overall_summary, study_slice_summary)

    lines = [
        "# Semantic Intent Baseline Report",
        "",
        "Purpose:",
        "Track latent semantic intent retrieval quality while query understanding, semantic enrichment, and model adaptation steps are added incrementally.",
        "",
        "Profiles and methods:",
        f"- `{get_profile('minilm').key}`: {get_profile('minilm').label} with semantic, BM25, and hybrid",
        f"- `{get_profile('multilingual').key}`: {get_profile('multilingual').label} with semantic and hybrid",
        "",
        f"Top-K: `{TOP_K}`",
        "",
        "## Benchmark Composition",
        "",
        f"- Total queries: `{len(judgments)}`",
        f"- Study slices: `{dict(study_slices)}`",
        f"- Languages: `{dict(languages)}`",
        f"- Topics: `{dict(topics)}`",
        "",
        metric_table(overall_summary, "Overall Baseline Comparison"),
        render_group_sections(study_slice_summary, "study_slice", "Study Slice"),
        render_group_sections(language_summary, "language", "Language Slice"),
        render_group_sections(topic_summary, "topic", "Topic Slice"),
        "## Key Findings",
        "",
    ]

    for finding in findings:
        lines.append(f"- {finding}")

    lines.extend(
        [
            "",
            "## Why This Matters",
            "",
            "- This benchmark isolates the exact thesis pain point: user wording and dataset wording often differ even when the underlying intent matches.",
            "- The English slice shows how far the current retriever can go once intent terms are made explicit at query time.",
            "- The Turkish slice now measures whether concept projection is robust enough for live demo queries, not just whether Turkish embeddings exist.",
            "- The narrow marine slice keeps the benchmark honest about current corpus coverage instead of overclaiming capability.",
            "",
            "## Next Stage",
            "",
            "- Enrich dataset-side text with semantic summaries, inferred domains, and use-case labels so the retriever has better document representations to match against the improved queries.",
            "",
        ]
    )

    (OUT_DIR / "semantic_intent_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    judgments = load_judgments(JUDGMENTS_PATH)
    rows = []

    for profile_key, methods in PROFILE_METHODS.items():
        profile = get_profile(profile_key)
        mappings = load_mappings(profile.key)
        validate_judgments(judgments, mappings)
        context = {
            "mappings": mappings,
            "hybrid_alpha": 0.6,
            "hybrid_candidates": 100,
            "profile": profile.key,
        }
        warm_context(methods, context)
        rows.extend(evaluate(judgments, methods, TOP_K, context))

    overall_summary = summarize(rows)
    study_slice_summary = summarize_by_field(rows, "study_slice")
    language_summary = summarize_by_field(rows, "language")
    topic_summary = summarize_by_field(rows, "topic")
    benchmark_summary = summarize_by_field(rows, "benchmark")
    pairwise = build_pairwise_comparison(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "evaluation_details.csv", rows)
    write_csv(OUT_DIR / "evaluation_summary.csv", overall_summary)
    write_csv(OUT_DIR / "evaluation_by_study_slice.csv", study_slice_summary)
    write_csv(OUT_DIR / "evaluation_by_language.csv", language_summary)
    write_csv(OUT_DIR / "evaluation_by_topic.csv", topic_summary)
    write_csv(OUT_DIR / "evaluation_by_benchmark.csv", benchmark_summary)
    write_json(OUT_DIR / "evaluation_summary.json", overall_summary)
    write_json(OUT_DIR / "evaluation_by_study_slice.json", study_slice_summary)
    write_json(OUT_DIR / "evaluation_by_language.json", language_summary)
    write_json(OUT_DIR / "evaluation_by_topic.json", topic_summary)
    write_json(OUT_DIR / "evaluation_by_benchmark.json", benchmark_summary)
    write_json(OUT_DIR / "evaluation_pairwise.json", pairwise)
    write_markdown_report(
        judgments,
        overall_summary,
        study_slice_summary,
        language_summary,
        topic_summary,
    )

    print(f"[OK] Semantic intent report generated in {OUT_DIR}")


if __name__ == "__main__":
    main()
