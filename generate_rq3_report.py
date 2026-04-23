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
JUDGMENTS_PATH = BASE_DIR / "data" / "evaluation" / "relevance_judgments_rq3.json"
OUT_DIR = BASE_DIR / "reports" / "evaluation" / "rq3_model_effect"
TOP_K = 5
METHODS = ["semantic", "hybrid"]
PROFILE_KEYS = ["minilm", "multilingual"]


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
    return study_slices, languages


def build_findings(overall_summary, study_slice_summary):
    findings = []

    en_minilm_sem = row_for(study_slice_summary, "minilm", "semantic")
    en_multi_sem = row_for(study_slice_summary, "multilingual", "semantic")
    en_minilm_hyb = row_for(study_slice_summary, "minilm", "hybrid")
    en_multi_hyb = row_for(study_slice_summary, "multilingual", "hybrid")

    english_sem_rows = [row for row in study_slice_summary if row.get("study_slice") == "english_main"]
    tr_rows = [row for row in study_slice_summary if row.get("study_slice") == "tr_subset"]
    cross_rows = [row for row in study_slice_summary if row.get("study_slice") == "cross_source"]

    minilm_en_sem = row_for(english_sem_rows, "minilm", "semantic")
    multi_en_sem = row_for(english_sem_rows, "multilingual", "semantic")
    minilm_tr_sem = row_for(tr_rows, "minilm", "semantic")
    multi_tr_sem = row_for(tr_rows, "multilingual", "semantic")
    minilm_cross_sem = row_for(cross_rows, "minilm", "semantic")
    multi_cross_sem = row_for(cross_rows, "multilingual", "semantic")

    if minilm_en_sem and multi_en_sem:
        findings.append(
            "On the main English benchmark, the compact English-specific model remains stronger: "
            f"MiniLM semantic nDCG@{TOP_K}={minilm_en_sem['mean_ndcg_at_k']:.3f}, "
            f"Multilingual semantic nDCG@{TOP_K}={multi_en_sem['mean_ndcg_at_k']:.3f}."
        )

    if minilm_tr_sem and multi_tr_sem:
        findings.append(
            "On the Turkish subset, the multilingual profile is clearly better: "
            f"Multilingual semantic nDCG@{TOP_K}={multi_tr_sem['mean_ndcg_at_k']:.3f}, "
            f"MiniLM semantic nDCG@{TOP_K}={minilm_tr_sem['mean_ndcg_at_k']:.3f}."
        )

    if minilm_cross_sem and multi_cross_sem:
        findings.append(
            "For cross-source similarity, model choice also changes bridging quality: "
            f"MiniLM semantic nDCG@{TOP_K}={minilm_cross_sem['mean_ndcg_at_k']:.3f}, "
            f"Multilingual semantic nDCG@{TOP_K}={multi_cross_sem['mean_ndcg_at_k']:.3f}."
        )

    overall_minilm_sem = row_for(overall_summary, "minilm", "semantic")
    overall_multi_sem = row_for(overall_summary, "multilingual", "semantic")
    if overall_minilm_sem and overall_multi_sem:
        findings.append(
            "Within the current thesis scope, the comparison mainly captures model type and training objective, "
            "not a large size jump, because both active profiles are lightweight models."
        )

    return findings


def write_markdown_report(
    judgments,
    overall_summary,
    study_slice_summary,
    language_summary,
):
    study_slices, languages = counts(judgments)
    findings = build_findings(overall_summary, study_slice_summary)

    lines = [
        "# RQ3 Report",
        "",
        "Research question:",
        "How does the type and size of the language model affect semantic similarity results?",
        "",
        "Profiles compared:",
        f"- `{get_profile('minilm').key}`: {get_profile('minilm').label}",
        f"- `{get_profile('multilingual').key}`: {get_profile('multilingual').label}",
        "",
        f"Top-K: `{TOP_K}`",
        f"Methods: `{', '.join(METHODS)}`",
        "",
        "## Benchmark Composition",
        "",
        f"- Total queries: `{len(judgments)}`",
        f"- Study slices: `{dict(study_slices)}`",
        f"- Languages: `{dict(languages)}`",
        "",
        metric_table(overall_summary, "Overall Profile Comparison"),
        render_group_sections(study_slice_summary, "study_slice", "Study Slice"),
        render_group_sections(language_summary, "language", "Language Slice"),
        "## Interim Findings",
        "",
    ]

    for finding in findings:
        lines.append(f"- {finding}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- MiniLM should be treated as the main English thesis model.",
            "- Multilingual should be treated as the EN+TR extension profile, not as the default English system.",
            "- This supports the thesis framing: one main English pipeline, plus a smaller multilingual extension for the bilingual subset.",
            "",
        ]
    )

    (OUT_DIR / "rq3_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    judgments = load_judgments(JUDGMENTS_PATH)
    rows = []

    for profile_key in PROFILE_KEYS:
        profile = get_profile(profile_key)
        mappings = load_mappings(profile.key)
        validate_judgments(judgments, mappings)
        context = {
            "mappings": mappings,
            "hybrid_alpha": 0.6,
            "hybrid_candidates": 100,
            "profile": profile.key,
        }
        warm_context(METHODS, context)
        rows.extend(evaluate(judgments, METHODS, TOP_K, context))

    overall_summary = summarize(rows)
    study_slice_summary = summarize_by_field(rows, "study_slice")
    language_summary = summarize_by_field(rows, "language")
    benchmark_summary = summarize_by_field(rows, "benchmark")
    pairwise = build_pairwise_comparison(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "evaluation_details.csv", rows)
    write_csv(OUT_DIR / "evaluation_summary.csv", overall_summary)
    write_csv(OUT_DIR / "evaluation_by_study_slice.csv", study_slice_summary)
    write_csv(OUT_DIR / "evaluation_by_language.csv", language_summary)
    write_csv(OUT_DIR / "evaluation_by_benchmark.csv", benchmark_summary)
    write_json(OUT_DIR / "evaluation_summary.json", overall_summary)
    write_json(OUT_DIR / "evaluation_by_study_slice.json", study_slice_summary)
    write_json(OUT_DIR / "evaluation_by_language.json", language_summary)
    write_json(OUT_DIR / "evaluation_by_benchmark.json", benchmark_summary)
    write_json(OUT_DIR / "evaluation_pairwise.json", pairwise)
    write_markdown_report(judgments, overall_summary, study_slice_summary, language_summary)

    print(f"[OK] RQ3 report generated in {OUT_DIR}")


if __name__ == "__main__":
    main()
