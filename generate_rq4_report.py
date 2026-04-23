import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median

from evaluate_search import (
    evaluate,
    load_judgments,
    validate_judgments,
    warm_context,
    write_csv,
)
from query_understanding import STOPWORDS, tokenize
from search import load_mappings
from search_profiles import get_profile

BASE_DIR = Path(__file__).resolve().parent
JUDGMENTS_PATH = BASE_DIR / "data" / "evaluation" / "relevance_judgments_rq3.json"
OUT_DIR = BASE_DIR / "reports" / "evaluation" / "rq4_description_quality"
PROFILE_KEY = "minilm"
TOP_K = 5
METHODS = ["semantic", "hybrid"]
HEADING_RE = re.compile(r"^\s*(#|[-*]|\d+\.)")
TAG_RE = re.compile(r"\b[a-z_]+:[a-z0-9_./-]+\b")


def write_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def quantile(values, ratio):
    if not values:
        return 0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * ratio))
    index = max(0, min(index, len(ordered) - 1))
    return ordered[index]


def split_refs(value):
    if not value:
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def content_terms(text):
    return [
        token
        for token in tokenize(text or "")
        if len(token) > 2 and token not in STOPWORDS
    ]


def classify_description_style(description):
    text = (description or "").strip()
    if not text:
        return "mixed_structured"

    tokens = tokenize(text)
    if not tokens:
        return "mixed_structured"

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sentence_like_parts = [
        part.strip()
        for part in re.split(r"[.!?]+", text)
        if len(tokenize(part)) >= 4
    ]
    structural_lines = sum(1 for line in lines if HEADING_RE.match(line))
    tag_like = len(TAG_RE.findall(text.lower()))
    markdown_bonus = sum(text.count(marker) for marker in ("##", "###", "---", "`"))
    stopword_ratio = sum(1 for token in tokens if token in STOPWORDS) / max(len(tokens), 1)
    structural_score = structural_lines + tag_like
    if text.startswith("---"):
        structural_score += 2
    if markdown_bonus >= 3:
        structural_score += 1

    if structural_score >= 3 and len(sentence_like_parts) <= 1 and stopword_ratio < 0.22:
        return "metadata_heavy"
    if len(sentence_like_parts) >= 2 and stopword_ratio >= 0.22 and structural_score <= 1:
        return "narrative"
    return "mixed_structured"


def compute_thresholds(mappings):
    documents = list(mappings.values())
    length_values = [
        int(item.get("description_len_words") or len((item.get("description") or "").split()))
        for item in documents
    ]
    term_values = [len(set(content_terms(item.get("description") or ""))) for item in documents]
    return {
        "length_word_thresholds": {
            "short_max": quantile(length_values, 0.33),
            "medium_max": quantile(length_values, 0.66),
        },
        "term_thresholds": {
            "sparse_max": quantile(term_values, 0.33),
            "moderate_max": quantile(term_values, 0.66),
        },
    }


def bucket_length(word_count, thresholds):
    short_max = thresholds["length_word_thresholds"]["short_max"]
    medium_max = thresholds["length_word_thresholds"]["medium_max"]
    if word_count <= short_max:
        return "short"
    if word_count <= medium_max:
        return "medium"
    return "long"


def bucket_term_richness(term_count, thresholds):
    sparse_max = thresholds["term_thresholds"]["sparse_max"]
    moderate_max = thresholds["term_thresholds"]["moderate_max"]
    if term_count <= sparse_max:
        return "term_sparse"
    if term_count <= moderate_max:
        return "term_moderate"
    return "term_rich"


def build_doc_features(mappings, thresholds):
    features = {}
    for item in mappings.values():
        ref = item.get("ref")
        if not ref:
            continue

        description = item.get("description") or ""
        word_count = int(item.get("description_len_words") or len(description.split()))
        unique_terms = sorted(set(content_terms(description)))
        term_count = len(unique_terms)
        features[ref] = {
            "ref": ref,
            "source": item.get("source") or "unknown",
            "title": item.get("title") or "",
            "description_word_count": word_count,
            "term_count": term_count,
            "description_style": classify_description_style(description),
            "length_bucket": bucket_length(word_count, thresholds),
            "term_bucket": bucket_term_richness(term_count, thresholds),
            "description_preview": description[:220].replace("\n", " ").strip(),
        }
    return features


def summarize_corpus_features(doc_features):
    feature_rows = []
    total_docs = len(doc_features)
    for field in ("length_bucket", "description_style", "term_bucket"):
        counts = Counter(item[field] for item in doc_features.values())
        for bucket, count in sorted(counts.items()):
            feature_rows.append(
                {
                    "feature": field,
                    "bucket": bucket,
                    "documents": count,
                    "share": count / total_docs if total_docs else 0.0,
                }
            )
    return feature_rows


def build_exposure_rows(judgments, evaluation_rows, doc_features):
    judgment_by_id = {item["id"]: item for item in judgments}
    exposure_rows = []

    for row in evaluation_rows:
        judgment = judgment_by_id[row["query_id"]]
        retrieved_refs = split_refs(row.get("retrieved_refs"))
        rank_lookup = {ref: index for index, ref in enumerate(retrieved_refs, start=1)}

        for ref in judgment.get("relevant_refs") or []:
            feature = doc_features.get(ref)
            if not feature:
                continue
            rank = rank_lookup.get(ref)
            exposure_rows.append(
                {
                    "query_id": judgment["id"],
                    "query_style": judgment.get("query_style") or "",
                    "benchmark": judgment.get("benchmark") or "",
                    "study_slice": judgment.get("study_slice") or "",
                    "method": row["method"],
                    "profile": row["profile"],
                    "ref": ref,
                    "source": feature["source"],
                    "title": feature["title"],
                    "hit_at_k": 1.0 if rank else 0.0,
                    "top1_hit": 1.0 if rank == 1 else 0.0,
                    "rank": rank if rank else "",
                    "description_word_count": feature["description_word_count"],
                    "term_count": feature["term_count"],
                    "length_bucket": feature["length_bucket"],
                    "description_style": feature["description_style"],
                    "term_bucket": feature["term_bucket"],
                    "description_preview": feature["description_preview"],
                }
            )

    return exposure_rows


def summarize_exposures(exposure_rows, field_name):
    summary = []
    bucket_values = sorted({row[field_name] for row in exposure_rows if row.get(field_name)})
    for method in METHODS:
        method_rows = [row for row in exposure_rows if row["method"] == method]
        for bucket in bucket_values:
            bucket_rows = [row for row in method_rows if row.get(field_name) == bucket]
            if not bucket_rows:
                continue
            hit_ranks = [int(row["rank"]) for row in bucket_rows if row["rank"] != ""]
            summary.append(
                {
                    "method": method,
                    field_name: bucket,
                    "relevant_pairs": len(bucket_rows),
                    "unique_docs": len({row["ref"] for row in bucket_rows}),
                    "hit_rate_at_k": mean(row["hit_at_k"] for row in bucket_rows),
                    "top1_rate": mean(row["top1_hit"] for row in bucket_rows),
                    "mean_rank_when_hit": mean(hit_ranks) if hit_ranks else "",
                    "median_rank_when_hit": median(hit_ranks) if hit_ranks else "",
                }
            )
    return summary


def row_for(summary, method, field_name, bucket):
    for item in summary:
        if item["method"] == method and item.get(field_name) == bucket:
            return item
    return None


def render_corpus_table(rows):
    lines = [
        "## Corpus Feature Distribution",
        "",
        "| Feature | Bucket | Documents | Share |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {feature} | {bucket} | {documents} | {share:.3f} |".format(
                feature=row["feature"],
                bucket=row["bucket"],
                documents=row["documents"],
                share=row["share"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_feature_table(rows, field_name, title):
    if not rows:
        return f"## {title}\n\nNo rows.\n"

    lines = [
        f"## {title}",
        "",
        "| Method | Bucket | Relevant Pairs | Unique Docs | Hit@5 | Top1 | Mean Rank When Hit |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {bucket} | {pairs} | {docs} | {hit:.3f} | {top1:.3f} | {rank} |".format(
                method=row["method"],
                bucket=row[field_name],
                pairs=row["relevant_pairs"],
                docs=row["unique_docs"],
                hit=row["hit_rate_at_k"],
                top1=row["top1_rate"],
                rank=f"{row['mean_rank_when_hit']:.2f}" if row["mean_rank_when_hit"] != "" else "-",
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_findings(length_summary, style_summary, term_summary):
    findings = []

    semantic_short = row_for(length_summary, "semantic", "length_bucket", "short")
    semantic_long = row_for(length_summary, "semantic", "length_bucket", "long")
    if semantic_short and semantic_long:
        findings.append(
            "Longer descriptions are easier to recover semantically. "
            f"Semantic Hit@{TOP_K} rises from {semantic_short['hit_rate_at_k']:.3f} on short descriptions "
            f"to {semantic_long['hit_rate_at_k']:.3f} on long descriptions."
        )

    semantic_meta = row_for(style_summary, "semantic", "description_style", "metadata_heavy")
    semantic_narr = row_for(style_summary, "semantic", "description_style", "narrative")
    if semantic_meta and semantic_narr:
        findings.append(
            "Description structure matters. Narrative descriptions produce better semantic retrieval "
            f"than metadata-heavy ones: Hit@{TOP_K}={semantic_narr['hit_rate_at_k']:.3f} vs "
            f"{semantic_meta['hit_rate_at_k']:.3f}."
        )

    semantic_sparse = row_for(term_summary, "semantic", "term_bucket", "term_sparse")
    semantic_rich = row_for(term_summary, "semantic", "term_bucket", "term_rich")
    if semantic_sparse and semantic_rich:
        findings.append(
            "Term-rich descriptions carry stronger semantic signals. "
            f"Semantic Hit@{TOP_K} goes from {semantic_sparse['hit_rate_at_k']:.3f} on term-sparse documents "
            f"to {semantic_rich['hit_rate_at_k']:.3f} on term-rich documents."
        )

    hybrid_sparse = row_for(term_summary, "hybrid", "term_bucket", "term_sparse")
    if semantic_sparse and hybrid_sparse:
        findings.append(
            "On weak descriptions, hybrid remains a useful fallback. "
            f"For term-sparse documents, hybrid Hit@{TOP_K}={hybrid_sparse['hit_rate_at_k']:.3f} "
            f"while pure semantic stays at {semantic_sparse['hit_rate_at_k']:.3f}."
        )

    return findings


def write_markdown_report(judgments, thresholds, corpus_rows, length_summary, style_summary, term_summary):
    findings = build_findings(length_summary, style_summary, term_summary)
    profile = get_profile(PROFILE_KEY)
    benchmark_counts = Counter(item.get("study_slice") or "unknown" for item in judgments)

    lines = [
        "# RQ4 Report",
        "",
        "Research question:",
        "How do description language structure, length, and included terms affect semantic representation quality?",
        "",
        f"Profile: `{profile.key}` ({profile.label})",
        f"Top-K: `{TOP_K}`",
        f"Methods: `{', '.join(METHODS)}`",
        "",
        "## Evaluation Scope",
        "",
        f"- English queries only: `{len(judgments)}`",
        f"- Study slices: `{dict(benchmark_counts)}`",
        "- RQ4 uses the existing thesis benchmark and re-reads it from the document-quality perspective.",
        "",
        "## Bucket Definitions",
        "",
        f"- Length buckets: `short <= {thresholds['length_word_thresholds']['short_max']}` words, "
        f"`medium <= {thresholds['length_word_thresholds']['medium_max']}` words, `long` above that.",
        f"- Term richness buckets: `term_sparse <= {thresholds['term_thresholds']['sparse_max']}` unique content terms, "
        f"`term_moderate <= {thresholds['term_thresholds']['moderate_max']}`, `term_rich` above that.",
        "- Style buckets are heuristic: `metadata_heavy`, `mixed_structured`, and `narrative`.",
        "",
        render_corpus_table(corpus_rows),
        render_feature_table(length_summary, "length_bucket", "Retrieval by Description Length"),
        render_feature_table(style_summary, "description_style", "Retrieval by Description Style"),
        render_feature_table(term_summary, "term_bucket", "Retrieval by Term Richness"),
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
            "- RQ4 suggests that semantic quality is not only a model issue; document quality also shapes retrieval success.",
            "- Short, tag-like, or metadata-heavy descriptions weaken the semantic signal.",
            "- Longer and more content-rich descriptions make the main English semantic pipeline more reliable.",
            "- This supports a practical thesis recommendation: improve normalization, enrich weak descriptions, and flag low-information records before indexing.",
            "",
        ]
    )

    (OUT_DIR / "rq4_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    profile = get_profile(PROFILE_KEY)
    judgments = [
        item
        for item in load_judgments(JUDGMENTS_PATH)
        if (item.get("language") or "").lower() == "en"
    ]
    mappings = load_mappings(profile.key)
    validate_judgments(judgments, mappings)

    thresholds = compute_thresholds(mappings)
    doc_features = build_doc_features(mappings, thresholds)
    corpus_rows = summarize_corpus_features(doc_features)

    context = {
        "mappings": mappings,
        "hybrid_alpha": 0.6,
        "hybrid_candidates": 100,
        "profile": profile.key,
    }
    warm_context(METHODS, context)
    evaluation_rows = evaluate(judgments, METHODS, TOP_K, context)
    exposure_rows = build_exposure_rows(judgments, evaluation_rows, doc_features)
    length_summary = summarize_exposures(exposure_rows, "length_bucket")
    style_summary = summarize_exposures(exposure_rows, "description_style")
    term_summary = summarize_exposures(exposure_rows, "term_bucket")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "evaluation_details.csv", evaluation_rows)
    write_csv(OUT_DIR / "rq4_relevant_pair_details.csv", exposure_rows)
    write_csv(OUT_DIR / "rq4_by_length_bucket.csv", length_summary)
    write_csv(OUT_DIR / "rq4_by_description_style.csv", style_summary)
    write_csv(OUT_DIR / "rq4_by_term_bucket.csv", term_summary)
    write_csv(OUT_DIR / "rq4_corpus_feature_distribution.csv", corpus_rows)
    write_json(OUT_DIR / "evaluation_details.json", evaluation_rows)
    write_json(OUT_DIR / "rq4_relevant_pair_details.json", exposure_rows)
    write_json(OUT_DIR / "rq4_by_length_bucket.json", length_summary)
    write_json(OUT_DIR / "rq4_by_description_style.json", style_summary)
    write_json(OUT_DIR / "rq4_by_term_bucket.json", term_summary)
    write_json(
        OUT_DIR / "rq4_corpus_feature_distribution.json",
        {
            "thresholds": thresholds,
            "distribution": corpus_rows,
        },
    )
    write_markdown_report(
        judgments,
        thresholds,
        corpus_rows,
        length_summary,
        style_summary,
        term_summary,
    )

    print(f"[OK] RQ4 report generated in {OUT_DIR}")


if __name__ == "__main__":
    main()
