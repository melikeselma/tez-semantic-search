from query_understanding import normalize_text, tokenize
from quality_scoring import compute_quality_adjustment

DEFAULT_RERANK_DEPTH = 40

DOMAIN_FAMILIES = {
    "weather_climate": "climate",
    "climate_variability": "climate",
    "hydrology": "water",
    "hydrology_water": "water",
    "ocean_climate": "marine",
    "ocean_marine": "marine",
    "earth_science": "earth",
    "earth_observation": "earth",
    "computer_vision": "vision",
    "nlp_text": "text",
    "audio_speech": "audio",
    "health": "health",
    "health_medical": "health",
    "finance": "finance",
    "finance_markets": "finance",
}

USE_CASE_HINTS = {
    "classification": {"classification", "classify", "label", "sentiment", "categorize"},
    "detection": {"detect", "detection", "anomaly", "fraud", "wildfire", "deepfake"},
    "forecasting": {"forecast", "forecasting", "predict", "prediction", "future"},
    "segmentation": {"segmentation", "segment", "mask"},
    "retrieval_search": {"search", "retrieval", "rank", "ranking", "similarity"},
    "recommendation": {"recommendation", "recommender"},
    "ocr_document_ai": {"ocr", "document", "pdf", "markdown"},
    "qa_generation": {"qa", "question", "answering", "instruction", "generation", "chat"},
    "monitoring_analysis": {"monitoring", "analysis", "tracking", "benchmark"},
}

MODALITY_HINTS = {
    "text": {"text", "document", "corpus", "conversation"},
    "tabular": {"table", "tabular", "csv", "spreadsheet"},
    "image": {"image", "images", "vision", "satellite", "xray", "photo"},
    "audio": {"audio", "speech", "voice", "music", "acoustic"},
    "video": {"video", "videos"},
    "time_series": {"time", "timeseries", "temporal", "historical", "forecast"},
    "geospatial": {"geospatial", "gis", "remote", "spatial", "raster"},
    "multimodal": {"multimodal", "visionlanguage", "vision", "language"},
}


def unique_preserve_order(values):
    seen = set()
    ordered = []
    for value in values:
        clean = str(value or "").strip()
        if not clean or clean in seen:
            continue
        ordered.append(clean)
        seen.add(clean)
    return ordered


def build_text_token_set(parts) -> set[str]:
    tokens = []
    for part in parts:
        if isinstance(part, str) and part.strip():
            tokens.extend(tokenize(part))
        elif isinstance(part, list):
            for item in part:
                if str(item or "").strip():
                    tokens.extend(tokenize(str(item)))
    return set(tokens)


def ratio_overlap(query_terms, doc_terms) -> float:
    query_set = set(query_terms or [])
    if not query_set:
        return 0.0
    return len(query_set & set(doc_terms or [])) / len(query_set)


def canonical_domains(domains) -> set[str]:
    canonical = set()
    for domain in domains or []:
        clean = str(domain or "").strip().lower()
        if not clean:
            continue
        canonical.add(DOMAIN_FAMILIES.get(clean, clean))
    return canonical


def domain_alignment(query_domains, doc_domains) -> float:
    query_families = canonical_domains(query_domains)
    doc_families = canonical_domains(doc_domains)
    if not query_families or not doc_families:
        return 0.0
    return len(query_families & doc_families) / len(query_families)


def infer_query_labels(query_plan: dict, hints: dict[str, set[str]]) -> list[str]:
    token_set = set(query_plan.get("focus_terms") or [])
    token_set.update(tokenize(query_plan.get("normalized_query") or ""))
    labels = []
    for label, patterns in hints.items():
        if token_set & patterns:
            labels.append(label)
    return labels


def phrase_bonus(query_plan: dict, item: dict) -> float:
    intent = normalize_text(query_plan.get("intent_body") or query_plan.get("original_query") or "")
    if len(intent.split()) < 2:
        return 0.0
    combined = normalize_text(
        " ".join(
            [
                item.get("title") or "",
                item.get("semantic_summary") or "",
                item.get("description") or item.get("text") or "",
            ]
        )
    )
    return 1.0 if intent and intent in combined else 0.0


def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return {key: 1.0 for key in scores}
    return {
        key: (score - min_score) / (max_score - min_score)
        for key, score in scores.items()
    }


def rerank_candidates(
    query_plan: dict,
    candidates: list[dict],
    top_k: int,
    base_weight: float = 0.52,
    enable_quality_penalty: bool = True,
    quality_weight: float = 0.55,
):
    if not candidates:
        return []

    query_use_cases = infer_query_labels(query_plan, USE_CASE_HINTS)
    query_modalities = infer_query_labels(query_plan, MODALITY_HINTS)
    focus_terms = unique_preserve_order(
        (query_plan.get("raw_focus_terms") or []) + (query_plan.get("concept_terms") or [])
    )
    concept_terms = query_plan.get("concept_terms") or focus_terms
    base_scores = {row["key"]: float(row.get("stage1_score") or 0.0) for row in candidates}
    normalized_base = min_max_normalize(base_scores)

    reranked = []
    for row in candidates:
        item = row["item"]
        title_tokens = build_text_token_set([item.get("title") or ""])
        summary_tokens = build_text_token_set(
            [item.get("semantic_summary") or "", item.get("semantic_text") or ""]
        )
        keyword_tokens = build_text_token_set(
            [item.get("keywords") or [], item.get("metadata_terms") or []]
        )
        description_tokens = build_text_token_set(
            [item.get("description") or item.get("text") or ""]
        )

        base_component = normalized_base.get(row["key"], 0.0)
        title_focus = ratio_overlap(focus_terms, title_tokens)
        summary_concept = ratio_overlap(concept_terms, summary_tokens)
        keyword_concept = ratio_overlap(concept_terms, keyword_tokens)
        description_concept = ratio_overlap(concept_terms, description_tokens)
        raw_context_support = max(title_focus, keyword_concept, description_concept)
        domain_match = domain_alignment(
            query_plan.get("domains") or [],
            item.get("inferred_domains") or [],
        )
        effective_domain_match = domain_match
        if domain_match > 0 and raw_context_support == 0.0:
            # Auto-inferred labels can create false positives such as app-review datasets
            # matching marine queries only because of a product name.
            effective_domain_match *= 0.35
        use_case_match = ratio_overlap(query_use_cases, item.get("inferred_use_cases") or [])
        modality_match = ratio_overlap(query_modalities, item.get("inferred_modalities") or [])
        intent_phrase = phrase_bonus(query_plan, item)
        quality_signal = compute_quality_adjustment(item) if enable_quality_penalty else None

        feature_score = (
            0.30 * effective_domain_match
            + 0.20 * title_focus
            + 0.18 * summary_concept
            + 0.12 * keyword_concept
            + 0.08 * description_concept
            + 0.06 * use_case_match
            + 0.03 * modality_match
            + 0.03 * intent_phrase
        )
        inferred_only_match = (
            summary_concept > 0
            and raw_context_support == 0.0
            and domain_match > 0
        )
        title_only_match = (
            title_focus > 0
            and summary_concept <= 0.2
            and description_concept <= 0.05
            and keyword_concept == 0
        )
        context_penalty = 0.0
        if title_only_match:
            context_penalty += 0.08
        if inferred_only_match:
            context_penalty += 0.12
        rerank_score = base_weight * base_component + (1.0 - base_weight) * feature_score
        rerank_score -= context_penalty
        quality_penalty = 0.0
        quality_bonus = 0.0
        if quality_signal is not None:
            quality_penalty = quality_weight * float(quality_signal.get("score_penalty") or 0.0)
            quality_bonus = quality_weight * float(quality_signal.get("score_bonus") or 0.0)
            if raw_context_support <= 0.1 and quality_signal.get("severity") != "info":
                quality_penalty += 0.015
            rerank_score = rerank_score - quality_penalty + quality_bonus

        enriched = dict(item)
        enriched["rerank_score"] = float(rerank_score)
        enriched["rerank_features"] = {
            "base_component": round(base_component, 4),
            "feature_score": round(feature_score, 4),
            "context_penalty": round(context_penalty, 4),
            "quality_penalty": round(quality_penalty, 4),
            "quality_bonus": round(quality_bonus, 4),
            "domain_match": round(domain_match, 4),
            "effective_domain_match": round(effective_domain_match, 4),
            "title_focus": round(title_focus, 4),
            "summary_concept": round(summary_concept, 4),
            "keyword_concept": round(keyword_concept, 4),
            "description_concept": round(description_concept, 4),
            "raw_context_support": round(raw_context_support, 4),
            "inferred_only_match": round(1.0 if inferred_only_match else 0.0, 4),
            "use_case_match": round(use_case_match, 4),
            "modality_match": round(modality_match, 4),
            "intent_phrase": round(intent_phrase, 4),
        }
        if quality_signal is not None:
            enriched["quality_signal"] = quality_signal
        reranked.append((float(rerank_score), enriched))

    reranked.sort(key=lambda row: row[0], reverse=True)
    return reranked[:top_k]
