import math

from sentence_transformers import CrossEncoder

from query_understanding import STOPWORDS, normalize_text, tokenize
from quality_scoring import compute_quality_adjustment, should_deemphasize_title
from runtime_env import ensure_model_cache_dirs

# Same generic noise list the BM25 query builder uses; tokens like "research"
# show up in too many descriptions to be useful as topical anchors.
GENERIC_NOISE_TERMS = {
    "research",
    "researcher",
    "researchers",
    "study",
    "studies",
    "analysis",
    "analyses",
    "global",
    "intermediate",
    "advanced",
    "beginner",
    "kaggle",
    "huggingface",
    "dataset",
    "datasets",
    "open",
    "free",
    "public",
    "people",
    "users",
    "use",
    "task",
    "tasks",
    "model",
    "models",
    "data",
    "image",
    "images",
    "text",
    "texts",
    "video",
    "videos",
    "audio",
    "record",
    "records",
    "set",
    "sets",
    "year",
    "years",
    "month",
    "day",
    "time",
    "size",
    "sample",
    "samples",
    "training",
    "test",
    "validation",
    "label",
    "labels",
    "categories",
    "category",
    "type",
    "types",
    # Frame words that look topical but appear in too many descriptions to
    # discriminate.
    "benchmark",
    "benchmarks",
    "evaluation",
    "evaluating",
    "system",
    "systems",
    "corpus",
    "collection",
    "instance",
    "instances",
    "row",
    "rows",
    "column",
    "columns",
    "feature",
    "features",
    "table",
    "tables",
    "files",
    "file",
}
# Hard penalty applied when the strongest topical anchor of a query is fully
# absent from a candidate's text. Calibrated so semantically-tempting but
# topically-wrong matches (pokemon-images for "spider species") are pushed
# below honest weak-match candidates.
NO_ANCHOR_PENALTY = 0.55
PARTIAL_ANCHOR_PENALTY = 0.20
# Reward documents that cover many of the query's high-IDF anchor tokens.
# A perfect cover (every anchor present) adds this much; partial covers add
# proportionally. Calibrated to be smaller than NO_ANCHOR_PENALTY so a doc
# with all anchors still loses to a doc that has them AND high cosine.
ANCHOR_COVERAGE_BONUS = 0.22
SECONDARY_COVERAGE_BONUS = 0.08

DEFAULT_RERANK_DEPTH = 40
DEFAULT_CROSS_ENCODER_MODEL_NAME = "BAAI/bge-reranker-base"
FALLBACK_CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CROSS_ENCODER_WEIGHT = 0.34
# Heuristic features (domain_match, concept overlap) all collapse to zero when
# the query plan has no domains/concepts, so giving the cross-encoder more
# weight in that branch is the only way to keep the rerank meaningful.
OOD_CROSS_ENCODER_WEIGHT = 0.72
DEFAULT_CROSS_ENCODER_BATCH_SIZE = 16
DEFAULT_CROSS_ENCODER_MAX_LENGTH = 512
RERANK_TEXT_MAX_CHARS = 3200
# Confidence thresholds tuned against the curated probe set in
# dev_tools/regression_probe.py. A stage1 cosine ceiling below the strong line
# means even the best FAISS candidate is only loosely related to the query;
# the cross-encoder raw threshold is calibrated on bge-reranker-base.
STAGE1_STRONG_SCORE = 0.62
STAGE1_WEAK_SCORE = 0.50
CROSS_ENCODER_STRONG_RAW = 0.0
CROSS_ENCODER_WEAK_RAW = -3.5
_CROSS_ENCODER_CACHE = {}
_CROSS_ENCODER_FAILURES = {}

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


def clean_rerank_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def build_rerank_document_text(item: dict) -> str:
    # This keeps reranking semantic: the cross-encoder compares the natural
    # language query against the best available semantic description of each
    # dataset, rather than using lexical keyword matching.
    for field in ("semantic_text", "description", "text", "semantic_summary"):
        value = clean_rerank_text(item.get(field) or "")
        if value:
            return value[:RERANK_TEXT_MAX_CHARS]
    fallback = clean_rerank_text(item.get("title") or "")
    return fallback[:RERANK_TEXT_MAX_CHARS]


def min_max_normalize_list(values: list[float]) -> list[float]:
    if not values:
        return []
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return [1.0 for _ in values]
    return [
        (float(value) - min_score) / (max_score - min_score)
        for value in values
    ]


def load_cross_encoder_model(model_name: str = DEFAULT_CROSS_ENCODER_MODEL_NAME):
    requested = str(model_name or DEFAULT_CROSS_ENCODER_MODEL_NAME).strip()
    candidate_models = [requested]
    if requested != FALLBACK_CROSS_ENCODER_MODEL_NAME:
        candidate_models.append(FALLBACK_CROSS_ENCODER_MODEL_NAME)

    errors = []
    ensure_model_cache_dirs()
    for candidate in candidate_models:
        cached = _CROSS_ENCODER_CACHE.get(candidate)
        if cached is not None:
            return cached, candidate
        cached_error = _CROSS_ENCODER_FAILURES.get(candidate)
        if cached_error is not None:
            errors.append(f"{candidate}: {cached_error}")
            continue
        try:
            model = CrossEncoder(
                candidate,
                max_length=DEFAULT_CROSS_ENCODER_MAX_LENGTH,
            )
            _CROSS_ENCODER_CACHE[candidate] = model
            _CROSS_ENCODER_FAILURES.pop(candidate, None)
            return model, candidate
        except Exception as exc:
            error_text = str(exc)
            _CROSS_ENCODER_FAILURES[candidate] = error_text
            errors.append(f"{candidate}: {error_text}")

    raise RuntimeError(
        "Cross-encoder reranker could not be loaded. Tried: "
        + " | ".join(errors)
    )


def score_cross_encoder_pairs(
    query_text: str,
    candidates: list[dict],
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL_NAME,
    batch_size: int = DEFAULT_CROSS_ENCODER_BATCH_SIZE,
):
    query_text = clean_rerank_text(query_text)
    if not query_text or not candidates:
        return {
            "scores": {},
            "model_name": None,
            "error": None,
        }

    pairs = []
    keys = []
    for row in candidates:
        document_text = build_rerank_document_text(row.get("item") or {})
        if not document_text:
            continue
        keys.append(row["key"])
        pairs.append((query_text, document_text))

    if not pairs:
        return {
            "scores": {},
            "model_name": None,
            "error": None,
        }

    try:
        model, resolved_model_name = load_cross_encoder_model(model_name)
        raw_scores = model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        raw_scores = [float(score) for score in raw_scores]
        normalized_scores = min_max_normalize_list(raw_scores)
        return {
            "scores": {
                key: {
                    "raw": raw_score,
                    "normalized": normalized_score,
                }
                for key, raw_score, normalized_score in zip(keys, raw_scores, normalized_scores)
            },
            "model_name": resolved_model_name,
            "error": None,
        }
    except Exception as exc:
        return {
            "scores": {},
            "model_name": None,
            "error": str(exc),
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


def select_query_anchor_tokens(query_plan: dict, doc_freq: dict | None, total_docs: int) -> tuple[set[str], set[str]]:
    """Return (primary_anchors, secondary_anchors) for the query.

    Primary anchors are the high-IDF, content-bearing tokens of the query;
    they MUST appear in a candidate for it to be considered a real match.
    Secondary anchors are the rest of the focus terms — their absence is a
    softer signal. Generic descriptors ("species", "research") are excluded.
    """

    raw_focus = list(query_plan.get("raw_focus_terms") or [])
    concept_terms = list(query_plan.get("concept_terms") or [])
    candidates = []
    seen = set()
    for token in raw_focus + concept_terms:
        if not token or token in seen:
            continue
        seen.add(token)
        if len(token) < 3:
            continue
        if token in STOPWORDS or token in GENERIC_NOISE_TERMS:
            continue
        candidates.append(token)

    if not candidates:
        return set(), set()

    if doc_freq is None or total_docs <= 0:
        return set(candidates[:2]), set(candidates)

    scored = []
    for token in candidates:
        df = int((doc_freq or {}).get(token) or 0)
        idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5)) if df > 0 else math.log(1 + total_docs)
        scored.append((token, idf))
    scored.sort(key=lambda row: row[1], reverse=True)
    if not scored:
        return set(), set()
    top_idf = scored[0][1]
    # Anything within a small band of the top IDF is treated as primary; the
    # rest fall to secondary so a query like "spider species" anchors on
    # "spider" but does not penalise candidates that *also* have "species".
    primary = {token for token, idf in scored if idf >= max(top_idf - 0.6, 0.0)}
    secondary = {token for token, _ in scored if token not in primary}
    return primary, secondary


def candidate_text_token_set(item: dict) -> set[str]:
    parts = [
        item.get("title") or "",
        item.get("semantic_summary") or "",
        item.get("semantic_text") or "",
        item.get("description") or item.get("text") or "",
        " ".join(item.get("keywords") or []),
        " ".join(item.get("metadata_terms") or []),
    ]
    tokens: set[str] = set()
    for part in parts:
        if part:
            tokens.update(tokenize(part))
    return tokens


def lexical_anchor_signal(query_plan: dict, item: dict, doc_freq: dict | None, total_docs: int) -> dict:
    primary, secondary = select_query_anchor_tokens(query_plan, doc_freq, total_docs)
    if not primary and not secondary:
        return {
            "has_primary": True,
            "has_secondary": True,
            "missing_primary": [],
            "missing_secondary": [],
            "penalty": 0.0,
            "primary_anchors": [],
            "secondary_anchors": [],
        }
    doc_tokens = candidate_text_token_set(item)
    matched_primary = primary & doc_tokens
    matched_secondary = secondary & doc_tokens
    has_primary = bool(matched_primary)
    has_secondary = bool(matched_secondary) or not secondary
    penalty = 0.0
    bonus = 0.0
    if primary and not has_primary:
        # Tier the penalty so the heuristic can distinguish "no anchor at all"
        # (Reddit Australia, PyTorch wheels, opendatalab common-crawl) from
        # "primary missing but at least one secondary anchor present"
        # (pokemon/inaturalist on a spider query, both have "species").
        if secondary and matched_secondary:
            penalty += PARTIAL_ANCHOR_PENALTY
        else:
            penalty += NO_ANCHOR_PENALTY
    if primary and has_primary:
        coverage = len(matched_primary) / max(len(primary), 1)
        bonus += ANCHOR_COVERAGE_BONUS * coverage
    if secondary and matched_secondary:
        secondary_coverage = len(matched_secondary) / max(len(secondary), 1)
        bonus += SECONDARY_COVERAGE_BONUS * secondary_coverage
    return {
        "has_primary": has_primary,
        "has_secondary": has_secondary,
        "missing_primary": sorted(primary - doc_tokens),
        "missing_secondary": sorted(secondary - doc_tokens),
        "matched_primary": sorted(matched_primary),
        "matched_secondary": sorted(matched_secondary),
        "penalty": penalty,
        "bonus": bonus,
        "primary_anchors": sorted(primary),
        "secondary_anchors": sorted(secondary),
    }


def is_query_too_generic(query_plan: dict) -> bool:
    """A query is "too generic" if after stripping stopwords, generic noise,
    and one-character tokens, nothing meaningful is left to discriminate on.

    Examples:
      - "x"
      - "i want a dataset"
      - "datasets that are large"
      - ""
    Such queries should not return anything; they would just float corpus
    centroid noise to the top of the list.
    """
    if not query_plan:
        return True
    raw_focus = query_plan.get("raw_focus_terms") or []
    concept = query_plan.get("concept_terms") or []
    meaningful = [
        token
        for token in (list(raw_focus) + list(concept))
        if token
        and len(token) >= 3
        and token not in STOPWORDS
        and token not in GENERIC_NOISE_TERMS
    ]
    return len(meaningful) == 0


def is_query_out_of_distribution(query_plan: dict) -> bool:
    # An "out-of-distribution" query is one that triggered no curated phrase /
    # combo / token rules. In that case the heuristic ranker has no signal of
    # its own and shouldn't dominate the cross-encoder.
    if not query_plan:
        return True
    return not (
        (query_plan.get("domains") or [])
        or (query_plan.get("concept_terms") or [])
        or (query_plan.get("semantic_aspects") or [])
    )


def confidence_label(stage1_max: float, cross_encoder_max_raw: float | None) -> str:
    if stage1_max >= STAGE1_STRONG_SCORE and (
        cross_encoder_max_raw is None or cross_encoder_max_raw >= CROSS_ENCODER_STRONG_RAW
    ):
        return "strong"
    if stage1_max < STAGE1_WEAK_SCORE or (
        cross_encoder_max_raw is not None and cross_encoder_max_raw <= CROSS_ENCODER_WEAK_RAW
    ):
        return "weak"
    return "moderate"


def rerank_candidates(
    query_plan: dict,
    candidates: list[dict],
    top_k: int,
    base_weight: float = 0.52,
    enable_quality_penalty: bool = True,
    quality_weight: float = 0.55,
    enable_cross_encoder: bool = False,
    cross_encoder_model_name: str = DEFAULT_CROSS_ENCODER_MODEL_NAME,
    cross_encoder_weight: float = DEFAULT_CROSS_ENCODER_WEIGHT,
    cross_encoder_batch_size: int = DEFAULT_CROSS_ENCODER_BATCH_SIZE,
    corpus_doc_freq: dict | None = None,
    corpus_total_docs: int = 0,
):
    if not candidates:
        return []

    is_ood = is_query_out_of_distribution(query_plan)
    if is_ood and enable_cross_encoder:
        cross_encoder_weight = max(cross_encoder_weight, OOD_CROSS_ENCODER_WEIGHT)
    cross_encoder_weight = max(0.0, min(1.0, float(cross_encoder_weight)))
    # Whether to actually trust the cross-encoder weight is decided after we
    # see the raw scores: if the model's best score is near 0 it cannot
    # discriminate, and giving it 70% weight just promotes whichever doc the
    # encoder happened to be slightly less confused by (e.g. "epstein-files"
    # for "research on spiders").
    raw_ce_weight = cross_encoder_weight
    query_use_cases = infer_query_labels(query_plan, USE_CASE_HINTS)
    query_modalities = infer_query_labels(query_plan, MODALITY_HINTS)
    focus_terms = unique_preserve_order(
        (query_plan.get("raw_focus_terms") or []) + (query_plan.get("concept_terms") or [])
    )
    concept_terms = query_plan.get("concept_terms") or focus_terms
    base_scores = {row["key"]: float(row.get("stage1_score") or 0.0) for row in candidates}
    raw_stage1_scores = [
        float(row.get("item", {}).get("stage1_semantic_score") or row.get("stage1_score") or 0.0)
        for row in candidates
    ]
    stage1_max = max(raw_stage1_scores) if raw_stage1_scores else 0.0
    normalized_base = min_max_normalize(base_scores)
    cross_encoder_payload = {
        "scores": {},
        "model_name": None,
        "error": None,
    }
    if enable_cross_encoder:
        cross_encoder_payload = score_cross_encoder_pairs(
            query_plan.get("original_query") or query_plan.get("normalized_query") or "",
            candidates,
            model_name=cross_encoder_model_name,
            batch_size=cross_encoder_batch_size,
        )

    cross_encoder_raw_values = [
        float(meta.get("raw"))
        for meta in (cross_encoder_payload.get("scores") or {}).values()
        if isinstance(meta, dict) and meta.get("raw") is not None
    ]
    cross_encoder_max_raw = max(cross_encoder_raw_values) if cross_encoder_raw_values else None
    if enable_cross_encoder and cross_encoder_max_raw is not None and cross_encoder_max_raw < 0.05:
        # The cross-encoder is essentially abstaining; do not let it dominate
        # the rerank score in that case.
        cross_encoder_weight = min(cross_encoder_weight, 0.20)
    confidence = confidence_label(stage1_max, cross_encoder_max_raw if enable_cross_encoder else None)

    primary_anchors, secondary_anchors = select_query_anchor_tokens(
        query_plan, corpus_doc_freq, corpus_total_docs
    )
    anchor_active = bool(primary_anchors)

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
        quality_signal = compute_quality_adjustment(item) if enable_quality_penalty else None
        title_deemphasized = should_deemphasize_title(item, quality_signal)
        effective_title_focus = title_focus * 0.2 if title_deemphasized else title_focus
        raw_context_support = max(effective_title_focus, keyword_concept, description_concept)
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

        feature_score = (
            0.30 * effective_domain_match
            + 0.20 * effective_title_focus
            + 0.18 * summary_concept
            + 0.12 * keyword_concept
            + 0.08 * description_concept
            + 0.06 * use_case_match
            + 0.03 * modality_match
            + 0.03 * intent_phrase
        )
        anchor_signal = lexical_anchor_signal(
            query_plan, item, corpus_doc_freq, corpus_total_docs
        )
        # Low-information datasets (description ~ empty, only title + a topic
        # tag) live near the corpus centroid in embedding space and surface as
        # weak top-3 noise for every OOD query. Cap that explicitly.
        is_low_info_doc = bool(
            quality_signal
            and {"low_information", "keyword_only", "metadata_heavy", "empty"} & set(quality_signal.get("all_flags") or [])
        )
        if is_low_info_doc and anchor_active and not anchor_signal.get("has_primary"):
            anchor_signal = dict(anchor_signal)
            anchor_signal["penalty"] = float(anchor_signal.get("penalty") or 0.0) + 0.20
        inferred_only_match = (
            summary_concept > 0
            and raw_context_support == 0.0
            and domain_match > 0
        )
        title_only_match = (
            effective_title_focus > 0
            and summary_concept <= 0.2
            and description_concept <= 0.05
            and keyword_concept == 0
        )
        context_penalty = 0.0
        if title_only_match:
            context_penalty += 0.08
            if title_deemphasized:
                context_penalty += 0.05
        if inferred_only_match:
            context_penalty += 0.12
        rerank_score = base_weight * base_component + (1.0 - base_weight) * feature_score
        rerank_score -= context_penalty
        anchor_penalty = float(anchor_signal.get("penalty") or 0.0)
        anchor_bonus = float(anchor_signal.get("bonus") or 0.0)
        if anchor_penalty:
            rerank_score -= anchor_penalty
        if anchor_bonus:
            rerank_score += anchor_bonus
        quality_penalty = 0.0
        quality_bonus = 0.0
        if quality_signal is not None:
            quality_penalty = quality_weight * float(quality_signal.get("score_penalty") or 0.0)
            quality_bonus = quality_weight * float(quality_signal.get("score_bonus") or 0.0)
            if raw_context_support <= 0.1 and quality_signal.get("severity") != "info":
                quality_penalty += 0.015
            rerank_score = rerank_score - quality_penalty + quality_bonus

        heuristic_rerank_score = float(rerank_score)
        cross_encoder_meta = (cross_encoder_payload.get("scores") or {}).get(row["key"]) or {}
        cross_encoder_raw = cross_encoder_meta.get("raw")
        cross_encoder_norm = float(cross_encoder_meta.get("normalized") or 0.0)
        cross_encoder_active = bool(enable_cross_encoder and cross_encoder_payload.get("model_name"))
        final_rerank_score = heuristic_rerank_score
        if cross_encoder_active:
            # The cross-encoder is a second-stage semantic aligner on top of
            # FAISS retrieval. It improves query-document fit without turning
            # the system into a lexical or BM25-driven ranker.
            final_rerank_score = (
                (1.0 - cross_encoder_weight) * heuristic_rerank_score
                + cross_encoder_weight * cross_encoder_norm
            )

        per_item_weak = False
        per_item_strong = False
        if cross_encoder_active and cross_encoder_raw is not None:
            if cross_encoder_raw <= CROSS_ENCODER_WEAK_RAW:
                per_item_weak = True
            elif cross_encoder_raw >= CROSS_ENCODER_STRONG_RAW:
                per_item_strong = True
        stage1_score_value = float(item.get("stage1_semantic_score") or item.get("stage1_hybrid_score") or row.get("stage1_score") or 0.0)
        if stage1_score_value < STAGE1_WEAK_SCORE and not per_item_strong:
            per_item_weak = True
        if anchor_active and not anchor_signal.get("has_primary"):
            per_item_weak = True
            per_item_strong = False

        enriched = dict(item)
        enriched["rerank_score"] = float(final_rerank_score)
        enriched["semantic_rerank_score"] = float(final_rerank_score)
        enriched["heuristic_rerank_score"] = float(heuristic_rerank_score)
        enriched["cross_encoder_score"] = float(cross_encoder_raw) if cross_encoder_raw is not None else None
        enriched["cross_encoder_score_norm"] = float(cross_encoder_norm) if cross_encoder_active else None
        enriched["cross_encoder_model"] = cross_encoder_payload.get("model_name")
        enriched["cross_encoder_enabled"] = bool(enable_cross_encoder)
        enriched["cross_encoder_error"] = cross_encoder_payload.get("error")
        enriched["weak_match"] = bool(per_item_weak)
        enriched["query_confidence"] = confidence
        enriched["query_is_ood"] = bool(is_ood)
        enriched["lexical_anchor_signal"] = {
            "has_primary": bool(anchor_signal.get("has_primary")),
            "missing_primary": list(anchor_signal.get("missing_primary") or []),
            "matched_primary": list(anchor_signal.get("matched_primary") or []),
            "matched_secondary": list(anchor_signal.get("matched_secondary") or []),
            "primary_anchors": list(anchor_signal.get("primary_anchors") or []),
            "penalty": round(anchor_penalty, 4),
            "bonus": round(anchor_bonus, 4),
        }
        enriched["rerank_features"] = {
            "base_component": round(base_component, 4),
            "feature_score": round(feature_score, 4),
            "context_penalty": round(context_penalty, 4),
            "quality_penalty": round(quality_penalty, 4),
            "quality_bonus": round(quality_bonus, 4),
            "domain_match": round(domain_match, 4),
            "effective_domain_match": round(effective_domain_match, 4),
            "title_focus": round(title_focus, 4),
            "effective_title_focus": round(effective_title_focus, 4),
            "title_deemphasized": title_deemphasized,
            "summary_concept": round(summary_concept, 4),
            "keyword_concept": round(keyword_concept, 4),
            "description_concept": round(description_concept, 4),
            "raw_context_support": round(raw_context_support, 4),
            "inferred_only_match": round(1.0 if inferred_only_match else 0.0, 4),
            "use_case_match": round(use_case_match, 4),
            "modality_match": round(modality_match, 4),
            "intent_phrase": round(intent_phrase, 4),
            "heuristic_rerank_score": round(heuristic_rerank_score, 4),
            "cross_encoder_score": round(float(cross_encoder_raw), 4) if cross_encoder_raw is not None else None,
            "cross_encoder_score_norm": round(cross_encoder_norm, 4) if cross_encoder_active else None,
            "cross_encoder_weight": round(cross_encoder_weight, 4) if cross_encoder_active else 0.0,
        }
        if quality_signal is not None:
            enriched["quality_signal"] = quality_signal
        reranked.append((float(final_rerank_score), enriched))

    reranked.sort(key=lambda row: row[0], reverse=True)
    return reranked[:top_k]
