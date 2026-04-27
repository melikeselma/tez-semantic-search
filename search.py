import json
import math
import sys
from collections import Counter

import faiss
from sentence_transformers import SentenceTransformer

from query_understanding import build_query_plan, tokenize
from quality_scoring import compute_quality_adjustment
from reranker import (
    DEFAULT_CROSS_ENCODER_BATCH_SIZE,
    DEFAULT_CROSS_ENCODER_MODEL_NAME,
    DEFAULT_CROSS_ENCODER_WEIGHT,
    DEFAULT_RERANK_DEPTH,
    STAGE1_STRONG_SCORE,
    STAGE1_WEAK_SCORE,
    confidence_label,
    is_query_out_of_distribution,
    is_query_too_generic,
    rerank_candidates,
)
from runtime_env import ensure_model_cache_dirs
from search_profiles import DEFAULT_PROFILE_KEY, get_profile, get_profile_paths, prepare_query_text

TOP_K = 5
DEFAULT_QUALITY_WEIGHT = 0.12
TR_FUSION_PROFILE = "multilingual"
TR_FUSION_PRIMARY_WEIGHT = 0.72
TR_FUSION_AUX_WEIGHT = 0.58
TR_FUSION_AUX_ONLY_SCALE = 0.62
ASPECT_AVG_WEIGHT = 0.20
ASPECT_MIN_WEIGHT = 0.08
ASPECT_HIT_BONUS = 0.035
ASPECT_MATCH_THRESHOLD = 0.18
ASPECT_MISS_PENALTY = 0.03
_ENGINE_CACHE = {}
# Cache of (token -> document frequency) per profile so we can compute IDF
# of query tokens cheaply at search time. Built lazily from the same mapping
# used by FAISS, so it stays in sync with the indexed corpus.
_CORPUS_DF_CACHE: dict[str, tuple[dict, int]] = {}


def _candidate_text_tokens(item: dict) -> set[str]:
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


def get_corpus_doc_freq(profile_key: str, mappings: dict) -> tuple[dict, int]:
    cached = _CORPUS_DF_CACHE.get(profile_key)
    if cached is not None and cached[1] == len(mappings):
        return cached
    df: Counter[str] = Counter()
    total_docs = 0
    for item in mappings.values():
        total_docs += 1
        for token in _candidate_text_tokens(item):
            df[token] += 1
    _CORPUS_DF_CACHE[profile_key] = (dict(df), total_docs)
    return _CORPUS_DF_CACHE[profile_key]


def compute_token_idf(token: str, doc_freq: dict, total_docs: int) -> float:
    df = int(doc_freq.get(token) or 0)
    if df <= 0:
        # Unseen tokens are maximally informative (the query is asking about
        # something the corpus has never described). Cap so we don't divide by
        # zero in downstream weighting.
        return math.log(1 + total_docs)
    return math.log(1 + (total_docs - df + 0.5) / (df + 0.5))


def configure_output():
    # Prevent Windows console encoding errors when dataset text contains emoji.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_mappings(profile_key: str = DEFAULT_PROFILE_KEY):
    paths = get_profile_paths(profile_key)
    mapping_path = paths["mappings"]
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Mapping file not found: {mapping_path}\n"
            "Run build_faiss_index.py first."
        )
    with mapping_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_search_engine(profile_key: str = DEFAULT_PROFILE_KEY):
    cache_key = str(profile_key or DEFAULT_PROFILE_KEY).strip().lower()
    cached = _ENGINE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    profile = get_profile(profile_key)
    paths = get_profile_paths(profile.key)
    index_path = paths["index"]

    if not index_path.exists():
        raise FileNotFoundError(
            f"Index file not found: {index_path}\n"
            "Run normalize_merge.py and build_faiss_index.py first."
        )

    ensure_model_cache_dirs()

    print(f"[LOG] Yapay zeka modeli yukleniyor: {profile.model_name}")
    model = SentenceTransformer(profile.model_name)

    print(f"[LOG] Vektor veritabani yukleniyor: {index_path}")
    index = faiss.read_index(str(index_path))

    mappings = load_mappings(profile.key)

    if index.ntotal != len(mappings):
        print(
            "[UYARI] Index ve mapping kayit sayisi farkli: "
            f"index={index.ntotal}, mappings={len(mappings)}"
        )

    _ENGINE_CACHE[profile.key] = (model, index, mappings)
    return _ENGINE_CACHE[profile.key]


def normalize_ranked_scores(rows):
    if not rows:
        return []
    values = [float(score) for score, _ in rows]
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return [(1.0, item) for _, item in rows]
    return [
        ((float(score) - min_score) / (max_score - min_score), item)
        for score, item in rows
    ]


def candidate_key(item: dict) -> str:
    return str(item.get("ref") or f"{item.get('source') or 'unknown'}:{item.get('title') or ''}")


def compute_variant_fusion_score(entry: dict) -> float:
    return (
        float(entry.get("max_score") or 0.0)
        + 0.015 * float(entry.get("weight_hits") or 0.0)
        + 0.01 * max(int(entry.get("hits") or 0) - 1, 0)
    )


def compute_aspect_metrics(entry: dict, aspect_keys: list[str]) -> dict:
    if not aspect_keys:
        return {
            "scores": {},
            "avg_score": 0.0,
            "min_score": 0.0,
            "coverage": 0.0,
            "hits": 0,
        }

    scores = {
        key: float((entry.get("aspect_scores") or {}).get(key, 0.0))
        for key in aspect_keys
    }
    score_values = list(scores.values())
    hits = sum(1 for value in score_values if value >= ASPECT_MATCH_THRESHOLD)
    return {
        "scores": scores,
        "avg_score": sum(score_values) / len(score_values),
        "min_score": min(score_values),
        "coverage": hits / len(score_values),
        "hits": hits,
    }


def compute_stage1_fused_score(entry: dict, aspect_keys: list[str]) -> tuple[float, dict]:
    base_score = compute_variant_fusion_score(entry)
    metrics = compute_aspect_metrics(entry, aspect_keys)
    if len(aspect_keys) < 2:
        return base_score, metrics

    # Reward candidates that match multiple semantic aspects of the same
    # dataset-discovery intent instead of over-rewarding a single strong term.
    fused_score = (
        base_score
        + ASPECT_AVG_WEIGHT * metrics["avg_score"]
        + ASPECT_MIN_WEIGHT * metrics["min_score"]
        + ASPECT_HIT_BONUS * max(metrics["hits"] - 1, 0)
        - ASPECT_MISS_PENALTY * max(len(aspect_keys) - metrics["hits"], 0)
    )
    return fused_score, metrics


def collect_stage1_candidates(
    query: str,
    model,
    index,
    mappings,
    top_k: int,
    profile_key: str,
    query_plan: dict,
):
    plan = query_plan or build_query_plan(query)
    active_aspects = [
        aspect
        for aspect in (plan.get("semantic_aspects") or [])
        if str(aspect.get("text") or "").strip()
    ]
    use_multi_aspect = len(active_aspects) >= 2
    aspect_keys = [str(aspect.get("key") or f"aspect_{idx}") for idx, aspect in enumerate(active_aspects)]
    variants = plan.get("semantic_variants") or [
        {"text": text, "weight": 1.0}
        for text in (plan.get("semantic_queries") or [query])
        if text.strip()
    ]
    prepared_queries = [
        prepare_query_text(profile_key, variant["text"])
        for variant in variants
        if variant["text"].strip()
    ]
    variant_weights = [
        float(variant.get("weight", 1.0))
        for variant in variants
        if variant["text"].strip()
    ]
    query_vectors = model.encode(prepared_queries, normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vectors, min(top_k, index.ntotal))

    aspect_queries = []
    if use_multi_aspect:
        aspect_queries = [
            prepare_query_text(profile_key, aspect["text"])
            for aspect in active_aspects
            if str(aspect.get("text") or "").strip()
        ]
    aspect_scores = []
    aspect_indices = []
    if aspect_queries:
        aspect_vectors = model.encode(aspect_queries, normalize_embeddings=True).astype("float32")
        aspect_scores, aspect_indices = index.search(aspect_vectors, min(top_k, index.ntotal))

    fused = {}
    for variant_weight, variant_scores, variant_indices in zip(variant_weights, scores, indices):
        for score, idx in zip(variant_scores, variant_indices):
            if idx == -1:
                continue
            item = mappings.get(str(idx))
            if not item:
                continue
            weighted_score = float(score) * variant_weight
            entry = fused.setdefault(
                str(idx),
                {
                    "item": item,
                    "max_score": weighted_score,
                    "hits": 0,
                    "weight_hits": 0.0,
                    "aspect_scores": {},
                },
            )
            entry["max_score"] = max(entry["max_score"], weighted_score)
            entry["hits"] += 1
            entry["weight_hits"] += variant_weight

    for aspect_key, aspect_score_row, aspect_index_row in zip(aspect_keys, aspect_scores, aspect_indices):
        for score, idx in zip(aspect_score_row, aspect_index_row):
            if idx == -1:
                continue
            item = mappings.get(str(idx))
            if not item:
                continue
            entry = fused.setdefault(
                str(idx),
                {
                    "item": item,
                    "max_score": 0.0,
                    "hits": 0,
                    "weight_hits": 0.0,
                    "aspect_scores": {},
                },
            )
            entry["aspect_scores"][aspect_key] = max(
                float(score),
                float((entry.get("aspect_scores") or {}).get(aspect_key, 0.0)),
            )

    ranked = []
    for idx, entry in enumerate(
        sorted(
            fused.values(),
            key=lambda candidate: compute_stage1_fused_score(candidate, aspect_keys)[0],
            reverse=True,
        ),
        start=1,
    ):
        fused_score, aspect_metrics = compute_stage1_fused_score(entry, aspect_keys)
        item = dict(entry["item"])
        item["semantic_variant_hits"] = entry["hits"]
        item["semantic_aspect_count"] = len(aspect_keys)
        item["semantic_aspect_hits"] = aspect_metrics["hits"]
        item["semantic_aspect_labels"] = [aspect.get("label") or aspect.get("key") for aspect in active_aspects]
        item["semantic_aspect_scores"] = {
            aspect.get("label") or aspect.get("key"): round(float(aspect_metrics["scores"].get(aspect_key, 0.0)), 4)
            for aspect, aspect_key in zip(active_aspects, aspect_keys)
        }
        item["aspect_coverage"] = float(aspect_metrics["coverage"])
        item["aspect_min_score"] = float(aspect_metrics["min_score"])
        item["aspect_avg_score"] = float(aspect_metrics["avg_score"])
        item["stage1_semantic_rank"] = idx
        item["stage1_profile"] = profile_key
        ranked.append((float(fused_score), item))
    return ranked


def search(
    query: str,
    model,
    index,
    mappings,
    top_k: int = TOP_K,
    profile_key: str = DEFAULT_PROFILE_KEY,
    query_plan: dict | None = None,
    enable_rerank: bool = True,
    rerank_depth: int = DEFAULT_RERANK_DEPTH,
    enable_quality_penalty: bool = True,
    quality_weight: float = DEFAULT_QUALITY_WEIGHT,
    enable_tr_fusion: bool = True,
    enable_cross_encoder: bool = False,
    cross_encoder_model_name: str = DEFAULT_CROSS_ENCODER_MODEL_NAME,
    cross_encoder_weight: float = DEFAULT_CROSS_ENCODER_WEIGHT,
    cross_encoder_batch_size: int = DEFAULT_CROSS_ENCODER_BATCH_SIZE,
):
    plan = query_plan or build_query_plan(query)
    search_depth = top_k
    if enable_rerank:
        search_depth = max(top_k, min(rerank_depth, index.ntotal))
    ranked = collect_stage1_candidates(
        query,
        model,
        index,
        mappings,
        top_k=search_depth,
        profile_key=profile_key,
        query_plan=plan,
    )

    if enable_tr_fusion and plan.get("detected_language") == "tr":
        auxiliary_profile = None
        if profile_key != TR_FUSION_PROFILE:
            auxiliary_profile = TR_FUSION_PROFILE
        elif profile_key == TR_FUSION_PROFILE:
            auxiliary_profile = "minilm_ft"
        if auxiliary_profile and auxiliary_profile != profile_key:
            aux_model, aux_index, aux_mappings = load_search_engine(auxiliary_profile)
            auxiliary_ranked = collect_stage1_candidates(
                query,
                aux_model,
                aux_index,
                aux_mappings,
                top_k=search_depth,
                profile_key=auxiliary_profile,
                query_plan=plan,
            )
            normalized_primary = {candidate_key(item): (score, item) for score, item in normalize_ranked_scores(ranked)}
            normalized_aux = {candidate_key(item): (score, item) for score, item in normalize_ranked_scores(auxiliary_ranked)}
            fused_ranked = []
            fused_keys = list(dict.fromkeys(list(normalized_primary.keys()) + list(normalized_aux.keys())))
            for key in fused_keys:
                primary_entry = normalized_primary.get(key)
                aux_entry = normalized_aux.get(key)
                base_item = dict((primary_entry or aux_entry)[1])
                fusion_score = 0.0
                support = 0
                if primary_entry is not None:
                    fusion_score += TR_FUSION_PRIMARY_WEIGHT * float(primary_entry[0])
                    support += 1
                if aux_entry is not None:
                    aux_weight = TR_FUSION_AUX_WEIGHT
                    if primary_entry is None:
                        aux_weight *= TR_FUSION_AUX_ONLY_SCALE
                    fusion_score += aux_weight * float(aux_entry[0])
                    support += 1
                    base_item["tr_aux_profile"] = auxiliary_profile
                if support > 1:
                    fusion_score += 0.03
                    base_item["cross_profile_support"] = True
                fused_ranked.append((fusion_score, base_item))
            ranked = sorted(fused_ranked, key=lambda row: row[0], reverse=True)

    reranked_stage1 = []
    for stage_rank, (score, item) in enumerate(ranked, start=1):
        item = dict(item)
        item["stage1_semantic_rank"] = stage_rank
        quality_signal = compute_quality_adjustment(item) if enable_quality_penalty else None
        adjusted_score = float(score)
        if quality_signal is not None:
            adjusted_score = (
                adjusted_score
                - quality_weight * float(quality_signal.get("score_penalty") or 0.0)
                + quality_weight * float(quality_signal.get("score_bonus") or 0.0)
            )
            item["quality_signal"] = quality_signal
        reranked_stage1.append((adjusted_score, item))

    ranked = sorted(reranked_stage1, key=lambda row: row[0], reverse=True)

    if not enable_rerank:
        # Annotate confidence even when the user disables rerank so the UI
        # can still surface a "weak match" warning on the raw FAISS top-K.
        is_ood = is_query_out_of_distribution(plan)
        stage1_max = max((float(score) for score, _ in ranked), default=0.0)
        confidence = confidence_label(stage1_max, None)
        annotated = []
        for score, item in ranked[:top_k]:
            enriched = dict(item)
            enriched["query_confidence"] = confidence
            enriched["query_is_ood"] = bool(is_ood)
            enriched["weak_match"] = bool(float(score) < STAGE1_WEAK_SCORE and confidence != "strong")
            annotated.append((float(score), enriched))
        return annotated

    candidates = []
    for rank, (score, item) in enumerate(ranked[:search_depth], start=1):
        item_with_stage = dict(item)
        item_with_stage["stage1_semantic_score"] = float(score)
        item_with_stage["stage1_semantic_rank"] = rank
        key = item.get("ref") or f"{item.get('source') or 'unknown'}:{item.get('title') or ''}"
        candidates.append(
            {
                "key": str(key),
                "item": item_with_stage,
                "stage1_score": float(score),
            }
        )

    corpus_doc_freq, corpus_total_docs = get_corpus_doc_freq(profile_key, mappings)
    return rerank_candidates(
        plan,
        candidates,
        top_k=top_k,
        base_weight=0.52,
        enable_quality_penalty=enable_quality_penalty,
        quality_weight=0.18,
        enable_cross_encoder=enable_cross_encoder,
        cross_encoder_model_name=cross_encoder_model_name,
        cross_encoder_weight=cross_encoder_weight,
        cross_encoder_batch_size=cross_encoder_batch_size,
        corpus_doc_freq=corpus_doc_freq,
        corpus_total_docs=corpus_total_docs,
    )


def main():
    configure_output()
    model, index, mappings = load_search_engine()

    print("\n" + "=" * 50)
    print("   SEMANTIK VERI SETI ARAMA MOTORUNA HOS GELDINIZ")
    print("=" * 50)

    while True:
        query = input("\nNe tur bir veri seti ariyorsunuz? (Cikis icin 'q'): ").strip()
        if query.lower() == "q":
            break
        if not query:
            continue

        print(f"\n'{query}' icin en alakali sonuclar:")
        print("-" * 50)

        for rank, (score, result) in enumerate(search(query, model, index, mappings), start=1):
            desc = (result.get("text") or "").replace("\n", " ")
            print(f"{rank}. BASLIK: {result.get('title', 'Isimsiz')}")
            print(f"   KAYNAK: {(result.get('source') or 'unknown').upper()}")
            print(f"   SKOR: {score:.4f}")
            print(f"   URL: {result.get('url') or 'N/A'}")
            print(f"   OZET: {desc[:180]}...")
            print("-" * 50)


if __name__ == "__main__":
    main()
