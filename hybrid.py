from bm25 import search as bm25_search
from query_understanding import build_query_plan
from quality_scoring import compute_quality_adjustment
from reranker import (
    DEFAULT_CROSS_ENCODER_BATCH_SIZE,
    DEFAULT_CROSS_ENCODER_MODEL_NAME,
    DEFAULT_CROSS_ENCODER_WEIGHT,
    DEFAULT_RERANK_DEPTH,
    rerank_candidates,
)
from search import get_corpus_doc_freq, search as semantic_search
from search_profiles import DEFAULT_PROFILE_KEY

DEFAULT_SEMANTIC_WEIGHT = 0.6
DEFAULT_CANDIDATE_K = 100
DEFAULT_QUALITY_WEIGHT = 0.08


def result_key(item: dict) -> str:
    ref = item.get("ref")
    if ref:
        return str(ref)
    return f"{item.get('source') or 'unknown'}:{item.get('title') or ''}"


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


def search(
    query: str,
    model,
    index,
    mappings: dict,
    bm25_index,
    top_k: int = 5,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    candidate_k: int = DEFAULT_CANDIDATE_K,
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
    semantic_weight = max(0.0, min(1.0, semantic_weight))
    bm25_weight = 1.0 - semantic_weight
    candidate_k = max(top_k, min(candidate_k, len(mappings)))
    plan = query_plan or build_query_plan(query)

    semantic_results = semantic_search(
        query,
        model,
        index,
        mappings,
        top_k=candidate_k,
        profile_key=profile_key,
        query_plan=plan,
        enable_rerank=False,
        enable_quality_penalty=enable_quality_penalty,
        enable_tr_fusion=enable_tr_fusion,
    )
    bm25_results = bm25_search(query, bm25_index, top_k=candidate_k, query_plan=plan)

    items = {}
    semantic_scores = {}
    bm25_scores = {}

    for score, item in semantic_results:
        key = result_key(item)
        items[key] = item
        semantic_scores[key] = float(score)

    for score, item in bm25_results:
        key = result_key(item)
        items[key] = item
        bm25_scores[key] = float(score)

    normalized_semantic = min_max_normalize(semantic_scores)
    normalized_bm25 = min_max_normalize(bm25_scores)

    ranked = []
    for key, item in items.items():
        hybrid_score = (
            semantic_weight * normalized_semantic.get(key, 0.0)
            + bm25_weight * normalized_bm25.get(key, 0.0)
        )
        enriched = dict(item)
        enriched["semantic_score"] = semantic_scores.get(key)
        enriched["bm25_score"] = bm25_scores.get(key)
        enriched["semantic_weight"] = semantic_weight
        enriched["bm25_weight"] = bm25_weight
        quality_signal = compute_quality_adjustment(enriched) if enable_quality_penalty else None
        adjusted_score = float(hybrid_score)
        if quality_signal is not None:
            adjusted_score = (
                adjusted_score
                - quality_weight * float(quality_signal.get("score_penalty") or 0.0)
                + quality_weight * float(quality_signal.get("score_bonus") or 0.0)
            )
            enriched["quality_signal"] = quality_signal
        ranked.append((adjusted_score, enriched))

    ranked.sort(key=lambda row: row[0], reverse=True)
    if not enable_rerank:
        return ranked[:top_k]

    rerank_limit = max(top_k, min(rerank_depth, len(ranked)))
    candidates = []
    for rank, (score, item) in enumerate(ranked[:rerank_limit], start=1):
        item_with_stage = dict(item)
        item_with_stage["stage1_hybrid_score"] = float(score)
        item_with_stage["stage1_hybrid_rank"] = rank
        candidates.append(
            {
                "key": result_key(item),
                "item": item_with_stage,
                "stage1_score": float(score),
            }
        )

    corpus_doc_freq, corpus_total_docs = get_corpus_doc_freq(profile_key, mappings)
    return rerank_candidates(
        plan,
        candidates,
        top_k=top_k,
        base_weight=0.68,
        enable_quality_penalty=enable_quality_penalty,
        quality_weight=0.12,
        enable_cross_encoder=enable_cross_encoder,
        cross_encoder_model_name=cross_encoder_model_name,
        cross_encoder_weight=cross_encoder_weight,
        cross_encoder_batch_size=cross_encoder_batch_size,
        corpus_doc_freq=corpus_doc_freq,
        corpus_total_docs=corpus_total_docs,
    )
