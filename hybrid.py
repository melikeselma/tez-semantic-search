from bm25 import search as bm25_search
from query_understanding import build_query_plan
from search import search as semantic_search
from search_profiles import DEFAULT_PROFILE_KEY

DEFAULT_SEMANTIC_WEIGHT = 0.6
DEFAULT_CANDIDATE_K = 100


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
        ranked.append((float(hybrid_score), enriched))

    ranked.sort(key=lambda row: row[0], reverse=True)
    return ranked[:top_k]
