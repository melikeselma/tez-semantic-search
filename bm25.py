import math
from collections import Counter, defaultdict

from query_understanding import STOPWORDS, build_query_plan, tokenize

TITLE_WEIGHT = 3
KEYWORD_WEIGHT = 2
SKIP_KEYWORD_PREFIXES = {
    "format:",
    "language:",
    "library:",
    "license:",
    "modality:",
    "region:",
    "size_categories:",
    "task_categories:",
}
# Tokens that look like topic words but are actually platform/listing tags
# present in many descriptions, so they hand BM25 spurious matches like
# "Reddit Australia" appearing for queries that just contain the word "research".
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
    "sample",
    "samples",
    "training",
    "test",
    "validation",
    "label",
    "labels",
    "type",
    "types",
    "benchmark",
    "benchmarks",
    "evaluation",
    "evaluating",
    "system",
    "systems",
    "corpus",
    "collection",
    "file",
    "files",
}


def iter_keyword_texts(item: dict):
    for keyword in item.get("keywords") or []:
        raw = str(keyword or "").strip().lower()
        if not raw:
            continue
        if any(raw.startswith(prefix) for prefix in SKIP_KEYWORD_PREFIXES):
            continue
        yield raw


def build_document_tokens(item: dict) -> list[str]:
    title_tokens = tokenize(item.get("title", ""))
    description = item.get("description") or item.get("text", "")
    description_tokens = tokenize(description)

    keyword_tokens = []
    for keyword in iter_keyword_texts(item):
        keyword_tokens.extend(tokenize(keyword))

    return title_tokens * TITLE_WEIGHT + keyword_tokens * KEYWORD_WEIGHT + description_tokens


class BM25Index:
    def __init__(self, mappings: dict, k1: float = 1.5, b: float = 0.75):
        self.mappings = mappings
        self.k1 = k1
        self.b = b
        self.doc_ids = []
        self.doc_lengths = []
        self.inverted_index = defaultdict(list)
        self.idf = {}
        self.avg_doc_length = 0.0
        self._build()

    def _build(self):
        doc_frequencies = Counter()

        for doc_id, item in self.mappings.items():
            tokens = build_document_tokens(item)
            if not tokens:
                continue

            doc_index = len(self.doc_ids)
            term_frequencies = Counter(tokens)
            self.doc_ids.append(doc_id)
            self.doc_lengths.append(len(tokens))

            for token, frequency in term_frequencies.items():
                self.inverted_index[token].append((doc_index, frequency))
                doc_frequencies[token] += 1

        total_docs = len(self.doc_ids)
        if total_docs == 0:
            return

        self.avg_doc_length = sum(self.doc_lengths) / total_docs
        self.idf = {
            token: math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            for token, df in doc_frequencies.items()
        }

    def search(self, query: str, top_k: int = 5, query_plan: dict | None = None) -> list[tuple[float, dict]]:
        plan = query_plan or build_query_plan(query)
        query_counts = build_weighted_query_counts(query, plan)
        if not query_counts or not self.doc_ids:
            return []

        scores = defaultdict(float)

        for token, query_frequency in query_counts.items():
            postings = self.inverted_index.get(token)
            if not postings:
                continue

            idf = self.idf.get(token, 0.0)
            for doc_index, term_frequency in postings:
                doc_length = self.doc_lengths[doc_index]
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * doc_length / self.avg_doc_length
                )
                token_score = idf * (
                    term_frequency * (self.k1 + 1) / denominator
                )
                scores[doc_index] += query_frequency * token_score

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        results = []
        for doc_index, score in ranked:
            item = self.mappings.get(self.doc_ids[doc_index])
            if item:
                results.append((float(score), item))
        return results


def search(query: str, bm25_index: BM25Index, top_k: int = 5, query_plan: dict | None = None):
    return bm25_index.search(query, top_k=top_k, query_plan=query_plan)


def _is_meaningful_query_token(token: str) -> bool:
    if not token or len(token) < 2:
        return False
    if token in STOPWORDS:
        return False
    if token in GENERIC_NOISE_TERMS:
        return False
    return True


def build_weighted_query_counts(query: str, plan: dict) -> Counter:
    raw_focus = list(plan.get("raw_focus_terms") or [])
    concept_terms = list(plan.get("concept_terms") or [])
    domain_terms = list(plan.get("domain_terms") or [])

    query_counts = Counter()

    # Original-query tokens come into BM25 only as a thin baseline because
    # un-filtered words like "research", "want", "on" used to match Reddit /
    # Republicans datasets purely on stopword overlap.
    for token in tokenize(plan.get("original_query") or query):
        if _is_meaningful_query_token(token):
            query_counts[token] += 1

    for token in raw_focus:
        if _is_meaningful_query_token(token):
            query_counts[token] += 2

    concept_boost = 3 if plan.get("detected_language") == "tr" else 2
    for token in concept_terms:
        if _is_meaningful_query_token(token):
            query_counts[token] += concept_boost

    for token in domain_terms:
        if _is_meaningful_query_token(token):
            query_counts[token] += 1

    # Defensive fallback: if the cleaning above wiped every token (e.g. the
    # query is literally "research data"), keep a minimal raw token set so
    # BM25 still has something to score, otherwise we silently return nothing.
    if not query_counts:
        for token in tokenize(plan.get("original_query") or query):
            if token and len(token) >= 2:
                query_counts[token] += 1

    return query_counts
