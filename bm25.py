import math
from collections import Counter, defaultdict

from query_understanding import build_query_plan, tokenize

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


def build_weighted_query_counts(query: str, plan: dict) -> Counter:
    query_counts = Counter(tokenize(plan.get("original_query") or query))

    for token in plan.get("raw_focus_terms") or []:
        query_counts[token] += 2

    concept_boost = 3 if plan.get("detected_language") == "tr" else 2
    for token in plan.get("concept_terms") or []:
        query_counts[token] += concept_boost

    for token in plan.get("domain_terms") or []:
        query_counts[token] += 1

    return query_counts
