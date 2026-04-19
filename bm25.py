import math
import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


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
            tokens = tokenize(item.get("text", ""))
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

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, dict]]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.doc_ids:
            return []

        scores = defaultdict(float)
        query_counts = Counter(query_tokens)

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


def search(query: str, bm25_index: BM25Index, top_k: int = 5):
    return bm25_index.search(query, top_k=top_k)
