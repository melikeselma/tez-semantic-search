# RQ1 Report

Research question:
Can semantic data discovery be performed using dataset descriptions, and how effective is this approach compared with content-based methods?

Profile: `minilm` (MiniLM (EN baseline))
Top-K: `5`
Methods: `semantic, bm25, hybrid`

## Benchmark Composition

- Total queries: `34`
- Benchmarks: `{'general_keyword': 12, 'general_sentence': 12, 'earthquake_sentence': 10}`
- Query styles: `{'keyword': 12, 'sentence': 22}`

## Overall Method Comparison

| Method | P@1 | P@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.735 | 0.459 | 0.659 | 0.849 | 0.598 | 0.689 | 1.4 |
| hybrid | 0.824 | 0.459 | 0.662 | 0.877 | 0.621 | 0.708 | 344.2 |
| semantic | 0.824 | 0.441 | 0.638 | 0.870 | 0.602 | 0.689 | 279.6 |

## Benchmark Slice: earthquake_sentence

| Method | P@1 | P@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 1.000 | 0.500 | 0.933 | 1.000 | 0.906 | 0.934 | 1.8 |
| hybrid | 1.000 | 0.520 | 0.967 | 1.000 | 0.950 | 0.969 | 415.8 |
| semantic | 0.900 | 0.520 | 0.967 | 0.950 | 0.908 | 0.938 | 316.7 |

## Benchmark Slice: general_keyword

| Method | P@1 | P@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.500 | 0.483 | 0.589 | 0.736 | 0.499 | 0.608 | 1.1 |
| hybrid | 0.750 | 0.433 | 0.536 | 0.833 | 0.490 | 0.606 | 282.4 |
| semantic | 0.833 | 0.383 | 0.480 | 0.854 | 0.472 | 0.579 | 275.0 |

## Benchmark Slice: general_sentence

| Method | P@1 | P@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.750 | 0.400 | 0.501 | 0.836 | 0.439 | 0.565 | 1.6 |
| hybrid | 0.750 | 0.433 | 0.536 | 0.819 | 0.479 | 0.594 | 308.7 |
| semantic | 0.750 | 0.433 | 0.522 | 0.819 | 0.478 | 0.593 | 238.8 |

## Query Style Slice: keyword

| Method | P@1 | P@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.500 | 0.483 | 0.589 | 0.736 | 0.499 | 0.608 | 1.1 |
| hybrid | 0.750 | 0.433 | 0.536 | 0.833 | 0.490 | 0.606 | 282.4 |
| semantic | 0.833 | 0.383 | 0.480 | 0.854 | 0.472 | 0.579 | 275.0 |

## Query Style Slice: sentence

| Method | P@1 | P@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.864 | 0.445 | 0.698 | 0.911 | 0.651 | 0.733 | 1.7 |
| hybrid | 0.864 | 0.473 | 0.732 | 0.902 | 0.693 | 0.764 | 383.5 |
| semantic | 0.818 | 0.473 | 0.724 | 0.879 | 0.673 | 0.749 | 284.7 |

## Interim Findings

- Overall, hybrid is the strongest method if the main criterion is ranking quality: nDCG@5=0.708, compared with BM25=0.689 and semantic=0.689.
- Pure semantic retrieval improves over BM25 on sentence-style intent matching, while BM25 remains a useful lexical baseline on short keyword queries.
- On keyword queries, BM25 is still competitive: BM25 MRR=0.736, semantic MRR=0.854.
- On sentence queries, semantic retrieval gains are clearer: semantic nDCG@5=0.749, BM25 nDCG@5=0.733.

## Next Step

- Expand the English benchmark with more keyword and sentence queries before freezing thesis claims.
- Keep BM25 as the content-based baseline and use hybrid as the strongest practical system.
- In the next iteration, add graded relevance labels to make the comparison academically stronger.
