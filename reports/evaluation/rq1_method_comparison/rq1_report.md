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
| bm25 | 0.735 | 0.435 | 0.626 | 0.839 | 0.537 | 0.648 | 2.6 |
| hybrid | 0.912 | 0.512 | 0.752 | 0.949 | 0.695 | 0.788 | 34.2 |
| semantic | 0.824 | 0.453 | 0.655 | 0.900 | 0.606 | 0.705 | 31.6 |

## Benchmark Slice: earthquake_sentence

| Method | P@1 | P@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.700 | 0.420 | 0.783 | 0.787 | 0.594 | 0.698 | 3.2 |
| hybrid | 1.000 | 0.540 | 1.000 | 1.000 | 0.919 | 0.962 | 36.8 |
| semantic | 0.900 | 0.480 | 0.867 | 0.933 | 0.781 | 0.846 | 33.7 |

## Benchmark Slice: general_keyword

| Method | P@1 | P@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.667 | 0.517 | 0.638 | 0.833 | 0.581 | 0.684 | 0.4 |
| hybrid | 0.917 | 0.550 | 0.700 | 0.958 | 0.657 | 0.762 | 25.0 |
| semantic | 0.750 | 0.450 | 0.598 | 0.875 | 0.551 | 0.660 | 23.1 |

## Benchmark Slice: general_sentence

| Method | P@1 | P@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.833 | 0.367 | 0.482 | 0.889 | 0.445 | 0.570 | 2.9 |
| hybrid | 0.833 | 0.450 | 0.598 | 0.896 | 0.547 | 0.667 | 35.5 |
| semantic | 0.833 | 0.433 | 0.536 | 0.896 | 0.515 | 0.633 | 31.8 |

## Query Style Slice: keyword

| Method | P@1 | P@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.667 | 0.517 | 0.638 | 0.833 | 0.581 | 0.684 | 0.4 |
| hybrid | 0.917 | 0.550 | 0.700 | 0.958 | 0.657 | 0.762 | 25.0 |
| semantic | 0.750 | 0.450 | 0.598 | 0.875 | 0.551 | 0.660 | 23.1 |

## Query Style Slice: sentence

| Method | P@1 | P@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.773 | 0.391 | 0.619 | 0.842 | 0.513 | 0.628 | 3.0 |
| hybrid | 0.909 | 0.491 | 0.781 | 0.943 | 0.716 | 0.802 | 36.2 |
| semantic | 0.864 | 0.455 | 0.686 | 0.913 | 0.636 | 0.730 | 33.4 |

## Interim Findings

- Overall, hybrid is the strongest method if the main criterion is ranking quality: nDCG@5=0.788, compared with BM25=0.648 and semantic=0.705.
- Pure semantic retrieval improves over BM25 on sentence-style intent matching, while BM25 remains a useful lexical baseline on short keyword queries.
- On keyword queries, BM25 is still competitive: BM25 MRR=0.833, semantic MRR=0.875.
- On sentence queries, semantic retrieval gains are clearer: semantic nDCG@5=0.730, BM25 nDCG@5=0.628.

## Next Step

- Expand the English benchmark with more keyword and sentence queries before freezing thesis claims.
- Keep BM25 as the content-based baseline and use hybrid as the strongest practical system.
- In the next iteration, add graded relevance labels to make the comparison academically stronger.
