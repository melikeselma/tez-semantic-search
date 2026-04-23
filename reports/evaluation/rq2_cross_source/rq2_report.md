# RQ2 Report

Research question:
How successful is an embedding-based system at establishing semantic similarity between datasets collected from different sources?

Profile: `minilm` (MiniLM (EN baseline))
Top-K: `5`
Methods: `semantic, bm25, hybrid`

## Benchmark Composition

- Total anchors: `13`
- Directions: `{'huggingface_to_kaggle': 6, 'kaggle_to_huggingface': 7}`
- Anchor sources: `{'huggingface': 6, 'kaggle': 7}`
- Topics: `{'movie_reviews': 2, 'amazon_reviews': 4, 'turkish_product_reviews': 2, 'climate_change': 2, 'quran_text_audio': 2, 'sentiment_analysis': 1}`

## Overall Cross-Source Comparison

| Method | Bridge@1 | Bridge@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.385 | 0.462 | 0.462 | 0.410 | 0.417 | 0.429 | 6.5 |
| hybrid | 0.615 | 0.846 | 0.744 | 0.705 | 0.643 | 0.687 | 65.1 |
| semantic | 0.615 | 0.846 | 0.744 | 0.731 | 0.645 | 0.695 | 61.1 |

## Direction Slice: huggingface_to_kaggle

| Method | Bridge@1 | Bridge@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.333 | 0.500 | 0.500 | 0.389 | 0.403 | 0.428 | 2.9 |
| hybrid | 0.500 | 0.667 | 0.583 | 0.556 | 0.472 | 0.519 | 58.3 |
| semantic | 0.500 | 0.667 | 0.583 | 0.583 | 0.500 | 0.541 | 57.4 |

## Direction Slice: kaggle_to_huggingface

| Method | Bridge@1 | Bridge@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.429 | 0.429 | 0.429 | 0.429 | 0.429 | 0.429 | 10.0 |
| hybrid | 0.714 | 1.000 | 0.881 | 0.833 | 0.790 | 0.832 | 103.4 |
| semantic | 0.714 | 1.000 | 0.881 | 0.857 | 0.770 | 0.828 | 86.7 |

## Topic Slice: amazon_reviews

| Method | Bridge@1 | Bridge@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.750 | 1.000 | 1.000 | 0.833 | 0.854 | 0.893 | 4.7 |
| hybrid | 1.000 | 1.000 | 0.875 | 1.000 | 0.875 | 0.903 | 61.4 |
| semantic | 1.000 | 1.000 | 0.875 | 1.000 | 0.812 | 0.873 | 53.7 |

## Topic Slice: climate_change

| Method | Bridge@1 | Bridge@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 7.6 |
| hybrid | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 104.1 |
| semantic | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 91.2 |

## Topic Slice: movie_reviews

| Method | Bridge@1 | Bridge@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 3.4 |
| hybrid | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 47.5 |
| semantic | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 48.6 |

## Topic Slice: quran_text_audio

| Method | Bridge@1 | Bridge@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 8.4 |
| hybrid | 0.000 | 1.000 | 0.750 | 0.417 | 0.292 | 0.443 | 87.0 |
| semantic | 0.000 | 1.000 | 0.750 | 0.500 | 0.375 | 0.509 | 73.9 |

## Topic Slice: sentiment_analysis

| Method | Bridge@1 | Bridge@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 12.1 |
| hybrid | 0.000 | 1.000 | 0.667 | 0.333 | 0.278 | 0.437 | 154.1 |
| semantic | 0.000 | 1.000 | 0.667 | 0.500 | 0.389 | 0.531 | 139.1 |

## Topic Slice: turkish_product_reviews

| Method | Bridge@1 | Bridge@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 9.9 |
| hybrid | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 111.3 |
| semantic | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.0 |

## Interim Findings

- Across the cross-source benchmark, hybrid is the strongest practical system: Bridge@5=0.846, nDCG@5=0.687.
- Pure semantic retrieval shows whether embeddings can bridge sources directly; it reaches Bridge@5=0.846, compared with BM25=0.462.
- Direction matters. The semantic system performs differently across source pairs: HF->Kaggle nDCG@5=0.541, Kaggle->HF nDCG@5=0.828.
- This direction gap is useful for the thesis because it reflects description quality and metadata style differences between sources, not just model quality.

## Next Step

- Expand the cross-source anchor list with more carefully matched topic pairs before freezing thesis claims.
- Keep semantic retrieval as the direct answer to RQ2, and use hybrid as the strongest applied system.
- In the next phase, compare multiple embedding models on the same RQ2 benchmark to answer RQ3 cleanly.
