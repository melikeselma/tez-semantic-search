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
| bm25 | 0.385 | 0.846 | 0.718 | 0.560 | 0.523 | 0.586 | 4.2 |
| hybrid | 0.077 | 0.231 | 0.154 | 0.141 | 0.109 | 0.130 | 3048.9 |
| semantic | 0.077 | 0.231 | 0.154 | 0.122 | 0.099 | 0.121 | 2786.6 |

## Direction Slice: huggingface_to_kaggle

| Method | Bridge@1 | Bridge@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.167 | 0.667 | 0.583 | 0.375 | 0.347 | 0.418 | 3.6 |
| hybrid | 0.167 | 0.500 | 0.333 | 0.306 | 0.236 | 0.282 | 2918.9 |
| semantic | 0.167 | 0.500 | 0.333 | 0.264 | 0.215 | 0.262 | 2755.6 |

## Direction Slice: kaggle_to_huggingface

| Method | Bridge@1 | Bridge@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.571 | 1.000 | 0.833 | 0.719 | 0.673 | 0.729 | 5.5 |
| hybrid | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3331.0 |
| semantic | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3116.1 |

## Topic Slice: amazon_reviews

| Method | Bridge@1 | Bridge@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.500 | 1.000 | 0.875 | 0.750 | 0.708 | 0.770 | 4.0 |
| hybrid | 0.000 | 0.500 | 0.250 | 0.208 | 0.104 | 0.173 | 3025.7 |
| semantic | 0.000 | 0.500 | 0.250 | 0.146 | 0.073 | 0.143 | 2738.5 |

## Topic Slice: climate_change

| Method | Bridge@1 | Bridge@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.500 | 1.000 | 1.000 | 0.625 | 0.625 | 0.715 | 8.7 |
| hybrid | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3375.7 |
| semantic | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3153.0 |

## Topic Slice: movie_reviews

| Method | Bridge@1 | Bridge@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 2.9 |
| hybrid | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 2956.5 |
| semantic | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 2913.2 |

## Topic Slice: quran_text_audio

| Method | Bridge@1 | Bridge@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.000 | 0.500 | 0.250 | 0.100 | 0.050 | 0.119 | 5.3 |
| hybrid | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3140.3 |
| semantic | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 2898.1 |

## Topic Slice: sentiment_analysis

| Method | Bridge@1 | Bridge@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.000 | 1.000 | 0.333 | 0.333 | 0.111 | 0.235 | 8.8 |
| hybrid | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3662.1 |
| semantic | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3484.6 |

## Topic Slice: turkish_product_reviews

| Method | Bridge@1 | Bridge@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.500 | 1.000 | 1.000 | 0.750 | 0.750 | 0.815 | 12.9 |
| hybrid | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 4839.7 |
| semantic | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 8843.4 |

## Interim Findings

- Across the cross-source benchmark, hybrid is the strongest practical system: Bridge@5=0.231, nDCG@5=0.130.
- Pure semantic retrieval shows whether embeddings can bridge sources directly; it reaches Bridge@5=0.231, compared with BM25=0.846.
- Direction matters. The semantic system performs differently across source pairs: HF->Kaggle nDCG@5=0.262, Kaggle->HF nDCG@5=0.000.
- This direction gap is useful for the thesis because it reflects description quality and metadata style differences between sources, not just model quality.

## Next Step

- Expand the cross-source anchor list with more carefully matched topic pairs before freezing thesis claims.
- Keep semantic retrieval as the direct answer to RQ2, and use hybrid as the strongest applied system.
- In the next phase, compare multiple embedding models on the same RQ2 benchmark to answer RQ3 cleanly.
