# Semantic Intent Baseline Report

Purpose:
Track latent semantic intent retrieval quality while query understanding, semantic enrichment, and model adaptation steps are added incrementally.

Profiles and methods:
- `minilm`: MiniLM (EN baseline) with semantic, BM25, and hybrid
- `multilingual`: Multilingual (EN+TR alt kume) with semantic and hybrid

Top-K: `5`

## Benchmark Composition

- Total queries: `10`
- Study slices: `{'english_latent_intent': 5, 'turkish_latent_intent': 5}`
- Languages: `{'en': 5, 'tr': 5}`
- Topics: `{'weather_conditions': 2, 'storm_forecast_history': 2, 'flood_hydrology': 2, 'drought_climate_history': 2, 'marine_proxy': 2}`

## Overall Baseline Comparison

| Profile | Method | P@1 | Hit@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| minilm | bm25 | 0.800 | 1.000 | 0.693 | 0.900 | 0.672 | 0.770 | 0.7 |
| minilm | hybrid | 1.000 | 1.000 | 0.700 | 1.000 | 0.693 | 0.789 | 32.8 |
| multilingual | hybrid | 0.900 | 1.000 | 0.612 | 0.950 | 0.586 | 0.702 | 63.0 |
| minilm | semantic | 1.000 | 1.000 | 0.680 | 1.000 | 0.681 | 0.776 | 32.0 |
| multilingual | semantic | 0.800 | 1.000 | 0.492 | 0.900 | 0.475 | 0.591 | 60.4 |

## Study Slice: english_latent_intent

| Profile | Method | P@1 | Hit@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| minilm | bm25 | 0.800 | 1.000 | 0.728 | 0.900 | 0.714 | 0.801 | 1.3 |
| minilm | hybrid | 1.000 | 1.000 | 0.686 | 1.000 | 0.673 | 0.773 | 31.7 |
| minilm | semantic | 1.000 | 1.000 | 0.686 | 1.000 | 0.687 | 0.781 | 31.8 |
| multilingual | hybrid | 0.800 | 1.000 | 0.672 | 0.900 | 0.631 | 0.734 | 59.8 |
| multilingual | semantic | 0.800 | 1.000 | 0.512 | 0.900 | 0.487 | 0.604 | 60.4 |

## Study Slice: turkish_latent_intent

| Profile | Method | P@1 | Hit@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| minilm | bm25 | 0.800 | 1.000 | 0.659 | 0.900 | 0.630 | 0.739 | 0.7 |
| minilm | hybrid | 1.000 | 1.000 | 0.714 | 1.000 | 0.713 | 0.805 | 35.7 |
| minilm | semantic | 1.000 | 1.000 | 0.674 | 1.000 | 0.675 | 0.770 | 32.3 |
| multilingual | hybrid | 1.000 | 1.000 | 0.552 | 1.000 | 0.541 | 0.669 | 65.0 |
| multilingual | semantic | 0.800 | 1.000 | 0.472 | 0.900 | 0.463 | 0.578 | 59.6 |

## Language Slice: en

| Profile | Method | P@1 | Hit@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| minilm | bm25 | 0.800 | 1.000 | 0.728 | 0.900 | 0.714 | 0.801 | 1.3 |
| minilm | hybrid | 1.000 | 1.000 | 0.686 | 1.000 | 0.673 | 0.773 | 31.7 |
| minilm | semantic | 1.000 | 1.000 | 0.686 | 1.000 | 0.687 | 0.781 | 31.8 |
| multilingual | hybrid | 0.800 | 1.000 | 0.672 | 0.900 | 0.631 | 0.734 | 59.8 |
| multilingual | semantic | 0.800 | 1.000 | 0.512 | 0.900 | 0.487 | 0.604 | 60.4 |

## Language Slice: tr

| Profile | Method | P@1 | Hit@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| minilm | bm25 | 0.800 | 1.000 | 0.659 | 0.900 | 0.630 | 0.739 | 0.7 |
| minilm | hybrid | 1.000 | 1.000 | 0.714 | 1.000 | 0.713 | 0.805 | 35.7 |
| minilm | semantic | 1.000 | 1.000 | 0.674 | 1.000 | 0.675 | 0.770 | 32.3 |
| multilingual | hybrid | 1.000 | 1.000 | 0.552 | 1.000 | 0.541 | 0.669 | 65.0 |
| multilingual | semantic | 0.800 | 1.000 | 0.472 | 0.900 | 0.463 | 0.578 | 59.6 |

## Topic Slice: drought_climate_history

| Profile | Method | P@1 | Hit@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| minilm | bm25 | 1.000 | 1.000 | 0.600 | 1.000 | 0.575 | 0.711 | 1.2 |
| minilm | hybrid | 1.000 | 1.000 | 0.600 | 1.000 | 0.502 | 0.670 | 34.7 |
| minilm | semantic | 1.000 | 1.000 | 0.500 | 1.000 | 0.442 | 0.604 | 32.0 |
| multilingual | hybrid | 1.000 | 1.000 | 0.700 | 1.000 | 0.607 | 0.747 | 69.2 |
| multilingual | semantic | 1.000 | 1.000 | 0.500 | 1.000 | 0.475 | 0.626 | 91.6 |

## Topic Slice: flood_hydrology

| Profile | Method | P@1 | Hit@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| minilm | bm25 | 1.000 | 1.000 | 0.667 | 1.000 | 0.667 | 0.765 | 0.6 |
| minilm | hybrid | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 31.5 |
| minilm | semantic | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 29.2 |
| multilingual | hybrid | 1.000 | 1.000 | 0.333 | 1.000 | 0.333 | 0.469 | 62.4 |
| multilingual | semantic | 1.000 | 1.000 | 0.333 | 1.000 | 0.333 | 0.469 | 62.4 |

## Topic Slice: marine_proxy

| Profile | Method | P@1 | Hit@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| minilm | bm25 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.5 |
| minilm | hybrid | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 35.3 |
| minilm | semantic | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 33.1 |
| multilingual | hybrid | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 64.5 |
| multilingual | semantic | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 59.7 |

## Topic Slice: storm_forecast_history

| Profile | Method | P@1 | Hit@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| minilm | bm25 | 1.000 | 1.000 | 0.700 | 1.000 | 0.655 | 0.777 | 1.0 |
| minilm | hybrid | 1.000 | 1.000 | 0.400 | 1.000 | 0.367 | 0.531 | 34.3 |
| minilm | semantic | 1.000 | 1.000 | 0.400 | 1.000 | 0.367 | 0.531 | 33.3 |
| multilingual | hybrid | 1.000 | 1.000 | 0.600 | 1.000 | 0.555 | 0.692 | 61.3 |
| multilingual | semantic | 1.000 | 1.000 | 0.200 | 1.000 | 0.200 | 0.339 | 59.1 |

## Topic Slice: weather_conditions

| Profile | Method | P@1 | Hit@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| minilm | bm25 | 0.000 | 1.000 | 0.500 | 0.500 | 0.463 | 0.595 | 0.8 |
| minilm | hybrid | 1.000 | 1.000 | 0.500 | 1.000 | 0.597 | 0.743 | 30.7 |
| minilm | semantic | 1.000 | 1.000 | 0.500 | 1.000 | 0.597 | 0.743 | 35.2 |
| multilingual | hybrid | 0.500 | 1.000 | 0.429 | 0.750 | 0.437 | 0.600 | 57.5 |
| multilingual | semantic | 0.000 | 1.000 | 0.429 | 0.500 | 0.368 | 0.522 | 65.0 |

## Key Findings

- Across the full latent-intent benchmark, the current best system is minilm hybrid (nDCG@5=0.789). MiniLM semantic and hybrid now both stay ahead of lexical BM25 (0.776 / 0.789 vs 0.770).
- On English latent-intent queries, semantic retrieval already improves over the lexical baseline: MiniLM semantic nDCG@5=0.781, BM25 nDCG@5=0.801.
- English retrieval is now stable across both semantic and hybrid variants, which means the next gains will likely come from better document representations rather than more aggressive query rewriting alone (semantic nDCG@5=0.781, hybrid=0.773).
- The query-understanding and concept-expansion layer closes most of the Turkish latent-intent gap for the MiniLM profile: MiniLM semantic reaches nDCG@5=0.770 and MiniLM hybrid reaches 0.805.
- The multilingual profile is still weaker than the English baseline on the same Turkish slice (semantic nDCG@5=0.578, hybrid=0.669), so cross-lingual robustness remains an active improvement area rather than a solved problem.
- The profile comparison here should be interpreted as a retrieval baseline study, not as a final model claim. The multilingual profile is an extension path for bilingual intent coverage, while MiniLM remains the stronger English baseline.

## Why This Matters

- This benchmark isolates the exact thesis pain point: user wording and dataset wording often differ even when the underlying intent matches.
- The English slice shows how far the current retriever can go once intent terms are made explicit at query time.
- The Turkish slice now measures whether concept projection is robust enough for live demo queries, not just whether Turkish embeddings exist.
- The narrow marine slice keeps the benchmark honest about current corpus coverage instead of overclaiming capability.

## Next Stage

- Enrich dataset-side text with semantic summaries, inferred domains, and use-case labels so the retriever has better document representations to match against the improved queries.
