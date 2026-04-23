# RQ3 Report

Research question:
How does the type and size of the language model affect semantic similarity results?

Profiles compared:
- `minilm`: MiniLM (EN baseline)
- `multilingual`: Multilingual (EN+TR alt kume)

Top-K: `5`
Methods: `semantic, hybrid`

## Benchmark Composition

- Total queries: `59`
- Study slices: `{'english_main': 34, 'cross_source': 13, 'tr_subset': 12}`
- Languages: `{'en': 47, 'tr': 12}`

## Overall Profile Comparison

| Profile | Method | P@1 | Hit@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| minilm | hybrid | 0.678 | 0.814 | 0.611 | 0.736 | 0.552 | 0.622 | 25.9 |
| multilingual | hybrid | 0.644 | 0.780 | 0.551 | 0.710 | 0.503 | 0.574 | 49.0 |
| minilm | semantic | 0.610 | 0.780 | 0.544 | 0.688 | 0.493 | 0.563 | 23.0 |
| multilingual | semantic | 0.492 | 0.644 | 0.418 | 0.559 | 0.372 | 0.432 | 46.2 |

## Study Slice: cross_source

| Profile | Method | P@1 | Hit@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| minilm | hybrid | 0.615 | 0.846 | 0.744 | 0.705 | 0.643 | 0.687 | 69.5 |
| minilm | semantic | 0.615 | 0.846 | 0.744 | 0.731 | 0.645 | 0.695 | 58.6 |
| multilingual | hybrid | 0.462 | 0.615 | 0.519 | 0.538 | 0.490 | 0.510 | 121.7 |
| multilingual | semantic | 0.308 | 0.462 | 0.365 | 0.372 | 0.304 | 0.335 | 114.8 |

## Study Slice: english_main

| Profile | Method | P@1 | Hit@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| minilm | hybrid | 0.912 | 1.000 | 0.752 | 0.949 | 0.695 | 0.788 | 25.4 |
| minilm | semantic | 0.824 | 1.000 | 0.655 | 0.900 | 0.606 | 0.705 | 23.0 |
| multilingual | hybrid | 0.853 | 0.971 | 0.705 | 0.909 | 0.634 | 0.731 | 48.3 |
| multilingual | semantic | 0.676 | 0.824 | 0.563 | 0.740 | 0.508 | 0.585 | 45.5 |

## Study Slice: tr_subset

| Profile | Method | P@1 | Hit@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| minilm | hybrid | 0.083 | 0.250 | 0.067 | 0.167 | 0.046 | 0.082 | 17.2 |
| minilm | semantic | 0.000 | 0.083 | 0.014 | 0.042 | 0.008 | 0.018 | 16.6 |
| multilingual | hybrid | 0.250 | 0.417 | 0.149 | 0.333 | 0.143 | 0.199 | 32.4 |
| multilingual | semantic | 0.167 | 0.333 | 0.067 | 0.250 | 0.061 | 0.103 | 31.3 |

## Language Slice: en

| Profile | Method | P@1 | Hit@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| minilm | hybrid | 0.830 | 0.957 | 0.750 | 0.881 | 0.681 | 0.760 | 26.9 |
| minilm | semantic | 0.766 | 0.957 | 0.680 | 0.853 | 0.617 | 0.702 | 23.5 |
| multilingual | hybrid | 0.745 | 0.872 | 0.653 | 0.807 | 0.594 | 0.670 | 51.7 |
| multilingual | semantic | 0.574 | 0.723 | 0.508 | 0.638 | 0.452 | 0.516 | 48.5 |

## Language Slice: tr

| Profile | Method | P@1 | Hit@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| minilm | hybrid | 0.083 | 0.250 | 0.067 | 0.167 | 0.046 | 0.082 | 17.2 |
| minilm | semantic | 0.000 | 0.083 | 0.014 | 0.042 | 0.008 | 0.018 | 16.6 |
| multilingual | hybrid | 0.250 | 0.417 | 0.149 | 0.333 | 0.143 | 0.199 | 32.4 |
| multilingual | semantic | 0.167 | 0.333 | 0.067 | 0.250 | 0.061 | 0.103 | 31.3 |

## Interim Findings

- On the main English benchmark, the compact English-specific model remains stronger: MiniLM semantic nDCG@5=0.705, Multilingual semantic nDCG@5=0.585.
- On the Turkish subset, the multilingual profile is clearly better: Multilingual semantic nDCG@5=0.103, MiniLM semantic nDCG@5=0.018.
- For cross-source similarity, model choice also changes bridging quality: MiniLM semantic nDCG@5=0.695, Multilingual semantic nDCG@5=0.335.
- Within the current thesis scope, the comparison mainly captures model type and training objective, not a large size jump, because both active profiles are lightweight models.

## Interpretation

- MiniLM should be treated as the main English thesis model.
- Multilingual should be treated as the EN+TR extension profile, not as the default English system.
- This supports the thesis framing: one main English pipeline, plus a smaller multilingual extension for the bilingual subset.
