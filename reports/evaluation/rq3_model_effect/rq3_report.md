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
| minilm | hybrid | 0.559 | 0.712 | 0.483 | 0.621 | 0.434 | 0.503 | 378.5 |
| multilingual | hybrid | 0.576 | 0.746 | 0.494 | 0.645 | 0.445 | 0.512 | 476.3 |
| minilm | semantic | 0.576 | 0.695 | 0.463 | 0.617 | 0.426 | 0.492 | 274.8 |
| multilingual | semantic | 0.525 | 0.746 | 0.496 | 0.613 | 0.424 | 0.497 | 360.7 |

## Study Slice: cross_source

| Profile | Method | P@1 | Hit@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| minilm | hybrid | 0.077 | 0.231 | 0.154 | 0.141 | 0.109 | 0.130 | 2950.7 |
| minilm | semantic | 0.077 | 0.231 | 0.154 | 0.122 | 0.099 | 0.121 | 2768.8 |
| multilingual | hybrid | 0.154 | 0.308 | 0.269 | 0.218 | 0.205 | 0.227 | 3591.1 |
| multilingual | semantic | 0.077 | 0.308 | 0.269 | 0.147 | 0.140 | 0.178 | 3078.1 |

## Study Slice: english_main

| Profile | Method | P@1 | Hit@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| minilm | hybrid | 0.824 | 0.941 | 0.662 | 0.877 | 0.621 | 0.708 | 343.2 |
| minilm | semantic | 0.824 | 0.941 | 0.638 | 0.870 | 0.602 | 0.689 | 259.1 |
| multilingual | hybrid | 0.794 | 0.941 | 0.623 | 0.855 | 0.584 | 0.666 | 410.9 |
| multilingual | semantic | 0.765 | 0.941 | 0.606 | 0.841 | 0.558 | 0.648 | 319.1 |

## Study Slice: tr_subset

| Profile | Method | P@1 | Hit@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| minilm | hybrid | 0.333 | 0.583 | 0.329 | 0.412 | 0.255 | 0.323 | 323.6 |
| minilm | semantic | 0.417 | 0.500 | 0.302 | 0.438 | 0.280 | 0.332 | 246.3 |
| multilingual | hybrid | 0.417 | 0.667 | 0.369 | 0.514 | 0.312 | 0.385 | 380.0 |
| multilingual | semantic | 0.333 | 0.667 | 0.430 | 0.472 | 0.348 | 0.418 | 288.1 |

## Language Slice: en

| Profile | Method | P@1 | Hit@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| minilm | hybrid | 0.617 | 0.745 | 0.522 | 0.674 | 0.480 | 0.548 | 382.0 |
| minilm | semantic | 0.617 | 0.745 | 0.504 | 0.663 | 0.463 | 0.532 | 277.6 |
| multilingual | hybrid | 0.617 | 0.766 | 0.525 | 0.679 | 0.479 | 0.545 | 482.8 |
| multilingual | semantic | 0.574 | 0.766 | 0.513 | 0.649 | 0.443 | 0.518 | 368.7 |

## Language Slice: tr

| Profile | Method | P@1 | Hit@5 | Recall@5 | MRR | MAP@5 | nDCG@5 | Median ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| minilm | hybrid | 0.333 | 0.583 | 0.329 | 0.412 | 0.255 | 0.323 | 323.6 |
| minilm | semantic | 0.417 | 0.500 | 0.302 | 0.438 | 0.280 | 0.332 | 246.3 |
| multilingual | hybrid | 0.417 | 0.667 | 0.369 | 0.514 | 0.312 | 0.385 | 380.0 |
| multilingual | semantic | 0.333 | 0.667 | 0.430 | 0.472 | 0.348 | 0.418 | 288.1 |

## Interim Findings

- On the main English benchmark, the compact English-specific model remains stronger: MiniLM semantic nDCG@5=0.689, Multilingual semantic nDCG@5=0.648.
- On the Turkish subset, the multilingual profile is clearly better: Multilingual semantic nDCG@5=0.418, MiniLM semantic nDCG@5=0.332.
- For cross-source similarity, model choice also changes bridging quality: MiniLM semantic nDCG@5=0.121, Multilingual semantic nDCG@5=0.178.
- Within the current thesis scope, the comparison mainly captures model type and training objective, not a large size jump, because both active profiles are lightweight models.

## Interpretation

- MiniLM should be treated as the main English thesis model.
- Multilingual should be treated as the EN+TR extension profile, not as the default English system.
- This supports the thesis framing: one main English pipeline, plus a smaller multilingual extension for the bilingual subset.
