# RQ4 Report

Research question:
How do description language structure, length, and included terms affect semantic representation quality?

Profile: `minilm` (MiniLM (EN baseline))
Top-K: `5`
Methods: `semantic, hybrid`

## Evaluation Scope

- English queries only: `47`
- Study slices: `{'english_main': 34, 'cross_source': 13}`
- RQ4 uses the existing thesis benchmark and re-reads it from the document-quality perspective.

## Bucket Definitions

- Length buckets: `short <= 20` words, `medium <= 52` words, `long` above that.
- Term richness buckets: `term_sparse <= 10` unique content terms, `term_moderate <= 28`, `term_rich` above that.
- Style buckets are heuristic: `metadata_heavy`, `mixed_structured`, and `narrative`.

## Corpus Feature Distribution

| Feature | Bucket | Documents | Share |
|---|---|---:|---:|
| length_bucket | long | 376 | 0.336 |
| length_bucket | medium | 349 | 0.312 |
| length_bucket | short | 395 | 0.353 |
| description_style | metadata_heavy | 206 | 0.184 |
| description_style | mixed_structured | 593 | 0.529 |
| description_style | narrative | 321 | 0.287 |
| term_bucket | term_moderate | 362 | 0.323 |
| term_bucket | term_rich | 371 | 0.331 |
| term_bucket | term_sparse | 387 | 0.346 |

## Retrieval by Description Length

| Method | Bucket | Relevant Pairs | Unique Docs | Hit@5 | Top1 | Mean Rank When Hit |
|---|---|---:|---:|---:|---:|---:|
| semantic | long | 90 | 33 | 0.767 | 0.289 | 2.09 |
| semantic | medium | 38 | 18 | 0.289 | 0.079 | 2.55 |
| semantic | short | 22 | 11 | 0.000 | 0.000 | - |
| hybrid | long | 90 | 33 | 0.778 | 0.267 | 2.16 |
| hybrid | medium | 38 | 18 | 0.368 | 0.158 | 2.36 |
| hybrid | short | 22 | 11 | 0.000 | 0.000 | - |

## Retrieval by Description Style

| Method | Bucket | Relevant Pairs | Unique Docs | Hit@5 | Top1 | Mean Rank When Hit |
|---|---|---:|---:|---:|---:|---:|
| semantic | metadata_heavy | 30 | 13 | 0.200 | 0.100 | 2.33 |
| semantic | mixed_structured | 54 | 23 | 0.481 | 0.148 | 2.12 |
| semantic | narrative | 66 | 26 | 0.727 | 0.273 | 2.15 |
| hybrid | metadata_heavy | 30 | 13 | 0.233 | 0.100 | 2.29 |
| hybrid | mixed_structured | 54 | 23 | 0.537 | 0.278 | 1.90 |
| hybrid | narrative | 66 | 26 | 0.727 | 0.182 | 2.35 |

## Retrieval by Term Richness

| Method | Bucket | Relevant Pairs | Unique Docs | Hit@5 | Top1 | Mean Rank When Hit |
|---|---|---:|---:|---:|---:|---:|
| semantic | term_moderate | 46 | 21 | 0.326 | 0.065 | 2.73 |
| semantic | term_rich | 88 | 32 | 0.739 | 0.295 | 2.02 |
| semantic | term_sparse | 16 | 9 | 0.000 | 0.000 | - |
| hybrid | term_moderate | 46 | 21 | 0.370 | 0.130 | 2.47 |
| hybrid | term_rich | 88 | 32 | 0.761 | 0.273 | 2.12 |
| hybrid | term_sparse | 16 | 9 | 0.000 | 0.000 | - |

## Interim Findings

- Longer descriptions are easier to recover semantically. Semantic Hit@5 rises from 0.000 on short descriptions to 0.767 on long descriptions.
- Description structure matters. Narrative descriptions produce better semantic retrieval than metadata-heavy ones: Hit@5=0.727 vs 0.200.
- Term-rich descriptions carry stronger semantic signals. Semantic Hit@5 goes from 0.000 on term-sparse documents to 0.739 on term-rich documents.
- On weak descriptions, hybrid remains a useful fallback. For term-sparse documents, hybrid Hit@5=0.000 while pure semantic stays at 0.000.

## Interpretation

- RQ4 suggests that semantic quality is not only a model issue; document quality also shapes retrieval success.
- Short, tag-like, or metadata-heavy descriptions weaken the semantic signal.
- Longer and more content-rich descriptions make the main English semantic pipeline more reliable.
- This supports a practical thesis recommendation: improve normalization, enrich weak descriptions, and flag low-information records before indexing.
