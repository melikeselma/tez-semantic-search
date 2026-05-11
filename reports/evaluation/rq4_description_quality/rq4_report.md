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

- Length buckets: `short <= 22` words, `medium <= 60` words, `long` above that.
- Term richness buckets: `term_sparse <= 11` unique content terms, `term_moderate <= 32`, `term_rich` above that.
- Style buckets are heuristic: `metadata_heavy`, `mixed_structured`, and `narrative`.

## Corpus Feature Distribution

| Feature | Bucket | Documents | Share |
|---|---|---:|---:|
| length_bucket | long | 423 | 0.339 |
| length_bucket | medium | 400 | 0.321 |
| length_bucket | short | 425 | 0.341 |
| description_style | metadata_heavy | 251 | 0.201 |
| description_style | mixed_structured | 630 | 0.505 |
| description_style | narrative | 367 | 0.294 |
| term_bucket | term_moderate | 407 | 0.326 |
| term_bucket | term_rich | 421 | 0.337 |
| term_bucket | term_sparse | 420 | 0.337 |

## Retrieval by Description Length

| Method | Bucket | Relevant Pairs | Unique Docs | Hit@5 | Top1 | Mean Rank When Hit |
|---|---|---:|---:|---:|---:|---:|
| semantic | long | 86 | 31 | 0.767 | 0.302 | 2.09 |
| semantic | medium | 41 | 19 | 0.293 | 0.073 | 2.75 |
| semantic | short | 23 | 12 | 0.000 | 0.000 | - |
| hybrid | long | 86 | 31 | 0.779 | 0.279 | 2.18 |
| hybrid | medium | 41 | 19 | 0.341 | 0.122 | 2.50 |
| hybrid | short | 23 | 12 | 0.000 | 0.000 | - |

## Retrieval by Description Style

| Method | Bucket | Relevant Pairs | Unique Docs | Hit@5 | Top1 | Mean Rank When Hit |
|---|---|---:|---:|---:|---:|---:|
| semantic | metadata_heavy | 30 | 13 | 0.167 | 0.100 | 2.00 |
| semantic | mixed_structured | 54 | 23 | 0.481 | 0.148 | 2.19 |
| semantic | narrative | 66 | 26 | 0.712 | 0.273 | 2.21 |
| hybrid | metadata_heavy | 30 | 13 | 0.200 | 0.100 | 2.00 |
| hybrid | mixed_structured | 54 | 23 | 0.519 | 0.278 | 1.86 |
| hybrid | narrative | 66 | 26 | 0.712 | 0.167 | 2.49 |

## Retrieval by Term Richness

| Method | Bucket | Relevant Pairs | Unique Docs | Hit@5 | Top1 | Mean Rank When Hit |
|---|---|---:|---:|---:|---:|---:|
| semantic | term_moderate | 47 | 21 | 0.255 | 0.064 | 2.75 |
| semantic | term_rich | 86 | 31 | 0.767 | 0.302 | 2.09 |
| semantic | term_sparse | 17 | 10 | 0.000 | 0.000 | - |
| hybrid | term_moderate | 47 | 21 | 0.298 | 0.106 | 2.50 |
| hybrid | term_rich | 86 | 31 | 0.779 | 0.279 | 2.18 |
| hybrid | term_sparse | 17 | 10 | 0.000 | 0.000 | - |

## Interim Findings

- Longer descriptions are easier to recover semantically. Semantic Hit@5 rises from 0.000 on short descriptions to 0.767 on long descriptions.
- Description structure matters. Narrative descriptions produce better semantic retrieval than metadata-heavy ones: Hit@5=0.712 vs 0.167.
- Term-rich descriptions carry stronger semantic signals. Semantic Hit@5 goes from 0.000 on term-sparse documents to 0.767 on term-rich documents.
- On weak descriptions, hybrid remains a useful fallback. For term-sparse documents, hybrid Hit@5=0.000 while pure semantic stays at 0.000.

## Interpretation

- RQ4 suggests that semantic quality is not only a model issue; document quality also shapes retrieval success.
- Short, tag-like, or metadata-heavy descriptions weaken the semantic signal.
- Longer and more content-rich descriptions make the main English semantic pipeline more reliable.
- This supports a practical thesis recommendation: improve normalization, enrich weak descriptions, and flag low-information records before indexing.
