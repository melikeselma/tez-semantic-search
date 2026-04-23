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
| description_style | mixed_structured | 597 | 0.533 |
| description_style | narrative | 317 | 0.283 |
| term_bucket | term_moderate | 361 | 0.322 |
| term_bucket | term_rich | 372 | 0.332 |
| term_bucket | term_sparse | 387 | 0.346 |

## Retrieval by Description Length

| Method | Bucket | Relevant Pairs | Unique Docs | Hit@5 | Top1 | Mean Rank When Hit |
|---|---|---:|---:|---:|---:|---:|
| semantic | long | 90 | 33 | 0.644 | 0.244 | 2.12 |
| semantic | medium | 38 | 18 | 0.579 | 0.237 | 2.09 |
| semantic | short | 22 | 11 | 0.545 | 0.227 | 2.25 |
| hybrid | long | 90 | 33 | 0.722 | 0.256 | 2.31 |
| hybrid | medium | 38 | 18 | 0.658 | 0.289 | 2.12 |
| hybrid | short | 22 | 11 | 0.545 | 0.227 | 2.25 |

## Retrieval by Description Style

| Method | Bucket | Relevant Pairs | Unique Docs | Hit@5 | Top1 | Mean Rank When Hit |
|---|---|---:|---:|---:|---:|---:|
| semantic | metadata_heavy | 30 | 13 | 0.533 | 0.200 | 2.25 |
| semantic | mixed_structured | 54 | 23 | 0.611 | 0.278 | 1.94 |
| semantic | narrative | 66 | 26 | 0.652 | 0.227 | 2.23 |
| hybrid | metadata_heavy | 30 | 13 | 0.700 | 0.233 | 2.62 |
| hybrid | mixed_structured | 54 | 23 | 0.667 | 0.333 | 2.00 |
| hybrid | narrative | 66 | 26 | 0.682 | 0.212 | 2.29 |

## Retrieval by Term Richness

| Method | Bucket | Relevant Pairs | Unique Docs | Hit@5 | Top1 | Mean Rank When Hit |
|---|---|---:|---:|---:|---:|---:|
| semantic | term_moderate | 46 | 21 | 0.609 | 0.217 | 2.04 |
| semantic | term_rich | 88 | 32 | 0.636 | 0.250 | 2.12 |
| semantic | term_sparse | 16 | 9 | 0.500 | 0.250 | 2.50 |
| hybrid | term_moderate | 46 | 21 | 0.652 | 0.261 | 2.20 |
| hybrid | term_rich | 88 | 32 | 0.716 | 0.261 | 2.32 |
| hybrid | term_sparse | 16 | 9 | 0.562 | 0.250 | 2.00 |

## Interim Findings

- Longer descriptions are easier to recover semantically. Semantic Hit@5 rises from 0.545 on short descriptions to 0.644 on long descriptions.
- Description structure matters. Narrative descriptions produce better semantic retrieval than metadata-heavy ones: Hit@5=0.652 vs 0.533.
- Term-rich descriptions carry stronger semantic signals. Semantic Hit@5 goes from 0.500 on term-sparse documents to 0.636 on term-rich documents.
- On weak descriptions, hybrid remains a useful fallback. For term-sparse documents, hybrid Hit@5=0.562 while pure semantic stays at 0.500.

## Interpretation

- RQ4 suggests that semantic quality is not only a model issue; document quality also shapes retrieval success.
- Short, tag-like, or metadata-heavy descriptions weaken the semantic signal.
- Longer and more content-rich descriptions make the main English semantic pipeline more reliable.
- This supports a practical thesis recommendation: improve normalization, enrich weak descriptions, and flag low-information records before indexing.
