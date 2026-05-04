# Hard-Negative Retriever Training Set

Purpose:
Create a small sentence-transformers training set from live semantic failure cases.

Why this matters:
These triples keep the project semantic-search focused while teaching the retriever to push down semantically nearby but wrong datasets.

## Summary

- Profile: `minilm`
- Query specs: `C:\Users\user\Desktop\Tez\data\training\hard_negative_query_specs.json`
- Queries: `7`
- Triples: `39`
- Unique positives: `13`
- Unique negatives: `20`
- Positive selection quality: `{'strong': 27, 'weak': 12}`
- Negative reasons: `{'negative_clue_match': 23, 'missing_query_aspects': 35, 'partial_semantic_overlap': 35}`

## Queries

- `hnq001`: I want to find data for predicting house prices.
- `hnq002`: I want to study how weather affects crop production.
- `hnq003`: I want to research diabetes prediction using patient records.
- `hnq004`: I want to study how cities change using images taken from above.
- `hnq005`: I want to find data for financial fraud detection.
- `hnq006`: I want to find data about people's emotions from short online messages.
- `hnq007`: I want to detect harmful language in user comments.