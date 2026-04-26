# Retriever Training Set Report

Purpose:
Create hard-negative triplets and positive pairs for domain-adaptive retriever fine-tuning.

## Summary

- Profile: `minilm`
- Queries: `69`
- Pairs: `242`
- Triplets: `726`
- Unique positives: `67`
- Unique negatives: `120`
- Benchmarks: `{'general_keyword': 12, 'general_sentence': 12, 'earthquake_sentence': 10, 'cross_source_similarity': 13, 'tr': 12, 'semantic_intent': 10}`
- Languages: `{'en': 63, 'tr': 6}`
- Pair splits: `{'train': 207, 'dev': 35}`
- Triplet splits: `{'train': 621, 'dev': 105}`
- Negative strategies: `{'retrieval_confusion': 726}`
- Method usage: `{'hybrid': 712, 'semantic': 692, 'bm25': 528}`

## Output Files

- Corpus: `C:\Users\user\Desktop\Tez\data\training\retriever_corpus.jsonl`
- Pairs: `C:\Users\user\Desktop\Tez\data\training\retriever_pairs.jsonl`
- Triplets: `C:\Users\user\Desktop\Tez\data\training\retriever_triplets.jsonl`
- Summary: `C:\Users\user\Desktop\Tez\data\training\retriever_training_summary.json`
