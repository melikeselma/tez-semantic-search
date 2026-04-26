# Retriever Fine-Tuning Report

Purpose:
Fine-tune the English-centered MiniLM retriever with domain pairs and hard negatives.

## Configuration

- Base profile: `minilm`
- Base model: `sentence-transformers/all-MiniLM-L6-v2`
- Pair epochs: `2`
- Triplet epochs: `1`
- Batch size: `16`
- Learning rate: `2e-05`
- Triplet margin: `0.25`

## Data

- Train pairs: `207`
- Dev pairs: `35`
- Train triplets: `621`
- Dev triplets: `105`
- Corpus docs: `1120`

## Dev Retrieval

- Baseline nDCG@5: `0.612`
- Fine-tuned nDCG@5: `0.692`
- Baseline MRR: `0.783`
- Fine-tuned MRR: `0.733`
- Baseline Hit@5: `0.900`
- Fine-tuned Hit@5: `0.900`

## Outputs

- Final model: `C:\Users\user\Desktop\Tez\models\retriever\minilm-domain-ft\final`
- Summary: `C:\Users\user\Desktop\Tez\models\retriever\minilm-domain-ft\training_summary.json`
