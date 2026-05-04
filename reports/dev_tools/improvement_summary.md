# Semantic search quality fix - iteration summary

Probe set: `dev_tools/regression_probe.py` (29 probes mixing in-corpus, OOD,
English, Turkish, edge cases). Evaluator: `dev_tools/probe_evaluator.py`.

## Aggregate metrics (no cross-encoder, top-K=5)

| profile | method | P@1 base | P@1 final | Δ | P@3 base | P@3 final | Δ |
|---|---|---|---|---|---|---|---|
| minilm | semantic | 0.39 | 0.47 | +0.08 | 0.50 | 0.74 | +0.24 |
| minilm | hybrid | 0.44 | 0.58 | +0.14 | 0.56 | 0.68 | +0.12 |
| e5_base | semantic | 0.33 | 0.47 | +0.14 | 0.50 | 0.68 | +0.18 |
| e5_base | hybrid | 0.33 | 0.53 | +0.20 | 0.44 | 0.58 | +0.14 |
| minilm_ft | semantic | 0.33 | 0.53 | +0.20 | 0.61 | 0.68 | +0.07 |
| minilm_ft | hybrid | 0.28 | 0.53 | +0.25 | 0.61 | 0.68 | +0.07 |

**Average gain: P@1 +0.17 (~+50% relative), P@3 +0.14 (~+27% relative).**
OOD detection holds at 0.90-1.00 accuracy across all combos.

## Hardcoded query expansion ablation

Disabling `PHRASE_RULES`/`COMBO_RULES`/`TOKEN_RULES` and re-running the same
probes drops P@1 by an average of 0.13 across all profile/method combos. The
curated vocabulary is doing meaningful work; removing it for the sake of
"learned semantics" would regress the system. The honest framing for the
thesis writeup is: *the system uses a curated query-expansion vocabulary
plus learned dense retrieval, not pure end-to-end semantic learning.*

## Concrete fixes that produced these gains

| File | Change |
|---|---|
| `query_understanding.py` | Stop emitting fake stems like "specy" from "species"; treat irregular plurals as a closed set. |
| `query_understanding.py` | `derive_intent_body` strips dangling prepositions/articles ("research on", "a of") so they don't enter the encoded query. |
| `query_understanding.py` | `derive_intent_body` keeps the topic prefix when the "dataset for X" pattern matches (mushroom dataset for classification → "mushroom classification" instead of just "classification"). |
| `bm25.py` | Filter STOPWORDS and platform-noise tokens ("research", "global", "intermediate") from the BM25 query counter; previously those promoted Reddit Australia/PyTorch/Public Opinion etc. for any query that used the word "research". |
| `normalize_merge.py` | Always include the dataset title in `semantic_text`. The previous code dropped the title for any low-info dataset, so "mushrooms" / "Mushroom" had zero topical token in the embedding and could never be retrieved by a "mushroom" query. |
| `reranker.py` | Lexical-anchor penalty: identify the high-IDF content tokens of the query, penalise candidates whose text contains none of them; reward candidates that cover them. Pokemon-images for "spider species", Reddit Australia for "spider", chess-cheating for "credit card fraud" all drop out. |
| `reranker.py` | Confidence label (`strong`/`moderate`/`weak`) computed from stage-1 cosine ceiling and cross-encoder raw max; surfaced per result and via UI banner. |
| `reranker.py` | Cross-encoder rerank weight bumped to 0.72 for OOD queries (where heuristic features collapse to zero), but capped at 0.20 if the model's own max raw score is below 0.05 (it's abstaining). Stops `epstein-files` from being floated to rank-1. |
| `reranker.py` | When every top-K candidate is missing the primary anchor, trim the result list to 3 instead of returning a long list with implicit confidence. |
| `web_app.py` | Cross-encoder rerank checkbox defaults ON; payload exposes `weak_match`, `query_confidence`, `query_is_ood`, `lexical_anchor_signal`. |
| `web_app.py` | Result chips: `Weak match`, `No domain hint`. Confidence banner above the result list explains what "weak" means in plain Turkish.

## Concrete still-open issues

1. **Pokemon-images at rank 3 for "spider species"**: corpus has zero spider
   datasets, and the encoder ties pokemon's "various Pokémon species" to
   spider species. All such results carry the WEAK badge so the user is
   warned, but the heuristic cannot push the entry out without breaking
   legit "species" queries.
2. **Hardcoded curated vocabulary covers ~15 domains**. Outside that list
   (spider, fingerprint, fraud, crop) the system relies on raw cosine and
   the lexical-anchor honesty layer. This matches the thesis claim only if
   we describe the vocabulary as a *seed* layer, not "learned semantic".
3. **Multi-aspect queries** (satellite + weather, code LLM tuning) still
   pick a single dominant aspect; multi-aspect scoring fires only for the
   curated 15-domain set.

## Files added for thesis evaluation work

- `dev_tools/regression_probe.py` - reproducible probe runner
- `dev_tools/probe_evaluator.py` - precision@K + OOD-correctness metrics
- `reports/dev_tools/baseline.json` - probe set before any fix
- `reports/dev_tools/iter3.json` - probe set after current changes
- `reports/dev_tools/iter4_norules.json` - ablation: hardcoded rules off
