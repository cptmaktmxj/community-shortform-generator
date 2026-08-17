# Local KR-SBERT Clickbait Title Ranker Design

## Goal

Add an experimental local classifier that ranks five GPT-generated Korean title candidates by controlled clickbait strength while preserving the existing Stage 3 safety, factual-evidence, and title-length gates. Validate the research model quickly on CPU before deciding whether any classifier artifact should become a bundled default.

This work is independent from the CLI progress-display project and the Docker deployment project. Those projects are explicitly out of scope here.

## Research and License Boundary

The implementation is clean-room code written for this repository.

- `jjanoong2/clickbait_embeddings` is marked MIT and contains paired title/content KR-SBERT embeddings and labels. It may be used for the local research experiment.
- `power-TY/ClickBaitNews` has no declared license. Its code, trained weights, and repository assets must not be copied or modified into this project. Only high-level public facts about its title/body classification approach may inform the design.
- `sejongresearch/ClickbaitClassifier` has no declared license and reports low experimental accuracy. No code or weights may be reused.
- `reczoo/FuxiCTR` is Apache-2.0, but it targets advertising CTR prediction and is not a dependency of this implementation. Its modular training and evaluation concepts may be referenced without copying code.
- `snunlp/KR-SBERT-V40K-klueNLI-augSTS` has no explicit license in its model card and is approximately 468 MB. It is allowed only for this user-approved local research run. It must not be redistributed, embedded in a release image, or presented as a production-safe dependency.

Source links:

- https://huggingface.co/datasets/jjanoong2/clickbait_embeddings
- https://huggingface.co/snunlp/KR-SBERT-V40K-klueNLI-augSTS
- https://github.com/power-TY/ClickBaitNews
- https://github.com/sejongresearch/ClickbaitClassifier
- https://github.com/reczoo/FuxiCTR

## Scope

### Included

- Download and validate the embedding dataset through an explicit research command.
- Train a deterministic cosine-similarity baseline and a small MLP classifier.
- Evaluate Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and confusion matrix.
- Generate five distinct GPT title styles after the narration passes duration validation.
- Apply the current hard safety and exact-script-evidence checks before model ranking.
- Embed the final script once and the five candidate titles as one batch.
- Rank candidates by controlled clickbait fit and semantic similarity.
- Persist the top three candidates, selected title, per-candidate audit scores, and model manifest identity.
- Preserve resumable Stage 3 behavior when ranking or model loading fails.
- Record commands, metrics, failures, fixes, and limitations in one research process document.

### Excluded

- Copying code or weights from unlicensed repositories.
- Redistributing KR-SBERT or the multi-gigabyte embedding dataset.
- Making the experimental ranker the default before in-domain evaluation.
- Replacing Stage 2 `safety_ok` or Stage 3 factual-evidence validation.
- CLI Rich progress rendering.
- Docker packaging.
- Actual TTS, video rendering, or publishing.

## Architecture

The title phase becomes:

```text
validated final narration
  -> GPT Structured Output pool of five title candidates
  -> deterministic title safety and evidence gates
  -> KR-SBERT batch embedding
  -> local MLP clickbait probability
  -> semantic-similarity gate
  -> controlled-clickbait ranking
  -> top three titles and one selected title
```

The new package boundaries are:

- `community_shorts.clickbait.contracts`: strict training manifest, metrics, scored-candidate, and ranking-result models.
- `community_shorts.clickbait.embedder`: embedder protocol and lazy KR-SBERT adapter.
- `community_shorts.clickbait.model`: cosine baseline, MLP definition, checkpoint validation, and batch inference.
- `community_shorts.clickbait.dataset`: Hugging Face download, NPZ schema validation, paired sampling, and batch access.
- `community_shorts.clickbait.training`: deterministic train/validation loop and best-checkpoint export.
- `community_shorts.clickbait.evaluation`: metrics, threshold calibration, and one-time test evaluation.
- `community_shorts.clickbait.ranker`: hard gates, controlled score, ranking, and audit output.

All domain code depends on protocols rather than importing `sentence-transformers` or PyTorch directly. Unit and pipeline tests use fake embedders and deterministic classifier outputs.

## Candidate Contracts

GPT produces exactly five candidates using five distinct styles:

1. direct impact;
2. question;
3. conventional-wisdom reversal;
4. curiosity gap;
5. strong factual statement.

Each candidate contains a Korean title and an exact supporting substring from the final script. Titles remain 18 to 34 characters, contain no URL, and contain no direct investment instruction. Strong words such as “충격”, “무조건”, and “드디어 밝혀졌다” are not rejected by keyword alone.

The public artifact keeps the top three title strings and one selected title. It also stores an optional experimental audit block containing each candidate's clickbait probability, semantic similarity, controlled score, final rank score, acceptance decision, and rejection reasons. Old artifacts without this optional block remain readable.

## Model and Ranking

### Baseline

The baseline chooses a validation-calibrated threshold over cosine similarity between title and content embeddings. It is trained and evaluated first so the MLP must demonstrate useful improvement rather than only exceed random accuracy.

### MLP

For each pair, compute the following without materializing the complete combined feature matrix:

- title embedding: 768 values;
- content embedding: 768 values;
- absolute difference: 768 values;
- element-wise product: 768 values;
- cosine similarity: 1 value.

The 3,073-value vector passes through `3073 -> 256 -> 64 -> 1` with ReLU and dropout. The output is a calibrated probability that the title is clickbait relative to the content.

### Controlled Score

The classifier does not replace policy enforcement. Candidates first pass all hard safety and evidence gates. A semantic cosine similarity below `0.55` is rejected.

The desirable clickbait band is `0.65` through `0.85`. The controlled clickbait fit is:

- `1.0` inside the target band;
- linearly scaled from `0.0` to `1.0` below `0.65`;
- linearly scaled from `1.0` to `0.0` above `0.85`.

The final rank score is:

```text
0.70 * controlled_clickbait_fit + 0.30 * semantic_similarity
```

This deliberately avoids maximizing raw clickbait probability, which could favor misleading title/body mismatch.

## Training and Evaluation Flow

The explicit research command downloads only train and validation first:

- train NPZ: approximately 1.7 GB;
- validation NPZ: approximately 424 MB.

The Dataset Viewer cannot stream this repository, so local NPZ schema validation is mandatory. Required arrays are `title_embeddings`, `content_embeddings`, `labels`, and `article_ids`. Both embedding arrays must be `float32` with width 768. Labels must contain only zero and one.

Sampling is grouped by `article_id`. With seed fixed, select 50,000 article IDs and keep both paired rows, producing 100,000 balanced records. The MLP computes features per batch. It uses AdamW, at most five epochs, and early stopping with patience two on validation loss. Training, validation, and test article IDs must remain disjoint.

The first validation gate requires both Accuracy and F1 to be at least `0.80`. The MLP result is compared with the calibrated cosine baseline. Only after the validation gate passes may the test NPZ be downloaded and evaluated once.

The manifest records:

- dataset repository and immutable revision;
- NPZ file hashes;
- encoder identifier and revision;
- sample seed and article IDs hash;
- architecture and thresholds;
- dependency versions;
- training duration and peak observed memory when available;
- baseline, validation, and test metrics;
- research-only license warning.

## Runtime and Resource Limits

The target machine is an Intel Core i5 with 16 GB RAM and no required GPU.

- Never load train and validation arrays at the same time.
- Never concatenate all 3,073-value training features in memory.
- Use batch feature construction and a configurable batch size.
- Default CPU threads conservatively and allow an environment override.
- Cache the final-script embedding once per item and embed five titles in one batch.
- Do not add the dataset, encoder, Hugging Face cache, checkpoints, or metrics artifacts to Git.
- Keep research artifacts under `artifacts/clickbait/`, which must be ignored.

The MLP checkpoint is expected to be only a few megabytes. It remains a local research artifact until the in-domain gate is passed.

## Configuration and Commands

Clickbait ranking is disabled by default. Configuration includes:

- `enabled`;
- encoder model and optional local revision/path;
- checkpoint and manifest paths;
- similarity threshold;
- target clickbait band;
- score weights;
- batch size;
- offline/local-files-only behavior.

Research commands are explicit and do not run during normal ingestion or curation:

```text
community-shorts clickbait train
community-shorts clickbait evaluate
community-shorts clickbait inspect
```

The Stage 3 generation command gains an explicit experimental title-ranker option. Adding Rich progress display remains a separate later project; only the minimum command composition needed to run and inspect the title ranker is included here.

## Failure Behavior

- Missing optional dependencies produce a concise install-extra error.
- Missing or invalid dataset arrays stop training before allocating model tensors.
- Checkpoint/manifest mismatch stops inference.
- Encoder or classifier unavailability while the ranker is explicitly enabled is a visible `title_failed` result; it never silently falls back to an unaudited title.
- Fewer than three candidates surviving hard gates produces `title_failed` and retains the saved analysis and narration for retry.
- Model scores never override `safety_ok`, factual support, title length, URL, or investment-instruction rules.
- Existing generation remains unchanged when the experimental ranker is disabled.

## Testing

Unit tests cover:

- paired and deterministic article sampling;
- NPZ schema and leakage rejection;
- batch feature arithmetic and exact shape;
- baseline threshold calibration;
- MLP checkpoint/manifest validation;
- controlled-score boundaries;
- hard safety and evidence gates before model calls;
- stable tie-breaking and top-three selection;
- missing-dependency and missing-model diagnostics;
- old generated artifact compatibility.

Integration tests use fake embeddings and classifier scores to verify:

- analysis and duration validation still occur before title generation;
- five candidates are requested;
- only safe/evidence-supported candidates are embedded;
- ranking audit is persisted;
- a title failure reuses the saved analysis and narration on retry;
- disabled ranking preserves existing behavior.

The real research acceptance run records the cosine baseline, MLP validation result, and resource use. Official validation success does not authorize default activation because the runtime content is a short narration rather than a full news article. Before default activation, manually label at least 100 real `candidate title - generated narration` pairs and require F1 of at least `0.80` on that in-domain set.

## Process Record

Implementation and research results are recorded in `docs/research/clickbait-title-ranking.md`. It contains only the implementation process, commands, measurements, failures, fixes, and limitations. It must not contain dataset rows, copyrighted source code, credentials, or raw news bodies.
