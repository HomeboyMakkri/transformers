# Execution plan

Work on one item per request. For each item, read its matching `tasks/dayN.md` section and the relevant `SPEC.md` contract. Before detailing a new day, discuss its assignment and update the documents.

## Day 1 — Tokenization

### D1-01 — Decisions and scaffold

- [x] Choose the language/model and notebook/module format; record both in `SPEC.md`.
- **Done when:** the implementation location and one `model_name` are unambiguous.
- **Verify:** imports work in `.venv`; do not load `AutoModel` yet.

### D1-02 — Tokenizer inspection

- [x] Load `AutoTokenizer`; inspect vocabulary size, maximum length, and special tokens.
- **Done when:** the observed values and meanings are explained with one example.
- **Verify:** run a focused tokenizer smoke check; no model inference.

### D1-03 — Single-text walkthrough

- [x] Tokenize one text; inspect tokens and IDs; decode it back.
- **Done when:** the role of special tokens and non-perfect decoding is understood.
- **Verify:** demonstrate the encode/decode path without asserting exact library internals.

### D1-04 — Batch tokenization

- [x] Implement `tokenize_texts`; demonstrate padding, truncation, shapes, and masks.
- **Done when:** the function returns aligned PyTorch tensors for a text batch.
- **Verify:** pytest covers batch size, matching shapes, padding mask, and `max_length`.

### D1-05 — Explanation helper and review

- [x] Implement `explain_tokenization` and review the complete Day 1 data flow.
- **Done when:** the checkpoint in `tasks/day1.md` is satisfied and explainable.
- **Verify:** run focused tests, Ruff, and a manual example; report any network-dependent check separately.

## Day 2 — Hidden states

### D2-01 — Model inference

- [x] Load the matching `AutoModel`; use `eval()` and `torch.no_grad()`; inspect its output type.
- **Done when:** tokenizer and model share `model_name`, and inference performs no training.
- **Verify:** one real-model smoke run reports device and output shape.

### D2-02 — Representation walkthrough

- [x] Inspect `last_hidden_state` and extract the first-token representation.
- **Done when:** `[batch, sequence, hidden]` and `[batch, hidden]` are explained from actual tensors.
- **Verify:** assert dimensional invariants, not exact embedding values.

### D2-03 — Batched embeddings

- [x] Implement `get_embeddings` with batching and NumPy output.
- **Done when:** it returns one row per input text in the original order, including a partial final batch.
- **Verify:** fast pytest coverage checks batching, order, output shape, and empty-input policy.

### D2-04 — Similarity experiment

- [x] Implement cosine similarity and compare several deliberately chosen text pairs.
- **Done when:** results are interpreted as embedding similarity, not guaranteed sentiment quality.
- **Verify:** check type, finiteness, and numeric bounds; do not require a specific ranking from the pretrained model.

### D2-05 — Review

- [x] Review Day 2 concepts and the reusable API before proceeding to attention.
- **Done when:** the checkpoint in `tasks/day2.md` is satisfied and limitations are documented.
- **Verify:** run focused offline tests, Ruff, and one separate real-model smoke check.

## Day 3 — Attention

### D3-01 — Model inference with attention outputs

- [x] Load the shared `AutoModel` with `output_attentions=True` and run one small example in evaluation mode under `torch.no_grad()`.
- **Done when:** the model output exposes attention weights without changing the tokenizer/model checkpoint.
- **Verify:** report the output type and the number of attention layers.

### D3-02 — Attention tensor walkthrough

- [x] Inspect attention as `[batch, heads, sequence, sequence]` and extract one layer and one head as `[sequence, sequence]`.
- **Done when:** batch, layer, head, query, and key dimensions are explained correctly.
- **Verify:** assert dimensional invariants rather than exact attention values.

### D3-03 — Attention visualization

- [x] Implement a small visualization helper that labels both axes with tokens and renders one selected layer/head as a heatmap.
- **Done when:** attention for a short text can be saved and inspected for selected layers and heads.
- **Verify:** keep plots as generated artifacts outside Git and check that the selected indices are valid.

### D3-04 — Layer and head comparison

- [x] Compare attention heatmaps from early, middle, and late layers and from several heads of one layer.
- **Done when:** observations distinguish layer/head variation from universal model rules.
- **Verify:** use small, reproducible examples and describe observations cautiously.

### D3-05 — Attention interpretation review

- [x] Explain the difference between `attention_mask` and attention weights, and document that attention visualization is not by itself a reliable sentiment explanation.
- **Done when:** the Day 3 checkpoint in `tasks/day3.md` is satisfied without introducing classification work.
- **Verify:** run focused offline checks plus a separate real-model visualization smoke check.

## Day 4 — Frozen-embedding classification baseline

### D4-01 — SST-2 selection and dataset contract

- [x] Select `stanfordnlp/sst2`; document its source, binary label mapping,
  labelled split sizes/class counts, and the source's `unknown` license status.
- [x] Adapt a labelled SST-2 table from `sentence,label` to the project-wide
  `text,label` contract and validate text, labels, and the presence of both
  classes without downloading or committing the dataset.
- **Done when:** a validated local table has non-empty `text` values and both
  `0 = negative` and `1 = positive` labels.
- **Verify:** fast offline tests cover the metadata, source-column conversion,
  defensive copying, and invalid/missing labels.

### D4-02 — Tokenization and frozen embedding dataset

- [x] Reuse the shared `load_tokenizer`, `tokenize_texts`, `load_model`, and
  `get_embeddings` utilities from Days 1–2 with `distilbert-base-uncased`.
- [x] Combine a validated `text,label` table with `get_embeddings` to produce
  aligned features `[n_samples, hidden_size]` and labels `[n_samples]`.
- [x] Validate model-output rank, alignment, feature width, and finiteness at
  the project boundary.
- **Done when:** each feature row corresponds to its original text and label;
  `get_embeddings` performs inference in `eval()`/`no_grad()`, so no classifier
  is fitted and no label influences embedding values.
- **Verify:** offline tests mock the embedding extractor and cover row order,
  batch-size forwarding, validation before model calls, and malformed outputs;
  a later integration run shows tokenization and `[n_samples, hidden_size]` on
  a small real SST-2 sample.

### D4-03 — Held-out split

- [x] Split frozen features and labels once with `test_size=0.2`,
  `random_state=42`, and `stratify=labels`.
- [x] Reserve the test partition for final evaluation; no training or model
  selection code can receive it at this stage.
- **Done when:** both classes occur in the 80/20 partitions and the result is
  deterministic for identical input rows.
- **Verify:** offline tests assert split sizes, class preservation,
  reproducibility, no row overlap, and invalid public inputs.

### D4-04 — Logistic Regression training

- [x] **Status: implementation complete; real artifacts require the separate
  approved three-epoch training run.**
- [x] Construct `LogisticRegression(max_iter=1000, n_jobs=-1)`.
- [x] Fit it only on `X_train, y_train`; do not pass `X_test` or `y_test` to
  `fit`, parameter selection, or preprocessing.
- **Done when:** the returned classifier is fitted from the frozen training
  embeddings and can later predict binary labels.
- **Verify:** offline tests assert the constructor parameters and that `fit`
  receives precisely the training partition, never the held-out partition.

### D4-05 — Held-out evaluation and result record

- [x] **Status: implementation complete; the real SST-2 metric remains a separate integration run.**
- [x] Implement prediction on `X_test` with the fitted baseline classifier.
- [x] Implement `classification_report(y_test, y_pred)` and calculate
  `f1_score(y_test, y_pred, average="macro")`; include per-class support in the
  reported output.
- [x] Save only macro F1 and minimal run context (checkpoint, dataset ID,
  label mapping, and split parameters) to ignored `baseline_results.txt`.
- **Done when:** macro F1 is reported strictly for the held-out set and is not
  used to choose settings in this baseline run.
- **Verify:** offline tests cover prediction/evaluation inputs and metric
  bounds; a separate real-model integration run records the actual metric.

### D4-06 — Day 4 notebook walkthrough and review

- [x] **Status: implementation complete; the real-model execution remains a separate integration run.**
- [x] Add a thin Day 4 notebook that calls the reusable functions: inspect
  tokenization for two or three texts, show one small embedding shape, then
  run the already-defined baseline flow.
- [x] Explain that the first-token vector is a contextual feature, not a
  sentiment probability, and that macro F1 weights the two classes equally.
- **Done when:** the notebook contains explanation and calls only; reusable
  logic and tests remain in `src/` and `tests/`.
- **Verify:** run the notebook only after explicitly allowing the model/data
  download; keep generated results and caches out of Git.

## Day 5 — Fine-tuned sequence classifier

### D5-01 — Shared holdout and training/validation partitions

- [x] **Status: complete.**
- [x] Reuse the Day 4 source rows, label mapping, and deterministic stratified
  80/20 outer split so the fine-tuned model and frozen baseline have exactly
  the same final held-out examples.
- [x] Split only the outer training rows again into training and validation
  partitions with `test_size=0.2`, `random_state=42`, and stratification. The
  outer test rows remain unavailable to the Day 5 training loop and metrics.
- **Done when:** every source row belongs to exactly one of train, validation,
  or test; both labels occur in every partition; the test row identities match
  Day 4 and cannot be passed to Day 5 training or validation code.
- **Verify:** fast offline tests check determinism, class preservation, no row
  overlap, full row coverage, and exact agreement with the Day 4 holdout.

### D5-02 — Tokenized PyTorch Dataset

- [x] **Status: complete.**
- [x] Implement a typed `SentimentDataset` over validated text/label rows,
  reusing the Day 1 tokenizer with `max_length=128`, truncation, and
  `padding="max_length"`.
- [x] Return one item with `input_ids: [max_length]`,
  `attention_mask: [max_length]`, and a scalar `torch.long` label; do not leave
  the tokenizer's temporary batch dimension on individual items.
- **Done when:** dataset length matches its aligned inputs and a `DataLoader`
  batch has inputs `[batch, max_length]` and labels `[batch]`.
- **Verify:** offline tests use a fake tokenizer to cover alignment, dtypes,
  shapes, truncation arguments, empty/mismatched inputs, and invalid labels.

### D5-03 — Classification model and DataLoaders

- [x] **Status: complete.**
- [x] Load `AutoModelForSequenceClassification` from
  `distilbert-base-uncased` with `num_labels=2` and create train/validation
  `DataLoader` objects with `batch_size=16`.
- [x] Shuffle only the training loader; select CUDA when available and
  otherwise use CPU, then move the model to that device.
- **Done when:** one batch produces logits `[batch, 2]` from input tensors
  `[batch, max_length]` without using outer-test rows.
- **Verify:** test loader configuration and boundaries offline; keep the real
  checkpoint/device forward pass as a separate integration smoke check.

### D5-04 — One training epoch

- [x] **Status: complete.**
- [x] Implement `train_epoch` with `model.train()`, device transfer,
  `optimizer.zero_grad()`, forward pass with labels, scalar cross-entropy loss,
  `loss.backward()`, and `optimizer.step()` for every batch.
- [x] Use `torch.optim.AdamW(model.parameters(), lr=2e-5)` and report mean loss
  per processed batch.
- **Done when:** gradients update model parameters only during training and the
  returned epoch loss is finite and non-negative.
- **Verify:** focused tests check mode selection, batch/device forwarding,
  optimizer call order, averaging, and the empty-loader policy without
  training a pretrained model.

### D5-05 — Validation evaluation

- [x] **Status: complete.**
- [x] Implement evaluation with `model.eval()` and a no-gradient context;
  convert logits `[batch, 2]` to predictions `[batch]` with `argmax(dim=1)`.
- [x] Report validation accuracy and macro F1 with explicit binary label
  handling; keep predictions aligned with their labels.
- **Done when:** evaluation cannot create gradients or update parameters, and
  both metrics are finite values in `[0, 1]` calculated only from validation
  rows.
- **Verify:** offline tests cover mode/gradient boundaries, concatenation over
  multiple batches, prediction order, metric values, and malformed outputs.

### D5-06 — Fixed three-epoch fine-tuning run

- [x] **Status: complete.**
- [x] Run exactly three epochs, calling `train_epoch` and validation evaluation
  once per epoch; record train loss, validation accuracy, and validation macro
  F1 for every epoch.
- [x] Treat validation metrics as monitoring information for this fixed run;
  do not inspect the outer test set, tune hyperparameters, or add early
  stopping in Day 5.
- **Done when:** the run produces three ordered metric records and leaves the
  final model in a clearly documented state after epoch 3.
- **Verify:** unit-test orchestration with small fakes; run full fine-tuning
  only after warning about runtime, memory use, and model/data access.

### D5-07 — Save model and validation results

- [x] **Status: complete.**
- [x] Save the final model and tokenizer with `save_pretrained` under ignored
  `fine_tuned_model/` and write the epoch-3 validation metrics plus minimal run
  context to ignored `fine_tuned_results.txt`.
- [x] Record checkpoint, dataset ID, label mapping, outer and inner split
  parameters, epochs, batch size, maximum length, learning rate, and device;
  do not label validation metrics as final held-out test results.
- **Done when:** the local artifact can be reloaded with its tokenizer and the
  result file is sufficient to identify the fixed Day 5 run.
- **Verify:** use temporary directories for save/load and result-format tests;
  keep generated weights and result files out of Git.

### D5-08 — Day 5 notebook walkthrough and review

- [x] **Status: implementation complete; real notebook execution remains a
  separate approved integration run.**
- [x] Add a thin notebook that calls reusable `src/` functions, displays one
  sample and one batch shape, then launches the explicitly approved training
  run and records its per-epoch validation history.
- [x] Explain fine-tuning versus frozen embeddings, `train()` versus `eval()`,
  enabled versus disabled gradients, and why validation and outer test data
  have different roles.
- **Done when:** the `tasks/day5.md` checkpoint is satisfied and the complete
  path from text to saved classifier can be explained without reusable logic
  living in notebook cells.
- **Verify:** run offline tests, Ruff, Pyright, and `git diff --check`; execute
  the real notebook separately because it may require downloads and a long
  CPU/GPU training run.

## Day 6 — Held-out inference and model comparison

### D6-01 — Shared comparison dataset and artifact boundaries

- [x] Recreate the deterministic outer 80/20 SST-2 row split from the same
  validated source-row order used in Days 4–5; select outer-train and
  outer-test rows by the preserved indices.
- [x] Define the Day 6 artifact inputs: the saved epoch-3
  `fine_tuned_model/` directory and a freshly recreated Day 4 baseline. Do not
  load or add `vectorizer.pkl` or a persisted baseline classifier.
- **Done when:** the comparison test texts and labels match the Day 4/5 outer
  holdout exactly, and no outer-test row is available to baseline fitting or
  model selection.
- **Verify:** offline tests assert exact row identities and order, complete
  train/test separation, and a clear failure when required fine-tuned
  artifacts are absent or incompatible.

### D6-02 — Recreate and load the two model paths

- [x] **Status: implementation complete; real inference paths require the
  ignored Day 5 `fine_tuned_model/` artifact and a separate approved
  data/model integration run.**
- [x] Recreate frozen embeddings for the outer-training rows and fit the fixed
  Day 4 `LogisticRegression(max_iter=1000, n_jobs=-1)` only on those features;
  use the shared base tokenizer and encoder in evaluation/no-gradient mode.
- [x] Reload the fine-tuned sequence classifier and tokenizer from
  `fine_tuned_model/`, select its execution device, move the model there, and
  set evaluation mode without performing further optimization.
- **Done when:** both final inference paths are ready before the outer-test
  metrics are inspected, with no fitted text vectorizer and no Day 6 training
  of the fine-tuned classifier.
- **Verify:** focused tests use fakes to prove the baseline fit receives only
  outer-training features and the fine-tuned loader uses the local artifact,
  binary head, selected device, and evaluation mode.

### D6-03 — Fine-tuned prediction API

- [x] Implement typed `predict_fine_tuned` logic for one string or an ordered
  text sequence, using batched tokenization, truncation, `max_length=128`,
  device transfer, `model.eval()`, and `torch.no_grad()`.
- [x] Convert logits `[batch, 2]` into probabilities in label order `[0, 1]`
  with softmax and predictions `[batch]` with `argmax(dim=1)`; preserve the
  original text and order in the returned records.
- **Done when:** every input produces exactly one binary prediction and a
  finite two-class probability vector summing to one.
- **Verify:** offline tests cover scalar/list input, multiple and partial
  batches, order, tensor shapes, device/mode/gradient boundaries, empty input,
  and malformed logits.

### D6-04 — Frozen-baseline prediction API

- [x] Implement typed `predict_baseline` logic for the same raw-text interface,
  deriving frozen first-token embeddings through the shared tokenizer/base
  encoder and applying the recreated logistic-regression classifier.
- [x] Return predictions and `predict_proba` values in explicit label order
  `[0, 1]`, preserving original text and input order without a separate
  cleaning function or fitted vectorizer.
- **Done when:** both predictors expose aligned result records with compatible
  label and probability semantics despite their different internal data
  flows.
- **Verify:** offline tests mock embedding extraction and classification to
  cover batching and order, feature/prediction alignment, encoder inference
  boundaries, finite probabilities, and invalid outputs.

### D6-05 — Five-example inference comparison

- [x] **Status: implementation complete; real example output requires the
  ignored Day 5 artifact and a separate approved integration run.**
- [x] Run both prediction APIs on the five English examples from
  `tasks/day6.md` and display labels, probabilities, and whether predictions
  agree.
- [x] Explain that the neutral-sounding example is still mapped into the
  binary SST-2 label space and that agreement or confidence on five selected
  sentences is not an evaluation metric.
- **Done when:** the small example makes both text-to-prediction paths and
  their shared label mapping observable before full held-out inference.
- **Verify:** test the comparison-table assembly offline; reserve real model
  execution for the explicitly approved integration run.

### D6-06 — Paired held-out metrics

- [x] **Status: implementation complete; real held-out metrics require a
  separate approved data/model integration run.**
- [x] Predict every shared outer-test row with both models in identical order,
  then validate one-to-one alignment with the same labels before evaluation.
- [x] Compute `classification_report`, accuracy, and macro F1 for each model
  with label order `[0, 1]`; report signed absolute fine-tuned-minus-baseline
  deltas and guard any optional relative F1 calculation against division by
  zero.
- **Done when:** both approaches are compared on exactly the same examples and
  macro F1 is clearly identified as the primary held-out metric.
- **Verify:** offline tests use known predictions to check metric values,
  class support, direction of deltas, alignment rejection, and the zero-F1
  edge case.

### D6-07 — Confusion matrices and comparison record

- [x] **Status: implementation complete; real PNG and result artifacts require
  a separate approved data/model integration run.**
- [x] Build raw-count confusion matrices for both models with identical true
  and predicted axes, explicit `negative`/`positive` labels, and class order
  `[0, 1]`; save them as ignored PNG artifacts.
- [x] Write ignored `comparison_results.txt` with both models' macro F1 and
  accuracy, signed deltas, and the required dataset/model/split/test-count and
  artifact context.
- **Done when:** the two plots are directly comparable and the result record
  identifies the evaluated run without persisting models, embeddings, or
  datasets.
- **Verify:** use temporary paths to test matrix orientation, fixed label
  order, file creation, result content, and refusal of inconsistent inputs.

### D6-08 — Day 6 notebook walkthrough and review

- [x] **Status: implementation complete; real notebook execution requires the
  ignored Day 5 artifact and a separate approved data/model integration run.**
- [x] Add a thin Day 6 notebook that calls reusable `src/` functions, first
  compares the five small examples, then performs the paired outer-test
  evaluation and displays both confusion matrices and metric deltas.
- [x] Explain both inference data flows, `eval()`/no-gradient behavior, the
  shared holdout and leakage boundary, macro F1 versus accuracy, probability
  interpretation, and why detailed error analysis remains Day 7 work.
- **Done when:** the checkpoint in `tasks/day6.md` is satisfied for both
  models and the comparison can be explained without reusable logic in
  notebook cells.
- **Verify:** run focused tests, Ruff, Pyright, and `git diff --check`; reload
  real artifacts and execute the full holdout comparison only after warning
  about model/data access, runtime, memory, and the required Day 5 artifact.

## Later backlog — detail only when reached

- [ ] **Day 7:** analyze errors and build a small demo.
