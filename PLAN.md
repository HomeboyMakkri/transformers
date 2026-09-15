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

- [x] **Status: complete.**
- [x] Construct `LogisticRegression(max_iter=1000, n_jobs=-1)`.
- [x] Fit it only on `X_train, y_train`; do not pass `X_test` or `y_test` to
  `fit`, parameter selection, or preprocessing.
- **Done when:** the returned classifier is fitted from the frozen training
  embeddings and can later predict binary labels.
- **Verify:** offline tests assert the constructor parameters and that `fit`
  receives precisely the training partition, never the held-out partition.

### D4-05 — Held-out evaluation and result record

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

- [ ] Add a thin Day 4 notebook that calls the reusable functions: inspect
  tokenization for two or three texts, show one small embedding shape, then
  run the already-defined baseline flow.
- [ ] Explain that the first-token vector is a contextual feature, not a
  sentiment probability, and that macro F1 weights the two classes equally.
- **Done when:** the notebook contains explanation and calls only; reusable
  logic and tests remain in `src/` and `tests/`.
- **Verify:** run the notebook only after explicitly allowing the model/data
  download; keep generated results and caches out of Git.

## Later backlog — detail only when reached

- [ ] **Day 5:** fine-tune a sequence classifier.
- [ ] **Day 6:** compare both approaches on the same held-out data.
- [ ] **Day 7:** analyze errors and build a small demo.
