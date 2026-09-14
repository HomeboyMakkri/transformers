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

- [ ] Load the shared `AutoModel` with `output_attentions=True` and run one small example in evaluation mode under `torch.no_grad()`.
- **Done when:** the model output exposes attention weights without changing the tokenizer/model checkpoint.
- **Verify:** report the output type and the number of attention layers.

### D3-02 — Attention tensor walkthrough

- [ ] Inspect attention as `[batch, heads, sequence, sequence]` and extract one layer and one head as `[sequence, sequence]`.
- **Done when:** batch, layer, head, query, and key dimensions are explained correctly.
- **Verify:** assert dimensional invariants rather than exact attention values.

### D3-03 — Attention visualization

- [ ] Implement a small visualization helper that labels both axes with tokens and renders one selected layer/head as a heatmap.
- **Done when:** attention for a short text can be saved and inspected for selected layers and heads.
- **Verify:** keep plots as generated artifacts outside Git and check that the selected indices are valid.

### D3-04 — Layer and head comparison

- [ ] Compare attention heatmaps from early, middle, and late layers and from several heads of one layer.
- **Done when:** observations distinguish layer/head variation from universal model rules.
- **Verify:** use small, reproducible examples and describe observations cautiously.

### D3-05 — Attention interpretation review

- [ ] Explain the difference between `attention_mask` and attention weights, and document that attention visualization is not by itself a reliable sentiment explanation.
- **Done when:** the Day 3 checkpoint in `tasks/day3.md` is satisfied without introducing classification work.
- **Verify:** run focused offline checks plus a separate real-model visualization smoke check.

## Later backlog — detail only when reached

- [ ] **Day 4:** build a frozen-embedding classification baseline.
- [ ] **Day 5:** fine-tune a sequence classifier.
- [ ] **Day 6:** compare both approaches on the same held-out data.
- [ ] **Day 7:** analyze errors and build a small demo.
