# Transformers learning project

## Goal

Build a seven-day sentiment-analysis project while understanding tokenization, hidden states, attention, baseline classification, fine-tuning, evaluation, and error analysis.

## Current scope

Days 1–4 are specified below. Days 5–7 remain high-level until their
requirements are discussed; later-day code is out of scope for now.

## Accepted decisions

- Language: English.
- Shared tokenizer/model checkpoint: `distilbert-base-uncased`.
- Working format: a small hybrid. Reusable project logic lives in
  `src/transformers_learning/`; notebooks in `notebooks/` contain only
  explanations, small examples, and calls to that reusable logic.
- Day 1 walkthrough notebook: `notebooks/day1_tokenization.ipynb`.

## Dataset decision

- Day 4 uses [`stanfordnlp/sst2`](https://huggingface.co/datasets/stanfordnlp/sst2),
  the English binary sentence-level Stanford Sentiment Treebank task. Its
  labelled source fields are `sentence,label`; project code adapts these to
  `text,label` and ignores the source `idx` field.
- Label mapping: `0 = negative`, `1 = positive`. The published split sizes are
  67,349 training rows (29,780 negative; 37,569 positive), 872 validation
  rows (428 negative; 444 positive), and 1,821 test rows. The official test
  labels are not public, so it is not a supervised-training input.
- The published dataset card declares its license as `unknown`. Download a
  local copy only for this educational exercise, observe the upstream terms,
  and do not redistribute the dataset. It remains excluded from Git.

## Day 1 contract: tokenization

- Load one `AutoTokenizer` and inspect its vocabulary size and maximum length.
- Demonstrate tokens, token IDs, special tokens, and decoding for one text.
- Provide `tokenize_texts(texts, tokenizer, max_length=128)` with explicit
  tokenizer injection, padding, truncation, and PyTorch tensors.
- Provide `explain_tokenization(text, tokenizer)` and explain `input_ids` and `attention_mask`.

## Day 2 contract: hidden states

- Load `AutoModel` using the same model name as the tokenizer and switch it to evaluation mode.
- Perform inference under `torch.no_grad()` and inspect `last_hidden_state` as `[batch, sequence, hidden]`.
- Extract the first-token representation as `[batch, hidden]` and state the limitations of calling it a CLS embedding.
- Provide batched `get_embeddings(...)` returning one NumPy row per input text.
- Compare text representations with cosine similarity and interpret the result cautiously.

## Day 3 contract: attention inspection

- Load the shared `AutoModel` with `output_attentions=True`, keeping evaluation mode and `torch.no_grad()` for inference.
- Inspect the returned attention weights across layers and explain their shape as `[batch, heads, sequence, sequence]`.
- Extract one selected layer and head as a `[sequence, sequence]` matrix, where rows represent query tokens and columns represent key tokens.
- Visualize selected attention matrices as token-labelled heatmaps for a short English text.
- Compare selected early, middle, and late layers and multiple heads, treating the results as observations of model behavior rather than fixed semantic roles.
- Explicitly distinguish attention weights from the input `attention_mask`: the mask hides padding positions, while attention weights describe token-to-token interactions produced by the model.
- Treat attention plots as exploratory diagnostics, not as sufficient evidence that a token caused a sentiment prediction; sentiment classification remains a later-day task.

## Day 4 contract: frozen-embedding classification baseline

- Select and document one sentiment dataset with the tabular `text,label`
  contract, its source, label mapping, class counts, and license/usage notes
  before extracting embeddings. Keep the dataset itself out of Git unless it is
  explicitly approved for version control.
- Reuse the shared `distilbert-base-uncased` tokenizer and `AutoModel` in
  evaluation mode. Produce one NumPy feature row per input text with
  `get_embeddings`; the Transformer weights remain frozen and gradients are
  disabled during embedding extraction.
- Use the first-token representation as the baseline feature vector, with
  shape `[n_samples, hidden_size]`. It is a contextual representation, not a
  sentiment score or a claim of optimal sentence embedding quality.
- Create exactly one stratified 80/20 train/test split with `random_state=42`.
  Fit `LogisticRegression(max_iter=1000, n_jobs=-1)` only on the training
  embeddings and labels. The test split is held out from model and
  hyperparameter selection.
- Report `classification_report` and macro F1 on the held-out test split.
  Macro F1 is the primary Day 4 comparison metric because it weights every
  class equally; report class support alongside it so the result is
  interpretable.
- Keep reusable dataset validation, split, training, and evaluation logic in
  `src/transformers_learning/`; use a Day 4 notebook only for explanation,
  small inspections, and calls to that logic. Add fast offline tests for
  project-owned behavior; keep a real pretrained-model run as a separate
  integration smoke check.
- Save the final metric and minimal run context (model name, split parameters,
  dataset identifier, and label mapping) to the ignored `baseline_results.txt`.
  Do not persist model weights or use this file as an experiment tracker.

## Project invariants

- Keep examples small before running full datasets or training.
- Make tensor shapes and device movement observable at important boundaries.
- Cover reusable preprocessing and embedding logic with fast offline tests; keep real-model checks separate.
- From Day 4 onward, use one reproducible data split and keep the final test set out of model selection.
- Compare models on the same examples, split, labels, and metrics.
