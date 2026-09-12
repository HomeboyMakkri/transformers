# Transformers learning project

## Goal

Build a seven-day sentiment-analysis project while understanding tokenization, hidden states, attention, baseline classification, fine-tuning, evaluation, and error analysis.

## Current scope

Days 1 and 2 are specified below. Days 3–7 remain high-level until their requirements are discussed; later-day code is out of scope for now.

## Accepted decisions

- Language: English.
- Shared tokenizer/model checkpoint: `distilbert-base-uncased`.
- Working format: a small hybrid. Reusable project logic lives in
  `src/transformers_learning/`; notebooks in `notebooks/` contain only
  explanations, small examples, and calls to that reusable logic.
- Day 1 walkthrough notebook: `notebooks/day1_tokenization.ipynb`.

## Open decisions

- Sentiment dataset and label mapping; the eventual tabular contract is `text,label`.

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

## Project invariants

- Keep examples small before running full datasets or training.
- Make tensor shapes and device movement observable at important boundaries.
- Cover reusable preprocessing and embedding logic with fast offline tests; keep real-model checks separate.
- From Day 4 onward, use one reproducible data split and keep the final test set out of model selection.
- Compare models on the same examples, split, labels, and metrics.
