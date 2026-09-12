# Project guidance

## Purpose and sources

- This is a staged educational project: understanding the ML code is as important as completing it.
- Treat `tasks/` as the original assignment, `SPEC.md` as accepted project decisions, and `PLAN.md` as the current execution order.
- If these sources conflict or a decision is missing, explain the conflict and ask the user before implementing it.

## Working method

- Before work, inspect `git status`, the requested PLAN item, and only the relevant task/spec sections.
- Work on one explicit PLAN item at a time. Do not implement later days or opportunistic features.
- Before ML changes, briefly explain the concept, data flow, tensor shapes, and important trade-offs.
- After ML changes, explain `train`/`eval` mode, gradient boundaries, leakage risks, and metric meaning when relevant.
- Use Russian for explanations and English for code, identifiers, docstrings, and commit messages.

## Implementation and verification

- Prefer simple, typed, reusable functions; avoid abstractions that hide the concept being learned.
- Use the project environment: `.venv/bin/python`.
- Treat static type errors in project code and tests as blocking defects, even
  when runtime tests pass. After changing Python code, run `.venv/bin/pyright`
  and resolve every error before reporting completion.
- Fix the underlying type contract instead of adding broad suppressions.
  Use `cast`, `# type: ignore`, or relaxed Pyright rules only for a documented
  third-party typing boundary that cannot be expressed or narrowed safely.
- Keep the editor on the project `.venv` interpreter so Pylance and the Pyright
  CLI analyze the same dependencies. If editor diagnostics disagree with the
  CLI, investigate the interpreter/configuration mismatch and report it.
- Add focused pytest coverage for reusable project logic and meaningful edge cases.
- Test project behavior, not Hugging Face or PyTorch internals. Prefer shapes, invariants, ordering, and numeric tolerances over exact tokens or floating-point embeddings.
- Keep default tests fast and offline. Mark checks that load real pretrained models as integration tests and do not make them depend on a fresh network download.
- Run focused tests first. Available general checks are `.venv/bin/ruff check .`,
  `.venv/bin/pyright`, and `.venv/bin/python -m pytest`.
- Report exactly what was verified and what remains unverified. Warn before long training runs or large downloads.
- Keep datasets, caches, generated plots, serialized models, and experiment results out of Git unless explicitly requested.
- Do not commit, push, or rewrite user changes unless explicitly requested.
