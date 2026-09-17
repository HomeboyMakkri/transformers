import ast
import json
from pathlib import Path
from typing import Any, cast

NOTEBOOK_PATH = Path("notebooks/day5_fine_tuning.ipynb")


def load_notebook_cells() -> list[dict[str, Any]]:
    notebook = cast(
        dict[str, Any],
        json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8")),
    )
    assert notebook["nbformat"] == 4
    return cast(list[dict[str, Any]], notebook["cells"])


def test_day5_notebook_is_thin_and_all_code_cells_compile() -> None:
    code_sources: list[str] = []

    for cell in load_notebook_cells():
        if cell["cell_type"] != "code":
            continue
        source = "".join(cast(list[str], cell["source"]))
        compile(source, str(NOTEBOOK_PATH), "exec")
        code_sources.append(source)

        tree = ast.parse(source)
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in ast.walk(tree)
        )

    combined_code = "\n".join(code_sources)
    assert "split_sentiment_row_indices(" in combined_code
    assert "SentimentDataset(" in combined_code
    assert "create_sentiment_dataloaders(" in combined_code
    assert "load_sequence_classifier(" in combined_code
    assert "create_fine_tuning_optimizer(" in combined_code
    assert "run_fine_tuning(" in combined_code
    assert "save_fine_tuning_artifacts(" in combined_code
    assert "RUN_FULL_TRAINING = False" in combined_code


def test_day5_notebook_explains_learning_and_leakage_boundaries() -> None:
    markdown = "\n".join(
        "".join(cast(list[str], cell["source"]))
        for cell in load_notebook_cells()
        if cell["cell_type"] == "markdown"
    )

    assert "frozen baseline" in markdown
    assert "model.train()" in markdown
    assert "model.eval()" in markdown
    assert "gradients" in markdown
    assert "outer test" in markdown.lower()
    assert "validation metrics, not final held-out test results" in markdown
