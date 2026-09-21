import ast
import json
from pathlib import Path
from typing import Any, cast

NOTEBOOK_PATH = Path("notebooks/day6_model_comparison.ipynb")


def load_notebook_cells() -> list[dict[str, Any]]:
    """Load the Day 6 notebook cells for structural contract tests."""

    notebook = cast(
        dict[str, Any],
        json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8")),
    )
    assert notebook["nbformat"] == 4
    return cast(list[dict[str, Any]], notebook["cells"])


def test_day6_notebook_is_thin_and_calls_reusable_pipeline() -> None:
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
    assert "RUN_DAY6_INTEGRATION = False" in combined_code
    assert "prepare_comparison_dataset(" in combined_code
    assert "recreate_frozen_baseline(" in combined_code
    assert "load_fine_tuned_inference(" in combined_code
    assert "compare_five_examples(" in combined_code
    assert "evaluate_paired_holdout(" in combined_code
    assert "save_paired_confusion_matrices(" in combined_code
    assert "save_comparison_results(" in combined_code


def test_day6_notebook_explains_inference_and_evaluation_boundaries() -> None:
    markdown = "\n".join(
        "".join(cast(list[str], cell["source"]))
        for cell in load_notebook_cells()
        if cell["cell_type"] == "markdown"
    )

    assert "model.eval()" in markdown
    assert "torch.no_grad()" in markdown
    assert "outer test" in markdown.lower()
    assert "Macro F1" in markdown
    assert "accuracy" in markdown.lower()
    assert "probabilities" in markdown
    assert "Day 7" in markdown
