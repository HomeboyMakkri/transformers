import ast
import json
from pathlib import Path
from typing import Any, cast

NOTEBOOK_PATH = Path("notebooks/day7_error_analysis_and_demo.ipynb")


def load_notebook_cells() -> list[dict[str, Any]]:
    """Load the incremental Day 7 notebook for structural contract tests."""

    notebook = cast(
        dict[str, Any],
        json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8")),
    )
    assert notebook["nbformat"] == 4
    return cast(list[dict[str, Any]], notebook["cells"])


def test_day7_notebook_is_thin_and_exposes_d7_01_checkpoint() -> None:
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
    assert "RUN_DAY7_D7_01_INTEGRATION =" in combined_code
    assert "run_day7_holdout_inference(" in combined_code
    assert "build_day7_error_tables(" in combined_code
    assert "summarize_day7_errors(" in combined_code
    assert "build_day7_length_summary_table(" in combined_code
    assert "save_day7_error_report(" in combined_code
    assert "build_day7_prediction_preview(" in combined_code
    assert "SST2_LABEL_MAP" in combined_code
    assert "fine_tuned_model" in combined_code


def test_day7_notebook_explains_inference_and_leakage_boundaries() -> None:
    markdown = "\n".join(
        "".join(cast(list[str], cell["source"]))
        for cell in load_notebook_cells()
        if cell["cell_type"] == "markdown"
    )

    assert "[batch, sequence]" in markdown
    assert "[batch, 2]" in markdown
    assert "model.eval()" in markdown
    assert "torch.no_grad()" in markdown
    assert "outer holdout" in markdown.lower()
    assert "D7-02" in markdown
    assert "false positive" in markdown.lower()
    assert "causal explanation" in markdown.lower()
    assert "qualitative hypotheses" in markdown.lower()
    assert "error_analysis.txt" in markdown
