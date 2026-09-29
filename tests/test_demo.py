from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from transformers_learning import demo
from transformers_learning.comparison import (
    Day6ArtifactInputs,
    FineTunedInferenceSetup,
    SentimentPrediction,
)
from transformers_learning.demo import (
    create_sentiment_demo_service,
    load_sentiment_demo_service,
)


def make_prediction(
    text: str,
    prediction: int = 1,
    probabilities: np.ndarray | None = None,
) -> SentimentPrediction:
    """Build one injected binary prediction for service tests."""

    if probabilities is None:
        probabilities = np.array([0.1234, 0.8766], dtype=np.float64)
    return SentimentPrediction(
        text=text,
        prediction=prediction,
        probabilities=probabilities,
    )


def test_demo_service_calls_predictor_once_and_formats_binary_result() -> None:
    received: list[str] = []

    def fake_predictor(text: str) -> tuple[SentimentPrediction, ...]:
        received.append(text)
        return (make_prediction(text),)

    service = create_sentiment_demo_service(fake_predictor)

    prediction = service.predict("A thoughtful and moving film.")
    formatted = service.predict_formatted("A second excellent film.")

    assert received == [
        "A thoughtful and moving film.",
        "A second excellent film.",
    ]
    assert prediction.text == "A thoughtful and moving film."
    assert prediction.predicted_label == 1
    assert prediction.label_name == "positive"
    assert prediction.probabilities == pytest.approx((0.1234, 0.8766))
    assert prediction.predicted_confidence == pytest.approx(0.8766)
    assert formatted == (
        "Prediction: positive\n\n"
        "Probabilities:\n"
        "negative: 12.34%\n"
        "positive: 87.66%\n\n"
        "Interpretation: confidence is relative to the binary SST-2 classes, "
        "is not calibrated certainty, and does not provide a neutral class."
    )


@pytest.mark.parametrize("text", ("", "   ", "\n\t"))
def test_demo_service_rejects_empty_text_without_calling_predictor(text: str) -> None:
    call_count = 0

    def fake_predictor(received_text: str) -> tuple[SentimentPrediction, ...]:
        nonlocal call_count
        call_count += 1
        return (make_prediction(received_text),)

    service = create_sentiment_demo_service(fake_predictor)

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        service.predict_formatted(text)

    assert call_count == 0


@pytest.mark.parametrize(
    ("records", "error_type", "message"),
    (
        ((), ValueError, "exactly one"),
        (
            (make_prediction("text"), make_prediction("text")),
            ValueError,
            "exactly one",
        ),
        ((make_prediction("different"),), ValueError, "preserve"),
        ((make_prediction("text", prediction=2),), ValueError, "labels 0 and 1"),
        (
            (make_prediction("text", probabilities=np.array([0.8])),),
            ValueError,
            "shape",
        ),
        (
            (
                make_prediction(
                    "text",
                    probabilities=np.array([float("nan"), 0.8]),
                ),
            ),
            ValueError,
            "finite",
        ),
        (
            (make_prediction("text", probabilities=np.array([0.2, 0.6])),),
            ValueError,
            "sum to one",
        ),
        (
            (
                make_prediction(
                    "text",
                    prediction=0,
                    probabilities=np.array([0.2, 0.8]),
                ),
            ),
            ValueError,
            "argmax",
        ),
    ),
)
def test_demo_service_rejects_malformed_or_multiple_results(
    records: tuple[SentimentPrediction, ...],
    error_type: type[Exception],
    message: str,
) -> None:
    service = create_sentiment_demo_service(lambda text: records)

    with pytest.raises(error_type, match=message):
        service.predict("text")


def test_load_demo_service_loads_artifact_once_and_reuses_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_directory = Path("fine_tuned_model")
    artifact_inputs = Day6ArtifactInputs(model_directory)
    setup = FineTunedInferenceSetup(
        model=cast(PreTrainedModel, object()),
        tokenizer=cast(PreTrainedTokenizerBase, object()),
        device=torch.device("cpu"),
    )
    calls: list[tuple[str, object]] = []

    def fake_get(directory: Path) -> Day6ArtifactInputs:
        calls.append(("get", directory))
        return artifact_inputs

    def fake_load(
        inputs: Day6ArtifactInputs,
        device: torch.device | None = None,
    ) -> FineTunedInferenceSetup:
        calls.append(("load", (inputs, device)))
        return setup

    def fake_predict(
        text: str,
        received_setup: FineTunedInferenceSetup,
        batch_size: int,
    ) -> tuple[SentimentPrediction, ...]:
        calls.append(("predict", (text, received_setup, batch_size)))
        return (make_prediction(text),)

    monkeypatch.setattr(demo, "get_day6_artifact_inputs", fake_get)
    monkeypatch.setattr(demo, "load_fine_tuned_inference", fake_load)
    monkeypatch.setattr(demo, "predict_fine_tuned", fake_predict)

    service = load_sentiment_demo_service(
        model_directory,
        device=torch.device("cpu"),
    )
    first = service.predict("first")
    second = service.predict("second")

    assert first.label_name == "positive"
    assert second.label_name == "positive"
    assert calls == [
        ("get", model_directory),
        ("load", (artifact_inputs, torch.device("cpu"))),
        ("predict", ("first", setup, 1)),
        ("predict", ("second", setup, 1)),
    ]
