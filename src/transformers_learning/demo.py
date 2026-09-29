"""Application-neutral single-text inference for the Day 7 demo."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from .comparison import (
    SentimentPrediction,
    get_day6_artifact_inputs,
    load_fine_tuned_inference,
    predict_fine_tuned,
)
from .datasets import SST2_LABEL_MAP
from .fine_tuning import DEFAULT_FINE_TUNED_MODEL_DIRECTORY

SingleTextPredictor = Callable[[str], Sequence[SentimentPrediction]]


@dataclass(frozen=True)
class DemoPrediction:
    """One validated binary prediction ready for UI-independent formatting."""

    text: str
    predicted_label: int
    label_name: str
    probabilities: tuple[float, float]
    predicted_confidence: float


@dataclass(frozen=True)
class SentimentDemoService:
    """Validate and format predictions from one already-constructed predictor."""

    _predictor: SingleTextPredictor = field(repr=False)

    def predict(self, text: str) -> DemoPrediction:
        """Predict one non-empty text through the bound inference path once."""

        normalized_text = _validate_demo_text(text)
        predictions = tuple(self._predictor(normalized_text))
        if len(predictions) != 1:
            raise ValueError("Demo inference must return exactly one prediction")
        prediction = predictions[0]
        if not isinstance(prediction, SentimentPrediction):
            raise TypeError("Demo inference must return SentimentPrediction records")
        if prediction.text != normalized_text:
            raise ValueError("Demo prediction text must preserve the submitted text")
        predicted_label, probabilities = _validate_demo_prediction(prediction)
        return DemoPrediction(
            text=normalized_text,
            predicted_label=predicted_label,
            label_name=SST2_LABEL_MAP[predicted_label],
            probabilities=(
                float(probabilities[0]),
                float(probabilities[1]),
            ),
            predicted_confidence=float(probabilities[predicted_label]),
        )

    def predict_formatted(self, text: str) -> str:
        """Return a stable human-readable result for a future UI callback."""

        return format_demo_prediction(self.predict(text))


def create_sentiment_demo_service(
    predictor: SingleTextPredictor,
) -> SentimentDemoService:
    """Create a service around an injected predictor without loading a model."""

    if not callable(predictor):
        raise TypeError("predictor must be callable")
    return SentimentDemoService(_predictor=predictor)


def load_sentiment_demo_service(
    fine_tuned_model_directory: Path = DEFAULT_FINE_TUNED_MODEL_DIRECTORY,
    device: torch.device | None = None,
) -> SentimentDemoService:
    """Load the local classifier once and bind it to single-text inference."""

    artifact_inputs = get_day6_artifact_inputs(fine_tuned_model_directory)
    setup = load_fine_tuned_inference(artifact_inputs, device=device)

    def predict_one(text: str) -> tuple[SentimentPrediction, ...]:
        return predict_fine_tuned(text, setup, batch_size=1)

    return create_sentiment_demo_service(predict_one)


def format_demo_prediction(prediction: DemoPrediction) -> str:
    """Format binary probabilities without implying calibration or neutrality."""

    negative_probability, positive_probability = prediction.probabilities
    return (
        f"Prediction: {prediction.label_name}\n\n"
        "Probabilities:\n"
        f"negative: {negative_probability:.2%}\n"
        f"positive: {positive_probability:.2%}\n\n"
        "Interpretation: confidence is relative to the binary SST-2 classes, "
        "is not calibrated certainty, and does not provide a neutral class."
    )


def _validate_demo_text(text: str) -> str:
    """Reject non-string, empty, or whitespace-only UI input."""

    if not isinstance(text, str):
        raise TypeError("Demo text must be a string")
    if not text.strip():
        raise ValueError("Demo text must not be empty or whitespace-only")
    return text


def _validate_demo_prediction(
    prediction: SentimentPrediction,
) -> tuple[int, npt.NDArray[np.float64]]:
    """Validate binary label and probabilities returned by injected inference."""

    predicted_label = prediction.prediction
    if isinstance(predicted_label, bool) or predicted_label not in SST2_LABEL_MAP:
        raise ValueError("Demo prediction must use SST-2 labels 0 and 1")
    probabilities = np.asarray(prediction.probabilities)
    if probabilities.shape != (2,):
        raise ValueError("Demo probabilities must have shape [2]")
    if not np.issubdtype(probabilities.dtype, np.number):
        raise TypeError("Demo probabilities must be numeric")
    numeric_probabilities = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(numeric_probabilities).all():
        raise ValueError("Demo probabilities must contain only finite values")
    if np.any(numeric_probabilities < 0.0) or np.any(
        numeric_probabilities > 1.0
    ):
        raise ValueError("Demo probabilities must be between zero and one")
    if not np.isclose(numeric_probabilities.sum(), 1.0):
        raise ValueError("Demo probabilities must sum to one")
    if int(numeric_probabilities.argmax()) != predicted_label:
        raise ValueError("Demo predicted label must match probability argmax")
    return predicted_label, numeric_probabilities
