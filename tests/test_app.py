import importlib
from typing import cast

import gradio as gr
import numpy as np
import pytest

import app
from transformers_learning import demo
from transformers_learning.comparison import SentimentPrediction
from transformers_learning.demo import (
    SentimentDemoService,
    create_sentiment_demo_service,
    load_sentiment_demo_service,
)


def make_service() -> tuple[SentimentDemoService, list[str]]:
    """Build an offline service and record submitted callback values."""

    received: list[str] = []

    def predict(text: str) -> tuple[SentimentPrediction, ...]:
        received.append(text)
        return (
            SentimentPrediction(
                text=text,
                prediction=1,
                probabilities=np.array([0.1, 0.9]),
            ),
        )

    return create_sentiment_demo_service(predict), received


def test_import_does_not_load_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_loaded() -> SentimentDemoService:
        raise AssertionError("Importing app must not load the model")

    monkeypatch.setattr(demo, "load_sentiment_demo_service", fail_if_loaded)
    importlib.reload(app)
    monkeypatch.setattr(app, "load_sentiment_demo_service", load_sentiment_demo_service)


def test_create_interface_wires_callback_without_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, received = make_service()

    def fail_if_loaded() -> SentimentDemoService:
        raise AssertionError("Injected service must avoid loading a model")

    monkeypatch.setattr(app, "load_sentiment_demo_service", fail_if_loaded)
    interface = app.create_interface(service)

    assert isinstance(interface, gr.Interface)
    assert interface.fn("A lovely film") == (
        "Prediction: positive\n\n"
        "Probabilities:\n"
        "negative: 10.00%\n"
        "positive: 90.00%\n\n"
        "Interpretation: confidence is relative to the binary SST-2 classes, "
        "is not calibrated certainty, and does not provide a neutral class."
    )
    assert received == ["A lovely film"]


def test_create_interface_loads_default_service_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _ = make_service()
    load_count = 0

    def load_service() -> SentimentDemoService:
        nonlocal load_count
        load_count += 1
        return service

    monkeypatch.setattr(app, "load_sentiment_demo_service", load_service)

    interface = app.create_interface()

    assert isinstance(interface, gr.Interface)
    assert load_count == 1


def test_launch_app_binds_to_localhost_without_sharing(monkeypatch) -> None:
    received: dict[str, object] = {}

    class FakeInterface:
        def launch(self, **kwargs: object) -> None:
            received.update(kwargs)

    app.launch_app(cast(gr.Interface, FakeInterface()))

    assert received == {"server_name": "127.0.0.1", "share": False}
