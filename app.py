"""Thin local Gradio application for the Day 7 sentiment demo."""

import sys
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = str(PROJECT_ROOT / "src")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from transformers_learning.demo import (
    SentimentDemoService,
    load_sentiment_demo_service,
)


def create_interface(
    service: SentimentDemoService | None = None,
) -> gr.Interface:
    """Construct the UI, loading the local model once when no service is given."""

    demo_service = (
        load_sentiment_demo_service() if service is None else service
    )
    return gr.Interface(
        fn=demo_service.predict_formatted,
        inputs=gr.Textbox(
            lines=3,
            label="Text",
            placeholder="Enter a movie review to analyze...",
        ),
        outputs=gr.Textbox(label="Sentiment prediction"),
        title="SST-2 sentiment analysis",
        description=(
            "Classifies text as negative or positive. Probabilities are not "
            "calibrated certainty, and this binary model has no neutral class."
        ),
    )


def launch_app(interface: gr.Interface | None = None) -> None:
    """Launch the interface on this machine without enabling Gradio sharing."""

    app_interface = interface if interface is not None else create_interface()
    app_interface.launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    launch_app()
