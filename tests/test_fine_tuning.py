from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizerBase

from transformers_learning import fine_tuning
from transformers_learning.datasets import SentimentDatasetValidationError
from transformers_learning.fine_tuning import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    NUM_SENTIMENT_LABELS,
    SentimentDataset,
    SentimentDatasetItem,
    create_fine_tuning_optimizer,
    create_sentiment_dataloaders,
    load_sequence_classifier,
    select_training_device,
    train_epoch,
)


class FakeSentimentTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, text: str, **options: object) -> BatchEncoding:
        self.calls.append((text, options))
        token_value = 11 if text == "Good movie" else 22
        return BatchEncoding(
            {
                "input_ids": torch.tensor([[1, token_value, 2, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            }
        )


def make_fake_tokenizer() -> tuple[PreTrainedTokenizerBase, FakeSentimentTokenizer]:
    fake = FakeSentimentTokenizer()
    return cast(PreTrainedTokenizerBase, fake), fake


def test_sentiment_dataset_tokenizes_lazily_and_returns_one_aligned_item() -> None:
    tokenizer, fake = make_fake_tokenizer()
    dataset = SentimentDataset(
        ["Good movie", "Bad movie"], [1, 0], tokenizer, max_length=4
    )

    assert len(dataset) == 2
    assert fake.calls == []

    item = dataset[1]

    assert fake.calls == [
        (
            "Bad movie",
            {
                "padding": "max_length",
                "truncation": True,
                "max_length": 4,
                "return_tensors": "pt",
            },
        )
    ]
    assert item["input_ids"].tolist() == [1, 22, 2, 0]
    assert item["attention_mask"].tolist() == [1, 1, 1, 0]
    assert item["input_ids"].shape == (4,)
    assert item["attention_mask"].shape == (4,)
    assert item["labels"].shape == ()
    assert item["input_ids"].dtype == torch.long
    assert item["attention_mask"].dtype == torch.long
    assert item["labels"].dtype == torch.long
    assert item["labels"].item() == 0


def test_sentiment_dataset_collates_to_expected_batch_shapes() -> None:
    tokenizer, _ = make_fake_tokenizer()
    dataset = SentimentDataset(
        ["Good movie", "Bad movie"], [1, 0], tokenizer, max_length=4
    )

    batch = next(iter(DataLoader(dataset, batch_size=2)))

    assert batch["input_ids"].shape == (2, 4)
    assert batch["attention_mask"].shape == (2, 4)
    assert batch["labels"].shape == (2,)
    assert batch["labels"].tolist() == [1, 0]


def test_sentiment_dataset_rejects_misaligned_inputs() -> None:
    tokenizer, _ = make_fake_tokenizer()

    with pytest.raises(ValueError, match="same number"):
        SentimentDataset(["Good", "Bad"], [1], tokenizer)


@pytest.mark.parametrize(
    ("texts", "labels", "message"),
    (
        ([], [], "contain rows"),
        (["Good", ""], [1, 0], "non-empty"),
        (["Good", "Bad"], [1, 2], "0 and 1"),
        (["Good", "Also good"], [1, 1], "both SST-2"),
    ),
)
def test_sentiment_dataset_rejects_invalid_examples(
    texts: list[str],
    labels: list[int],
    message: str,
) -> None:
    tokenizer, _ = make_fake_tokenizer()

    with pytest.raises(SentimentDatasetValidationError, match=message):
        SentimentDataset(texts, labels, tokenizer)


@pytest.mark.parametrize("max_length", (0, -1))
def test_sentiment_dataset_rejects_non_positive_max_length(max_length: int) -> None:
    tokenizer, _ = make_fake_tokenizer()

    with pytest.raises(ValueError, match="positive"):
        SentimentDataset(["Good", "Bad"], [1, 0], tokenizer, max_length=max_length)


def test_sentiment_dataset_rejects_malformed_tokenizer_shape() -> None:
    class WrongShapeTokenizer:
        def __call__(self, text: str, **options: object) -> BatchEncoding:
            return BatchEncoding(
                {
                    "input_ids": torch.tensor([[1, 2, 0]]),
                    "attention_mask": torch.tensor([[1, 1, 0]]),
                }
            )

    tokenizer = cast(PreTrainedTokenizerBase, WrongShapeTokenizer())
    dataset = SentimentDataset(["Good", "Bad"], [1, 0], tokenizer, max_length=4)

    with pytest.raises(ValueError, match=r"input_ids.*\[1, max_length\]"):
        dataset[0]


def test_create_sentiment_dataloaders_shuffles_only_training_data() -> None:
    tokenizer, _ = make_fake_tokenizer()
    train_dataset = SentimentDataset(
        ["Good movie", "Bad movie"], [1, 0], tokenizer, max_length=4
    )
    validation_dataset = SentimentDataset(
        ["Bad movie", "Good movie"], [0, 1], tokenizer, max_length=4
    )

    loaders = create_sentiment_dataloaders(train_dataset, validation_dataset)

    assert loaders.train.dataset is train_dataset
    assert loaders.validation.dataset is validation_dataset
    assert loaders.train.batch_size == DEFAULT_BATCH_SIZE == 16
    assert loaders.validation.batch_size == DEFAULT_BATCH_SIZE
    assert isinstance(loaders.train.sampler, RandomSampler)
    assert isinstance(loaders.validation.sampler, SequentialSampler)
    assert next(iter(loaders.validation))["labels"].tolist() == [0, 1]


@pytest.mark.parametrize("batch_size", (0, -1))
def test_create_sentiment_dataloaders_rejects_non_positive_batch_size(
    batch_size: int,
) -> None:
    tokenizer, _ = make_fake_tokenizer()
    dataset = SentimentDataset(
        ["Good movie", "Bad movie"], [1, 0], tokenizer, max_length=4
    )

    with pytest.raises(ValueError, match="positive"):
        create_sentiment_dataloaders(dataset, dataset, batch_size=batch_size)


def test_select_training_device_prefers_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert select_training_device() == torch.device("cuda")


def test_select_training_device_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert select_training_device() == torch.device("cpu")


def test_load_sequence_classifier_uses_binary_head_and_moves_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, object] = {}

    class FakeSequenceClassifier:
        def to(self, device: torch.device) -> "FakeSequenceClassifier":
            received["device"] = device
            return self

        def __call__(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> SimpleNamespace:
            assert input_ids.shape == attention_mask.shape
            return SimpleNamespace(
                logits=torch.zeros((input_ids.shape[0], NUM_SENTIMENT_LABELS))
            )

    fake_model = FakeSequenceClassifier()

    class FakeAutoModelForSequenceClassification:
        @staticmethod
        def from_pretrained(model_name: str, **options: object) -> PreTrainedModel:
            received["model_name"] = model_name
            received["options"] = options
            return cast(PreTrainedModel, fake_model)

    monkeypatch.setattr(
        fine_tuning,
        "AutoModelForSequenceClassification",
        FakeAutoModelForSequenceClassification,
    )
    tokenizer, _ = make_fake_tokenizer()
    dataset = SentimentDataset(
        ["Good movie", "Bad movie"], [1, 0], tokenizer, max_length=4
    )
    batch = next(
        iter(
            create_sentiment_dataloaders(
                dataset, dataset, batch_size=2
            ).validation
        )
    )

    setup = load_sequence_classifier("test-checkpoint", device=torch.device("cpu"))
    outputs = fake_model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    assert setup.model is fake_model
    assert setup.device == torch.device("cpu")
    assert received == {
        "model_name": "test-checkpoint",
        "options": {"num_labels": 2},
        "device": torch.device("cpu"),
    }
    assert batch["input_ids"].shape == (2, 4)
    assert outputs.logits.shape == (2, 2)


class TinyTrainingModel(torch.nn.Module):
    def __init__(self, events: list[str] | None = None) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))
        self.events = events
        self.received_devices: list[tuple[torch.device, torch.device, torch.device]] = []
        self.observed_losses: list[float] = []

    def train(self, mode: bool = True) -> "TinyTrainingModel":
        if self.events is not None:
            self.events.append("train")
        super().train(mode)
        return self

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> SimpleNamespace:
        if self.events is not None:
            self.events.append("forward")
        self.received_devices.append(
            (input_ids.device, attention_mask.device, labels.device)
        )
        scores = input_ids[:, 1].float() * self.weight
        logits = torch.stack((-scores, scores), dim=1)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        self.observed_losses.append(float(loss.detach().item()))
        if self.events is not None:
            loss.register_hook(lambda gradient: self._record_backward(gradient))
        return SimpleNamespace(loss=loss, logits=logits)

    def _record_backward(self, gradient: torch.Tensor) -> torch.Tensor:
        if self.events is not None:
            self.events.append("backward")
        return gradient


def make_ordered_training_loader(
    batch_size: int = 1,
) -> DataLoader[SentimentDatasetItem]:
    tokenizer, _ = make_fake_tokenizer()
    dataset = SentimentDataset(
        ["Good movie", "Bad movie"], [1, 0], tokenizer, max_length=4
    )
    return create_sentiment_dataloaders(
        dataset,
        dataset,
        batch_size=batch_size,
    ).validation


def test_create_fine_tuning_optimizer_uses_all_parameters_and_learning_rate() -> None:
    model = TinyTrainingModel()

    optimizer = create_fine_tuning_optimizer(cast(PreTrainedModel, model))

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == DEFAULT_LEARNING_RATE == 2e-5
    assert optimizer.param_groups[0]["params"] == [model.weight]


def test_train_epoch_updates_parameters_and_returns_mean_loss() -> None:
    model = TinyTrainingModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    initial_weight = model.weight.detach().clone()

    mean_loss = train_epoch(
        cast(PreTrainedModel, model),
        make_ordered_training_loader(batch_size=2),
        optimizer,
        torch.device("cpu"),
    )

    assert model.training is True
    assert mean_loss >= 0.0
    assert torch.isfinite(torch.tensor(mean_loss))
    assert not torch.equal(model.weight.detach(), initial_weight)
    assert model.received_devices == [
        (torch.device("cpu"), torch.device("cpu"), torch.device("cpu"))
    ]


def test_train_epoch_uses_the_required_operation_order_for_every_batch() -> None:
    events: list[str] = []
    model = TinyTrainingModel(events)

    class RecordingOptimizer:
        def zero_grad(self) -> None:
            events.append("zero_grad")
            model.weight.grad = None

        def step(self) -> None:
            events.append("step")

    mean_loss = train_epoch(
        cast(PreTrainedModel, model),
        make_ordered_training_loader(batch_size=1),
        cast(Optimizer, RecordingOptimizer()),
        torch.device("cpu"),
    )

    assert events == [
        "train",
        "zero_grad",
        "forward",
        "backward",
        "step",
        "zero_grad",
        "forward",
        "backward",
        "step",
    ]
    assert mean_loss == pytest.approx(
        sum(model.observed_losses) / len(model.observed_losses)
    )


def test_train_epoch_rejects_an_empty_dataloader() -> None:
    model = TinyTrainingModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    empty_loader = cast(DataLoader[SentimentDatasetItem], [])

    with pytest.raises(ValueError, match="at least one batch"):
        train_epoch(
            cast(PreTrainedModel, model),
            empty_loader,
            optimizer,
            torch.device("cpu"),
        )
