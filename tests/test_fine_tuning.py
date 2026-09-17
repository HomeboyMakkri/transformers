from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
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
    DEFAULT_NUM_EPOCHS,
    NUM_SENTIMENT_LABELS,
    FineTuningEpochMetrics,
    SentimentDataset,
    SentimentDatasetItem,
    ValidationEvaluation,
    create_fine_tuning_optimizer,
    create_sentiment_dataloaders,
    evaluate_sequence_classifier,
    load_sequence_classifier,
    run_fine_tuning,
    save_fine_tuning_artifacts,
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


class TinyEvaluationModel(torch.nn.Module):
    def __init__(self, *, always_negative: bool = False) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.25))
        self.always_negative = always_negative
        self.grad_enabled: list[bool] = []
        self.received_devices: list[tuple[torch.device, torch.device]] = []

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        self.grad_enabled.append(torch.is_grad_enabled())
        self.received_devices.append((input_ids.device, attention_mask.device))
        if self.always_negative:
            positive_scores = torch.zeros(input_ids.shape[0])
        else:
            positive_scores = (input_ids[:, 1] == 11).float()
        logits = torch.stack((1.0 - positive_scores, positive_scores), dim=1)
        logits = logits + self.bias * 0.0
        return SimpleNamespace(logits=logits)


def test_evaluate_sequence_classifier_preserves_order_and_disables_gradients() -> None:
    model = TinyEvaluationModel()
    initial_bias = model.bias.detach().clone()

    evaluation = evaluate_sequence_classifier(
        cast(PreTrainedModel, model),
        make_ordered_training_loader(batch_size=1),
        torch.device("cpu"),
    )

    assert model.training is False
    assert model.grad_enabled == [False, False]
    assert model.received_devices == [
        (torch.device("cpu"), torch.device("cpu")),
        (torch.device("cpu"), torch.device("cpu")),
    ]
    assert torch.equal(model.bias.detach(), initial_bias)
    assert model.bias.grad is None
    assert evaluation.labels.tolist() == [1, 0]
    assert evaluation.predictions.tolist() == [1, 0]
    assert evaluation.accuracy == 1.0
    assert evaluation.macro_f1 == 1.0


def test_evaluate_sequence_classifier_calculates_binary_macro_f1() -> None:
    model = TinyEvaluationModel(always_negative=True)

    evaluation = evaluate_sequence_classifier(
        cast(PreTrainedModel, model),
        make_ordered_training_loader(batch_size=2),
        torch.device("cpu"),
    )

    assert evaluation.labels.tolist() == [1, 0]
    assert evaluation.predictions.tolist() == [0, 0]
    assert evaluation.accuracy == 0.5
    assert evaluation.macro_f1 == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize(
    ("logits", "message"),
    (
        (torch.zeros(2), r"shape \[batch, 2\]"),
        (torch.zeros((2, 3)), r"shape \[batch, 2\]"),
        (torch.zeros((1, 2)), "align"),
        (torch.tensor([[float("nan"), 0.0], [0.0, 0.0]]), "finite"),
    ),
)
def test_evaluate_sequence_classifier_rejects_malformed_logits(
    logits: torch.Tensor,
    message: str,
) -> None:
    class MalformedEvaluationModel(torch.nn.Module):
        def forward(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> SimpleNamespace:
            return SimpleNamespace(logits=logits)

    with pytest.raises(ValueError, match=message):
        evaluate_sequence_classifier(
            cast(PreTrainedModel, MalformedEvaluationModel()),
            make_ordered_training_loader(batch_size=2),
            torch.device("cpu"),
        )


def test_evaluate_sequence_classifier_rejects_an_empty_dataloader() -> None:
    empty_loader = cast(DataLoader[SentimentDatasetItem], [])

    with pytest.raises(ValueError, match="at least one batch"):
        evaluate_sequence_classifier(
            cast(PreTrainedModel, TinyEvaluationModel()),
            empty_loader,
            torch.device("cpu"),
        )


def test_evaluate_sequence_classifier_requires_logits() -> None:
    class MissingLogitsModel(torch.nn.Module):
        def forward(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> SimpleNamespace:
            return SimpleNamespace(logits=None)

    with pytest.raises(RuntimeError, match="does not contain logits"):
        evaluate_sequence_classifier(
            cast(PreTrainedModel, MissingLogitsModel()),
            make_ordered_training_loader(batch_size=2),
            torch.device("cpu"),
        )


def test_run_fine_tuning_records_exactly_three_ordered_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    train_losses = (0.9, 0.6, 0.4)
    validation_accuracies = (0.7, 0.6, 0.8)
    validation_macro_f1_scores = (0.65, 0.55, 0.75)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    tokenizer, _ = make_fake_tokenizer()
    dataset = SentimentDataset(
        ["Good movie", "Bad movie"], [1, 0], tokenizer, max_length=4
    )
    dataloaders = create_sentiment_dataloaders(dataset, dataset, batch_size=2)

    def fake_train_epoch(
        received_model: PreTrainedModel,
        received_dataloader: DataLoader[SentimentDatasetItem],
        received_optimizer: Optimizer,
        received_device: torch.device,
    ) -> float:
        epoch_index = len(events) // 2
        events.append(f"train-{epoch_index + 1}")
        assert received_model is model
        assert received_dataloader is dataloaders.train
        assert received_optimizer is optimizer
        assert received_device == torch.device("cpu")
        received_model.train()
        return train_losses[epoch_index]

    def fake_evaluate_sequence_classifier(
        received_model: PreTrainedModel,
        received_dataloader: DataLoader[SentimentDatasetItem],
        received_device: torch.device,
    ) -> ValidationEvaluation:
        epoch_index = (len(events) + 1) // 2
        events.append(f"validation-{epoch_index}")
        assert received_model is model
        assert received_dataloader is dataloaders.validation
        assert received_device == torch.device("cpu")
        received_model.eval()
        return ValidationEvaluation(
            labels=np.array([1, 0], dtype=np.int64),
            predictions=np.array([1, 0], dtype=np.int64),
            accuracy=validation_accuracies[epoch_index - 1],
            macro_f1=validation_macro_f1_scores[epoch_index - 1],
        )

    monkeypatch.setattr(fine_tuning, "train_epoch", fake_train_epoch)
    monkeypatch.setattr(
        fine_tuning,
        "evaluate_sequence_classifier",
        fake_evaluate_sequence_classifier,
    )

    history = run_fine_tuning(
        cast(PreTrainedModel, model),
        dataloaders,
        optimizer,
        torch.device("cpu"),
    )

    assert DEFAULT_NUM_EPOCHS == 3
    assert events == [
        "train-1",
        "validation-1",
        "train-2",
        "validation-2",
        "train-3",
        "validation-3",
    ]
    assert history == (
        FineTuningEpochMetrics(1, 0.9, 0.7, 0.65),
        FineTuningEpochMetrics(2, 0.6, 0.6, 0.55),
        FineTuningEpochMetrics(3, 0.4, 0.8, 0.75),
    )
    assert model.training is False


def make_completed_fine_tuning_history() -> tuple[FineTuningEpochMetrics, ...]:
    return (
        FineTuningEpochMetrics(1, 0.9, 0.70, 0.65),
        FineTuningEpochMetrics(2, 0.6, 0.75, 0.70),
        FineTuningEpochMetrics(3, 0.4, 0.80, 0.77),
    )


def test_save_fine_tuning_artifacts_can_be_reloaded_and_records_context(
    tmp_path: Path,
) -> None:
    class FakeSavableModel:
        def save_pretrained(self, directory: Path) -> None:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "model.marker").write_text("model", encoding="utf-8")

        @classmethod
        def from_pretrained(cls, directory: Path) -> "FakeSavableModel":
            assert (directory / "model.marker").read_text(encoding="utf-8") == "model"
            return cls()

    class FakeSavableTokenizer:
        def save_pretrained(self, directory: Path) -> None:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "tokenizer.marker").write_text(
                "tokenizer", encoding="utf-8"
            )

        @classmethod
        def from_pretrained(cls, directory: Path) -> "FakeSavableTokenizer":
            assert (
                directory / "tokenizer.marker"
            ).read_text(encoding="utf-8") == "tokenizer"
            return cls()

    model_directory = tmp_path / "fine_tuned_model"
    results_path = tmp_path / "results" / "fine_tuned_results.txt"

    save_fine_tuning_artifacts(
        cast(PreTrainedModel, FakeSavableModel()),
        cast(PreTrainedTokenizerBase, FakeSavableTokenizer()),
        make_completed_fine_tuning_history(),
        device=torch.device("cpu"),
        model_directory=model_directory,
        results_path=results_path,
        model_name="test-checkpoint",
        dataset_identifier="test-dataset",
    )

    assert isinstance(
        FakeSavableModel.from_pretrained(model_directory), FakeSavableModel
    )
    assert isinstance(
        FakeSavableTokenizer.from_pretrained(model_directory), FakeSavableTokenizer
    )
    assert results_path.read_text(encoding="utf-8") == (
        "validation_accuracy: 0.800000\n"
        "validation_macro_f1: 0.770000\n"
        "final_epoch: 3\n"
        "model_name: test-checkpoint\n"
        "dataset_identifier: test-dataset\n"
        "label_mapping: 0=negative, 1=positive\n"
        "outer_test_size: 0.2\n"
        "inner_validation_size: 0.2\n"
        "random_state: 42\n"
        "epochs: 3\n"
        "batch_size: 16\n"
        "max_length: 128\n"
        "learning_rate: 2e-05\n"
        "device: cpu\n"
    )


def test_save_fine_tuning_artifacts_validates_history_before_writing(
    tmp_path: Path,
) -> None:
    class MustNotSave:
        def save_pretrained(self, directory: Path) -> None:
            raise AssertionError("Invalid history must be rejected before saving")

    with pytest.raises(ValueError, match="exactly three epochs"):
        save_fine_tuning_artifacts(
            cast(PreTrainedModel, MustNotSave()),
            cast(PreTrainedTokenizerBase, MustNotSave()),
            make_completed_fine_tuning_history()[:2],
            device=torch.device("cpu"),
            model_directory=tmp_path / "model",
            results_path=tmp_path / "results.txt",
        )

    assert not (tmp_path / "model").exists()
    assert not (tmp_path / "results.txt").exists()
