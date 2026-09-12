# День 5 — Fine-tuning модели

> Идентификатор: `transformers_day05`

## Цель дня

Дообучить трансформерную модель на выбранном датасете.

Используйте токенизатор из Дня 1 и ваш датасет.

## Задача 1 — Подготовка класса Dataset

Создайте класс для датасета:

```python
from torch.utils.data import Dataset


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }
```

## Задача 2 — Загрузка и подготовка данных

1. Загрузите датасет:

   ```python
   import pandas as pd

   df = pd.read_csv("your_dataset.csv")
   texts = df["text"].tolist()
   labels = df["label"].tolist()
   ```

2. Разделите данные на train/validation:

   ```python
   from sklearn.model_selection import train_test_split

   train_texts, val_texts, train_labels, val_labels = train_test_split(
       texts,
       labels,
       test_size=0.2,
       random_state=42,
       stratify=labels,
   )
   ```

3. Создайте объекты Dataset:

   ```python
   train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
   val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
   ```

## Задача 3 — Загрузка модели для классификации

1. Импортируйте `AutoModelForSequenceClassification` и создайте модель:

   ```python
   from transformers import AutoModelForSequenceClassification

   # 2 для бинарной классификации, 3 для positive/negative/neutral.
   num_labels = len(set(labels))

   model = AutoModelForSequenceClassification.from_pretrained(
       model_name,
       num_labels=num_labels,
   )
   ```

2. Создайте DataLoader для обучающей и валидационной выборок:

   ```python
   from torch.utils.data import DataLoader

   train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=16)
   ```

## Задача 4 — Настройка обучения

1. Импортируйте оптимизатор и метрики:

   ```python
   from transformers import AdamW
   from sklearn.metrics import accuracy_score, f1_score
   ```

2. Создайте оптимизатор:

   ```python
   optimizer = AdamW(model.parameters(), lr=2e-5)
   ```

3. Переместите модель на GPU, если он доступен:

   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model.to(device)
   ```

## Задача 5 — Функция обучения одной эпохи

```python
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

## Задача 6 — Функция оценки

```python
def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="macro")

    return accuracy, f1
```

## Задача 7 — Обучение модели

Обучите модель в течение трёх-пяти эпох:

```python
num_epochs = 3

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_acc, val_f1 = evaluate(model, val_loader, device)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Accuracy: {val_acc:.4f}")
    print(f"Val F1: {val_f1:.4f}")
    print("-" * 50)
```

## Задача 8 — Сохранение модели

1. Сохраните модель и токенизатор:

   ```python
   model.save_pretrained("./fine_tuned_model")
   tokenizer.save_pretrained("./fine_tuned_model")
   ```

2. Сохраните метрики:

   ```python
   with open("fine_tuned_results.txt", "w") as f:
       f.write(f"Final Validation F1: {val_f1:.4f}\n")
       f.write(f"Final Validation Accuracy: {val_acc:.4f}\n")
   ```

## Чекпоинт

К концу дня у вас должны быть:

- [ ] обученная fine-tuned модель;
- [ ] функции для обучения и оценки;
- [ ] модель, сохранённая в папке `fine_tuned_model`;
- [ ] зафиксированные метрики качества.

> В День 6 вы сравните fine-tuned модель с baseline из Дня 4.
