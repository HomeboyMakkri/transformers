[9/12/2026 8:21 AM] HwBot | ItGrind: 🔄 🔄 Трансформеры

📝 День 5: transformers_day05

День 5 — Fine-tuning модели

ЦЕЛЬ ДНЯ:
Дообучить трансформерную модель на вашем датасете (fine-tuning).

Используйте код из Дня 1 (токенизатор) и ваш датасет.

ЗАДАЧА 1: Подготовка Dataset класса

1. Создайте класс для датасета:
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
            padding=&apos;max_length&apos;,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            &apos;input_ids&apos;: encoding[&apos;input_ids&apos;].flatten(),
            &apos;attention_mask&apos;: encoding[&apos;attention_mask&apos;].flatten(),
            &apos;labels&apos;: torch.tensor(label, dtype=torch.long)
        }

ЗАДАЧА 2: Загрузка и подготовка данных

1. Загрузите ваш датасет:
import pandas as pd

df = pd.read_csv(&apos;your_dataset.csv&apos;)
texts = df[&apos;text&apos;].tolist()
labels = df[&apos;label&apos;].tolist()

2. Разделите на train/validation:
from sklearn.model_selection import train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

3. Создайте Dataset объекты:
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

ЗАДАЧА 3: Загрузка модели для классификации

1. Импортируйте AutoModelForSequenceClassification:
from transformers import AutoModelForSequenceClassification

# Определите количество классов
num_labels = len(set(labels))  # 2 для бинарной, 3 для positive/negative/neutral

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

2. Создайте DataLoaders:
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

ЗАДАЧА 4: Настройка обучения

1. Импортируйте оптимизатор и метрики:
from transformers import AdamW
from sklearn.metrics import accuracy_score, f1_score

2. Создайте оптимизатор:
optimizer = AdamW(model.parameters(), lr=2e-5)

3. Переместите модель на GPU если доступно:
device = torch.device(&apos;cuda&apos; if torch.cuda.is_available() else &apos;cpu&apos;)
model.to(device)

ЗАДАЧА 5: Функция для обучения одной эпохи

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch[&apos;input_ids&apos;].to(device)
        attention_mask = batch[&apos;attention_mask&apos;].to(device)
        labels = batch[&apos;labels&apos;].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

ЗАДАЧА 6: Функция для оценки

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[&apos;input_ids&apos;].to(device)
            attention_mask = batch[&apos;attention_mask&apos;].to(device)
            labels = batch[&apos;labels&apos;].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average=&apos;macro&apos;)

    return accuracy, f1

ЗАДАЧА 7: Обучение модели

1. Обучите модель на 3-5 эпох:
num_epochs = 3

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_acc, val_f1 = evaluate(model, val_loader, device)

    print(f&apos;Epoch {epoch+1}/{num_epochs}&apos;)
    print(f&apos;Train Loss: {train_loss:.4f}&apos;)
    print(f&apos;Val Accuracy: {val_acc:.4f}&apos;)
    print(f&apos;Val F1: {val_f1:.4f}&apos;)
    print(&apos;-&apos; * 50)

ЗАДАЧА 8: Сохранение модели

1. Сохраните модель:
model.save_pretrained(&apos;./fine_tuned_model&apos;)
tokenizer.save_pretrained(&apos;./fine_tuned_model&apos;)

2. Сохраните метрики:
with open(&apos;fine_tuned_results.txt&apos;, &apos;w&apos;) as f:
    f.write(f&apos;Final Validation F1: {val_f1:.4f}\n&apos;)
    f.write(f&apos;Final Validation Accuracy: {val_acc:.4f}\n&apos;)

ЧЕКПОИНТ:
К концу дня вы должны иметь:
• Обученную fine-tuned модель
• Функции для обучения и оценки
• Сохранённую модель в папке fine_tuned_model
• Зафиксированные метрики качества

В День 6 вы сравните fine-tuned модель с baseline из Дня 4.
