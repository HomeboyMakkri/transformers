📝 День 7: transformers_day07

День 7 — Анализ ошибок и мини-приложение

ЦЕЛЬ ДНЯ:
Провести детальный анализ ошибок модели и создать простое приложение для демонстрации.

Используйте код и модели из предыдущих дней.

ЗАДАЧА 1: Анализ False Positive и False Negative

1. Найдите ошибки модели:
import pandas as pd

# Загрузите данные и предсказания из Дня 6
df_test = pd.DataFrame({
    &apos;text&apos;: test_texts,
    &apos;true_label&apos;: test_labels,
    &apos;pred_label&apos;: y_pred_ft
})

# Найдите ошибки
errors = df_test[df_test[&apos;true_label&apos;] != df_test[&apos;pred_label&apos;]]

# False Positives (предсказала positive, а было negative/neutral)
fp = errors[(errors[&apos;pred_label&apos;] == 1) & (errors[&apos;true_label&apos;] == 0)]

# False Negatives (предсказала negative, а было positive)
fn = errors[(errors[&apos;pred_label&apos;] == 0) & (errors[&apos;true_label&apos;] == 1)]

print(f&apos;Всего ошибок: {len(errors)}&apos;)
print(f&apos;False Positives: {len(fp)}&apos;)
print(f&apos;False Negatives: {len(fn)}&apos;)

ЗАДАЧА 2: Анализ паттернов ошибок

1. Выведите примеры ошибок и найдите закономерности:
print("\n=== FALSE POSITIVES (сказали good, а было bad) ===")
for idx, row in fp.head(5).iterrows():
    print(f&apos;\nТекст: {row["text"][:100]}...&apos;)
    print(f&apos;Истинный класс: {row["true_label"]}, Предсказан: {row["pred_label"]}&apos;)

print("\n=== FALSE NEGATIVES (сказали bad, а было good) ===")
for idx, row in fn.head(5).iterrows():
    print(f&apos;\nТекст: {row["text"][:100]}...&apos;)
    print(f&apos;Истинный класс: {row["true_label"]}, Предсказан: {row["pred_label"]}&apos;)

2. Проанализируйте длину ошибочных текстов:
errors[&apos;text_length&apos;] = errors[&apos;text&apos;].str.len()
print(f&apos;\nСредняя длина ошибочных текстов: {errors["text_length"].mean():.0f}&apos;)
print(f&apos;Средняя длина всех текстов: {df_test["text"].str.len().mean():.0f}&apos;)

3. Сохраните анализ в файл:
with open(&apos;error_analysis.txt&apos;, &apos;w&apos;) as f:
    f.write(&apos;=== АНАЛИЗ ОШИБОК ===\n\n&apos;)
    f.write(f&apos;Всего ошибок: {len(errors)}\n&apos;)
    f.write(f&apos;False Positives: {len(fp)}\n&apos;)
    f.write(f&apos;False Negatives: {len(fn)}\n\n&apos;)

    f.write(&apos;=== ПРИМЕРЫ FALSE POSITIVES ===\n&apos;)
    for idx, row in fp.head(5).iterrows():
        f.write(f&apos;\nТекст: {row["text"]}\n&apos;)
        f.write(f&apos;Истинный: {row["true_label"]}, Предсказан: {row["pred_label"]}\n&apos;)

    f.write(&apos;\n\n=== ПРИМЕРЫ FALSE NEGATIVES ===\n&apos;)
    for idx, row in fn.head(5).iterrows():
        f.write(f&apos;\nТекст: {row["text"]}\n&apos;)
        f.write(f&apos;Истинный: {row["true_label"]}, Предсказан: {row["pred_label"]}\n&apos;)

    # Добавьте ваши наблюдения
    f.write(&apos;\n\n=== НАБЛЮДЕНИЯ ===\n&apos;)
    f.write(&apos;Добавьте сюда ваши выводы о паттернах ошибок...\n&apos;)

ЗАДАЧА 3: Создание демо-приложения (Gradio)

1. Установите Gradio:
pip install gradio

2. Создайте файл app.py:
import gradio as gr

def predict_sentiment(text):
    """
    Функция для Gradio интерфейса
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model_ft(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()

    label_map = {0: &apos;Negative&apos;, 1: &apos;Neutral&apos;, 2: &apos;Positive&apos;}

    # Форматируем вывод вероятностей
    result = f"Prediction: {label_map.get(pred, pred)}\n\n"
    result += "Probabilities:\n"

      for i, prob in enumerate(probs[0]):
        result += f"{label_map.get(i, i)}: {prob*100:.2f}%\n"

    return result

# Создаём интерфейс
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Введите текст для анализа..."),
    outputs=gr.Textbox(label="Результат"),
    title="Sentiment Analysis с BERT",
    description="Введите текст и модель определит его тональность"
)

if __name__ == "__main__":
    demo.launch()

3. Запустите приложение:
python app.py

4. Откройте браузер по адресу http://127.0.0.1:7860

ЗАДАЧА 4: Альтернатива — FastAPI (опционально)

Если предпочитаете FastAPI вместо Gradio:

1. Установите зависимости:
pip install fastapi uvicorn

2. Создайте api.py:
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictionRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: PredictionRequest):
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model_ft(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()

    label_map = {0: &apos;Negative&apos;, 1: &apos;Neutral&apos;, 2: &apos;Positive&apos;}

    return {
        "prediction": label_map.get(pred, str(pred)),
        "probabilities": {
            label_map.get(i, str(i)): float(probs[0][i])
            for i in range(len(probs[0]))
        }
    }

3. Запустите:
uvicorn api:app --reload

ЗАДАЧА 5: README документация

Создайте README.md с описанием проекта:

# Sentiment Analysis с BERT

## Описание
Проект по анализу тональности текстов с использованием трансформеров.

## Структура
- `fine_tuned_model/` — обученная модель
- `app.py` — Gradio приложение
- `error_analysis.txt` — анализ ошибок
- `comparison_results.txt` — сравнение моделей

## Запуск демо

### Вариант с Gradio:
```bash
pip install gradio transformers torch
python app.py
```
Откройте http://127.0.0.1:7860

### Вариант с FastAPI:
```bash
pip install fastapi uvicorn transformers torch
uvicorn api:app --reload
```
API будет доступен на http://127.0.0.1:8000

## Результаты

### Fine-tuned модель:
- F1 (macro): X.XX
- Accuracy: X.XX

### Baseline модель:
- F1 (macro): X.XX
- Accuracy: X.XX

Улучшение: XX%

## Использование в коде
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(&apos;./fine_tuned_model&apos;)
tokenizer = AutoTokenizer.from_pretrained(&apos;./fine_tuned_model&apos;)

# Предсказание
inputs = tokenizer("Your text here", return_tensors="pt")
outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=1)
```

## Требования
- Python 3.8+
- transformers
- torch
- gradio (для демо)


ЧЕКПОИНТ:
К концу дня вы должны иметь:
• Файл error_analysis.txt с анализом ошибок
• Работающее демо-приложение (Gradio или FastAPI)
• README.md с документацией
• Полный проект на 7 дней

ПОЗДРАВЛЯЕМ! Вы завершили недельный проект по изучению трансформеров!