# День 7 — Анализ ошибок и мини-приложение

> Идентификатор: `transformers_day07`

## Цель дня

Провести детальный анализ ошибок модели и создать простое демонстрационное приложение.

Используйте код и модели предыдущих дней.

## Задача 1 — Анализ False Positive и False Negative

Найдите ошибки модели:

```python
import pandas as pd

# Загружаем данные и предсказания из Дня 6.
df_test = pd.DataFrame(
    {
        "text": test_texts,
        "true_label": test_labels,
        "pred_label": y_pred_ft,
    }
)

# Находим ошибки.
errors = df_test[df_test["true_label"] != df_test["pred_label"]]

# False Positives: предсказала positive, а класс был negative/neutral.
fp = errors[(errors["pred_label"] == 1) & (errors["true_label"] == 0)]

# False Negatives: предсказала negative, а класс был positive.
fn = errors[(errors["pred_label"] == 0) & (errors["true_label"] == 1)]

print(f"Всего ошибок: {len(errors)}")
print(f"False Positives: {len(fp)}")
print(f"False Negatives: {len(fn)}")
```

## Задача 2 — Анализ паттернов ошибок

1. Выведите примеры ошибок и найдите закономерности:

   ```python
   print("\n=== FALSE POSITIVES (сказали good, а было bad) ===")
   for idx, row in fp.head(5).iterrows():
       print(f"\nТекст: {row['text'][:100]}...")
       print(
           f"Истинный класс: {row['true_label']}, "
           f"предсказан: {row['pred_label']}"
       )

   print("\n=== FALSE NEGATIVES (сказали bad, а было good) ===")
   for idx, row in fn.head(5).iterrows():
       print(f"\nТекст: {row['text'][:100]}...")
       print(
           f"Истинный класс: {row['true_label']}, "
           f"предсказан: {row['pred_label']}"
       )
   ```

2. Проанализируйте длину ошибочных текстов:

   ```python
   errors["text_length"] = errors["text"].str.len()
   print(f"\nСредняя длина ошибочных текстов: {errors['text_length'].mean():.0f}")
   print(f"Средняя длина всех текстов: {df_test['text'].str.len().mean():.0f}")
   ```

3. Сохраните анализ в файл:

   ```python
   with open("error_analysis.txt", "w") as f:
       f.write("=== АНАЛИЗ ОШИБОК ===\n\n")
       f.write(f"Всего ошибок: {len(errors)}\n")
       f.write(f"False Positives: {len(fp)}\n")
       f.write(f"False Negatives: {len(fn)}\n\n")

       f.write("=== ПРИМЕРЫ FALSE POSITIVES ===\n")
       for idx, row in fp.head(5).iterrows():
           f.write(f"\nТекст: {row['text']}\n")
           f.write(
               f"Истинный: {row['true_label']}, "
               f"предсказан: {row['pred_label']}\n"
           )

       f.write("\n\n=== ПРИМЕРЫ FALSE NEGATIVES ===\n")
       for idx, row in fn.head(5).iterrows():
           f.write(f"\nТекст: {row['text']}\n")
           f.write(
               f"Истинный: {row['true_label']}, "
               f"предсказан: {row['pred_label']}\n"
           )

       # Добавьте свои наблюдения.
       f.write("\n\n=== НАБЛЮДЕНИЯ ===\n")
       f.write("Добавьте сюда ваши выводы о паттернах ошибок...\n")
   ```

## Задача 3 — Демо-приложение на Gradio

1. Установите Gradio:

   ```bash
   pip install gradio
   ```

2. Создайте файл `app.py`:

   ```python
   import gradio as gr


   def predict_sentiment(text):
       """Функция для интерфейса Gradio."""
       inputs = tokenizer(
           text,
           return_tensors="pt",
           truncation=True,
           max_length=128,
       )

       with torch.no_grad():
           outputs = model_ft(**inputs)

       probs = torch.nn.functional.softmax(outputs.logits, dim=1)
       pred = torch.argmax(probs, dim=1).item()

       label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

       result = f"Prediction: {label_map.get(pred, pred)}\n\n"
       result += "Probabilities:\n"

       for i, prob in enumerate(probs[0]):
           result += f"{label_map.get(i, i)}: {prob * 100:.2f}%\n"

       return result


   demo = gr.Interface(
       fn=predict_sentiment,
       inputs=gr.Textbox(lines=3, placeholder="Введите текст для анализа..."),
       outputs=gr.Textbox(label="Результат"),
       title="Sentiment Analysis с BERT",
       description="Введите текст, и модель определит его тональность",
   )

   if __name__ == "__main__":
       demo.launch()
   ```

3. Запустите приложение:

   ```bash
   python app.py
   ```

4. Откройте в браузере <http://127.0.0.1:7860>.

## Задача 4 — Альтернатива на FastAPI (опционально)

1. Установите зависимости:

   ```bash
   pip install fastapi uvicorn
   ```

2. Создайте файл `api.py`:

   ```python
   from fastapi import FastAPI
   from pydantic import BaseModel

   app = FastAPI()


   class PredictionRequest(BaseModel):
       text: str


   @app.post("/predict")
   def predict(request: PredictionRequest):
       inputs = tokenizer(
           request.text,
           return_tensors="pt",
           truncation=True,
           max_length=128,
       )

       with torch.no_grad():
           outputs = model_ft(**inputs)

       probs = torch.nn.functional.softmax(outputs.logits, dim=1)
       pred = torch.argmax(probs, dim=1).item()

       label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

       return {
           "prediction": label_map.get(pred, str(pred)),
           "probabilities": {
               label_map.get(i, str(i)): float(probs[0][i])
               for i in range(len(probs[0]))
           },
       }
   ```

3. Запустите API:

   ```bash
   uvicorn api:app --reload
   ```

   API будет доступен по адресу <http://127.0.0.1:8000>.

## Задача 5 — README-документация

Создайте `README.md` с описанием проекта:

````markdown
# Sentiment Analysis с BERT

## Описание

Проект по анализу тональности текстов с использованием трансформеров.

## Структура

- `fine_tuned_model/` — обученная модель;
- `app.py` — Gradio-приложение;
- `error_analysis.txt` — анализ ошибок;
- `comparison_results.txt` — сравнение моделей.

## Запуск демо

### Вариант с Gradio

```bash
pip install gradio transformers torch
python app.py
```

Откройте <http://127.0.0.1:7860>.

### Вариант с FastAPI

```bash
pip install fastapi uvicorn transformers torch
uvicorn api:app --reload
```

API будет доступен по адресу <http://127.0.0.1:8000>.

## Результаты

### Fine-tuned модель

- F1 (macro): X.XX
- Accuracy: X.XX

### Baseline-модель

- F1 (macro): X.XX
- Accuracy: X.XX

Улучшение: XX%

## Использование в коде

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

inputs = tokenizer("Your text here", return_tensors="pt")
outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=1)
```

## Требования

- Python 3.8+
- transformers
- torch
- gradio — для демо
````

## Чекпоинт

К концу дня у вас должны быть:

- [ ] файл `error_analysis.txt` с анализом ошибок;
- [ ] работающее приложение на Gradio или FastAPI;
- [ ] `README.md` с документацией;
- [ ] полный проект за семь дней.

> Поздравляем! Вы завершили недельный проект по изучению трансформеров.
