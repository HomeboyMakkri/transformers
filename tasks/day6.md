# День 6 — Инференс и сравнение моделей

> Идентификатор: `transformers_day06`

## Цель дня

Сравнить baseline-модель из Дня 4 и fine-tuned модель из Дня 5.

Используйте код и модели предыдущих дней.

## Задача 1 — Загрузка моделей

1. Загрузите fine-tuned модель:

   ```python
   import torch
   from transformers import AutoModelForSequenceClassification, AutoTokenizer

   model_ft = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_model")
   tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
   model_ft.eval()
   ```

2. Загрузите baseline-модель из Дня 4:

   ```python
   import joblib

   # Если вы сохранили baseline-модель.
   baseline_model = joblib.load("baseline_model.pkl")
   baseline_vectorizer = joblib.load("vectorizer.pkl")
   ```

## Задача 2 — Предсказание fine-tuned модели

Напишите функцию для предсказания:

```python
def predict_fine_tuned(texts, model, tokenizer):
    if isinstance(texts, str):
        texts = [texts]

    predictions = []

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

        predictions.append(
            {
                "text": text,
                "prediction": pred,
                "probabilities": probs[0].cpu().numpy(),
            }
        )

    return predictions
```

## Задача 3 — Предсказание baseline-модели

```python
def predict_baseline(texts, model, vectorizer, clean_func=None):
    if isinstance(texts, str):
        texts = [texts]

    if clean_func:
        texts = [clean_func(t) for t in texts]

    X = vectorizer.transform(texts)
    predictions = model.predict(X)
    probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    results = []
    for i, text in enumerate(texts):
        results.append(
            {
                "text": text,
                "prediction": int(predictions[i]),
                "probabilities": probs[i] if probs is not None else None,
            }
        )

    return results
```

## Задача 4 — Сравнение на примерах

1. Создайте тестовые примеры:

   ```python
   test_texts = [
       "This movie was absolutely fantastic!",
       "Terrible, waste of my time.",
       "It was okay, nothing special.",
       "Best film I've seen this year!",
       "Boring and too long.",
   ]
   ```

2. Получите предсказания обеих моделей:

   ```python
   # Fine-tuned predictions.
   preds_ft = predict_fine_tuned(test_texts, model_ft, tokenizer)

   # Baseline predictions, если модель существует.
   if baseline_model:
       preds_baseline = predict_baseline(
           test_texts,
           baseline_model,
           baseline_vectorizer,
       )
   ```

3. Выведите сравнение:

   ```python
   for i, text in enumerate(test_texts):
       print(f"\nТекст: {text}")
       print(
           f"Fine-tuned: {preds_ft[i]['prediction']} "
           f"(probs: {preds_ft[i]['probabilities']})"
       )

       if baseline_model:
           print(f"Baseline: {preds_baseline[i]['prediction']}")
           print(
               "Совпадают: "
               f"{preds_ft[i]['prediction'] == preds_baseline[i]['prediction']}"
           )
   ```

## Задача 5 — Confusion matrix для обеих моделей

1. Загрузите тестовые данные:

   ```python
   import pandas as pd
   from sklearn.model_selection import train_test_split

   df = pd.read_csv("your_dataset.csv")
   _, test_df, _, _ = train_test_split(
       df,
       df["label"],
       test_size=0.2,
       random_state=42,
       stratify=df["label"],
   )

   test_texts = test_df["text"].tolist()
   test_labels = test_df["label"].tolist()
   ```

2. Получите предсказания fine-tuned модели:

   ```python
   preds_ft_all = predict_fine_tuned(test_texts, model_ft, tokenizer)
   y_pred_ft = [p["prediction"] for p in preds_ft_all]
   ```

3. Постройте confusion matrix:

   ```python
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.metrics import confusion_matrix

   cm_ft = confusion_matrix(test_labels, y_pred_ft)

   plt.figure(figsize=(8, 6))
   sns.heatmap(cm_ft, annot=True, fmt="d", cmap="Blues")
   plt.title("Confusion Matrix - Fine-tuned Model")
   plt.ylabel("True Label")
   plt.xlabel("Predicted Label")
   plt.savefig("confusion_matrix_finetuned.png")
   ```

## Задача 6 — Сравнение метрик

Посчитайте метрики обеих моделей:

```python
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Fine-tuned model.
print("=== Fine-tuned Model ===")
print(classification_report(test_labels, y_pred_ft))
f1_ft = f1_score(test_labels, y_pred_ft, average="macro")
acc_ft = accuracy_score(test_labels, y_pred_ft)

# Baseline model, если она существует.
if baseline_model:
    preds_baseline_all = predict_baseline(
        test_texts,
        baseline_model,
        baseline_vectorizer,
    )
    y_pred_base = [p["prediction"] for p in preds_baseline_all]

    print("\n=== Baseline Model ===")
    print(classification_report(test_labels, y_pred_base))
    f1_base = f1_score(test_labels, y_pred_base, average="macro")
    acc_base = accuracy_score(test_labels, y_pred_base)

    print("\n=== Сравнение ===")
    print(f"Fine-tuned F1: {f1_ft:.4f}, Accuracy: {acc_ft:.4f}")
    print(f"Baseline F1: {f1_base:.4f}, Accuracy: {acc_base:.4f}")
    print(f"Улучшение F1: {(f1_ft - f1_base) / f1_base * 100:.2f}%")
```

## Задача 7 — Сохранение результатов сравнения

```python
with open("comparison_results.txt", "w") as f:
    f.write("=== Сравнение моделей ===\n\n")
    f.write("Fine-tuned Model:\n")
    f.write(f"  F1 (macro): {f1_ft:.4f}\n")
    f.write(f"  Accuracy: {acc_ft:.4f}\n")

    if baseline_model:
        f.write("\nBaseline Model:\n")
        f.write(f"  F1 (macro): {f1_base:.4f}\n")
        f.write(f"  Accuracy: {acc_base:.4f}\n")
        f.write(f"\nУлучшение: {(f1_ft - f1_base) / f1_base * 100:.2f}%\n")
```

## Чекпоинт

К концу дня у вас должны быть:

- [ ] функции `predict_fine_tuned` и `predict_baseline`;
- [ ] confusion matrix для fine-tuned модели;
- [ ] сравнение метрик моделей;
- [ ] файл `comparison_results.txt` с результатами.

> В День 7 вы проведёте детальный анализ ошибок.
