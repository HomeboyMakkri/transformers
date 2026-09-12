📝 День 6: transformers_day06

День 6 — Инференс и сравнение моделей

ЦЕЛЬ ДНЯ:
Сравнить baseline модель (День 4) и fine-tuned модель (День 5).

Используйте код и модели из предыдущих дней.

ЗАДАЧА 1: Загрузка моделей

1. Загрузите fine-tuned модель:
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_ft = AutoModelForSequenceClassification.from_pretrained(&apos;./fine_tuned_model&apos;)
tokenizer = AutoTokenizer.from_pretrained(&apos;./fine_tuned_model&apos;)
model_ft.eval()

2. Загрузите baseline модель (из Дня 4):
import joblib

# Если вы сохранили baseline модель
baseline_model = joblib.load(&apos;baseline_model.pkl&apos;)
baseline_vectorizer = joblib.load(&apos;vectorizer.pkl&apos;)

ЗАДАЧА 2: Функция предсказания для fine-tuned

Напишите функцию для предсказания:
def predict_fine_tuned(texts, model, tokenizer):
    if isinstance(texts, str):
        texts = [texts]

    predictions = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

        predictions.append({
            &apos;text&apos;: text,
            &apos;prediction&apos;: pred,
            &apos;probabilities&apos;: probs[0].cpu().numpy()
        })

    return predictions

ЗАДАЧА 3: Функция предсказания для baseline

def predict_baseline(texts, model, vectorizer, clean_func=None):
    if isinstance(texts, str):
        texts = [texts]

    if clean_func:
        texts = [clean_func(t) for t in texts]

    X = vectorizer.transform(texts)
    predictions = model.predict(X)
    probs = model.predict_proba(X) if hasattr(model, &apos;predict_proba&apos;) else None

    results = []
    for i, text in enumerate(texts):
        results.append({
            &apos;text&apos;: text,
            &apos;prediction&apos;: int(predictions[i]),
            &apos;probabilities&apos;: probs[i] if probs is not None else None
        })

    return results

ЗАДАЧА 4: Сравнение на примерах

1. Создайте тестовые примеры:
test_texts = [
    "This movie was absolutely fantastic!",
    "Terrible, waste of my time.",
    "It was okay, nothing special.",
    "Best film I&apos;ve seen this year!",
    "Boring and too long."
]

2. Получите предсказания от обеих моделей:
# Fine-tuned predictions
preds_ft = predict_fine_tuned(test_texts, model_ft, tokenizer)

# Baseline predictions (если есть)
if baseline_model:
    preds_baseline = predict_baseline(test_texts, baseline_model, baseline_vectorizer)

3. Выведите сравнение:
for i, text in enumerate(test_texts):
    print(f&apos;\nТекст: {text}&apos;)
    print(f&apos;Fine-tuned: {preds_ft[i]["prediction"]} (probs: {preds_ft[i]["probabilities"]})&apos;)
    if baseline_model:
        print(f&apos;Baseline: {preds_baseline[i]["prediction"]}&apos;)
        print(f&apos;Совпадают: {preds_ft[i]["prediction"] == preds_baseline[i]["prediction"]}&apos;)

ЗАДАЧА 5: Confusion Matrix для обеих моделей

1. Загрузите тестовые данные:
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(&apos;your_dataset.csv&apos;)
_, test_df, _, _ = train_test_split(df, df[&apos;label&apos;], test_size=0.2, random_state=42, stratify=df[&apos;label&apos;])

test_texts = test_df[&apos;text&apos;].tolist()
test_labels = test_df[&apos;label&apos;].tolist()

2. Получите предсказания fine-tuned:
preds_ft_all = predict_fine_tuned(test_texts, model_ft, tokenizer)
y_pred_ft = [p[&apos;prediction&apos;] for p in preds_ft_all]
3. Постройте confusion matrix:
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm_ft = confusion_matrix(test_labels, y_pred_ft)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_ft, annot=True, fmt=&apos;d&apos;, cmap=&apos;Blues&apos;)
plt.title(&apos;Confusion Matrix - Fine-tuned Model&apos;)
plt.ylabel(&apos;True Label&apos;)
plt.xlabel(&apos;Predicted Label&apos;)
plt.savefig(&apos;confusion_matrix_finetuned.png&apos;)

ЗАДАЧА 6: Сравнение метрик

1. Посчитайте метрики для обеих моделей:
from sklearn.metrics import classification_report, f1_score, accuracy_score

# Fine-tuned
print("=== Fine-tuned Model ===")
print(classification_report(test_labels, y_pred_ft))
f1_ft = f1_score(test_labels, y_pred_ft, average=&apos;macro&apos;)
acc_ft = accuracy_score(test_labels, y_pred_ft)

# Baseline (если есть)
if baseline_model:
    preds_baseline_all = predict_baseline(test_texts, baseline_model, baseline_vectorizer)
    y_pred_base = [p[&apos;prediction&apos;] for p in preds_baseline_all]

    print("\n=== Baseline Model ===")
    print(classification_report(test_labels, y_pred_base))
    f1_base = f1_score(test_labels, y_pred_base, average=&apos;macro&apos;)
    acc_base = accuracy_score(test_labels, y_pred_base)

    print(f&apos;\n=== Сравнение ===&apos;)
    print(f&apos;Fine-tuned F1: {f1_ft:.4f}, Accuracy: {acc_ft:.4f}&apos;)
    print(f&apos;Baseline F1: {f1_base:.4f}, Accuracy: {acc_base:.4f}&apos;)
    print(f&apos;Улучшение F1: {(f1_ft - f1_base) / f1_base * 100:.2f}%&apos;)

ЗАДАЧА 7: Сохранение результатов сравнения

with open(&apos;comparison_results.txt&apos;, &apos;w&apos;) as f:
    f.write(&apos;=== Сравнение моделей ===\n\n&apos;)
    f.write(f&apos;Fine-tuned Model:\n&apos;)
    f.write(f&apos;  F1 (macro): {f1_ft:.4f}\n&apos;)
    f.write(f&apos;  Accuracy: {acc_ft:.4f}\n&apos;)

    if baseline_model:
        f.write(f&apos;\nBaseline Model:\n&apos;)
        f.write(f&apos;  F1 (macro): {f1_base:.4f}\n&apos;)
        f.write(f&apos;  Accuracy: {acc_base:.4f}\n&apos;)
        f.write(f&apos;\nУлучшение: {(f1_ft - f1_base) / f1_base * 100:.2f}%\n&apos;)

ЧЕКПОИНТ:
К концу дня вы должны иметь:
• Функции predict_fine_tuned и predict_baseline
• Confusion matrix для fine-tuned модели
• Сравнение метрик между моделями
• Файл comparison_results.txt с результатами

В День 7 вы проведёте детальный анализ ошибок.