# День 4 — Baseline без обучения трансформера

> Идентификатор: `transformers_day04`

## Задача 1 — Токенизация текстов

1. Импортируйте `AutoTokenizer` из библиотеки `transformers`.
2. Загрузите токенизатор модели `distilbert-base-uncased` или `bert-base-uncased`.
3. Напишите функцию `tokenize_texts(texts, max_length=128)`, которая:
   - принимает список текстов и максимальную длину;
   - вызывает `tokenizer` с параметрами `padding=True`, `truncation=True`, `max_length=max_length`, `return_tensors="pt"`;
   - возвращает токенизированный батч.
4. Протестируйте функцию на двух-трёх примерах текстов и выведите результат.

## Задача 2 — Извлечение CLS-эмбеддингов

1. Импортируйте `AutoModel` из `transformers` и библиотеку `torch`.
2. Загрузите предобученную модель через `AutoModel.from_pretrained()` с тем же именем, что и токенизатор.
3. Переведите модель в режим оценки:

   ```python
   model.eval()
   ```

4. Напишите функцию `get_cls_embeddings(texts, batch_size=32)`, которая:
   - принимает список текстов и размер батча;
   - разбивает тексты на батчи по `batch_size` элементов;
   - для каждого батча:
     - токенизирует тексты через функцию из Задачи 1;
     - пропускает данные через модель внутри `with torch.no_grad():`;
     - извлекает `outputs.last_hidden_state[:, 0, :]` — CLS-токен;
     - переводит результат в NumPy через `.cpu().numpy()` и добавляет его в список;
   - объединяет батчи через `np.vstack()` и возвращает результат.
5. Протестируйте функцию на небольшом списке текстов и проверьте размерность результата.

## Задача 3 — Logistic Regression на эмбеддингах

1. Загрузите датасет в `DataFrame` с колонками текста и метки класса.
2. Извлеките список текстов и список меток.
3. Получите эмбеддинги всех текстов через `get_cls_embeddings(texts)`.
4. Импортируйте `train_test_split` из `sklearn` и разделите эмбеддинги и метки на train/test в пропорции 80/20 с параметрами `stratify=y` и `random_state=42`.
5. Импортируйте `LogisticRegression`:

   ```python
   from sklearn.linear_model import LogisticRegression
   ```

6. Создайте модель:

   ```python
   model = LogisticRegression(max_iter=1000, n_jobs=-1)
   ```

7. Обучите её на train-выборке:

   ```python
   model.fit(X_train, y_train)
   ```

8. Сделайте предсказания на test-выборке:

   ```python
   y_pred = model.predict(X_test)
   ```

9. Импортируйте `classification_report` и `f1_score` из `sklearn.metrics`.
10. Выведите классификационный отчёт:

    ```python
    print(classification_report(y_test, y_pred))
    ```

11. Посчитайте и выведите macro F1:

    ```python
    f1 = f1_score(y_test, y_pred, average="macro")
    ```

12. Сохраните значение macro F1 в файл `baseline_results.txt`.
