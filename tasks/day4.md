📝 День 4: transformers_day04

День 4 — Baseline без обучения трансформера

ЗАДАЧА 1: Токенизация текстов

1. Импортируйте AutoTokenizer из библиотеки transformers

2. Загрузите токенизатор модели "distilbert-base-uncased" или "bert-base-uncased"

3. Напишите функцию tokenize_texts(texts, max_length=128), которая:
   - Принимает список текстов и максимальную длину
   - Вызывает tokenizer с параметрами: padding=True, truncation=True, max_length=max_length, return_tensors="pt"
   - Возвращает токенизированный батч

4. Протестируйте функцию на 2-3 примерах текстов и выведите результат

ЗАДАЧА 2: Извлечение CLS-эмбеддингов

1. Импортируйте AutoModel из transformers и torch

2. Загрузите предобученную модель через AutoModel.from_pretrained() с тем же именем, что и токенизатор

3. Переведите модель в режим eval: model.eval()

4. Напишите функцию get_cls_embeddings(texts, batch_size=32), которая:
   - Принимает список текстов и размер батча
   - Разбивает тексты на батчи по batch_size штук
   - Для каждого батча:
     • Токенизирует через вашу функцию из задачи 1
     • Прогоняет через модель (оберните в with torch.no_grad():)
     • Извлекает outputs.last_hidden_state[:, 0, :] — это CLS-токен
     • Переводит в numpy (.cpu().numpy()) и добавляет в список
   - Склеивает все батчи через np.vstack() и возвращает

5. Протестируйте функцию на небольшом списке текстов, проверьте размерность вывода

ЗАДАЧА 3: Logistic Regression на эмбеддингах

1. Загрузите ваш датасет — это должен быть DataFrame с колонками для текста и метки класса

2. Извлеките список текстов и список меток

3. Получите эмбеддинги для всех текстов, вызвав get_cls_embeddings(texts)

4. Импортируйте train_test_split из sklearn и разделите эмбеддинги и метки на train/test в пропорции 80/20 с параметром stratify=y и random_state=42

5. Импортируйте LogisticRegression из sklearn.linear_model

6. Создайте модель: LogisticRegression(max_iter=1000, n_jobs=-1)

7. Обучите её на train выборке: model.fit(X_train, y_train)

8. Сделайте предсказания на test выборке: y_pred = model.predict(X_test)

9. Импортируйте classification_report и f1_score из sklearn.metrics

10. Выведите классификационный отчет: print(classification_report(y_test, y_pred))

11. Посчитайте macro F1: f1 = f1_score(y_test, y_pred, average=&apos;macro&apos;) и выведите её

12. Сохраните результаты в файл baseline_results.txt с указанием macro F1