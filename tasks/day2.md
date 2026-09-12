# День 2 — Получение эмбеддингов (hidden states)

> Идентификатор: `transformers_day02`

## Цель дня

Научиться извлекать скрытые состояния (hidden states) из трансформерной модели.

Используйте функцию `tokenize_texts` из Дня 1.

## Задача 1 — Загрузка модели

1. Импортируйте `AutoModel`:

   ```python
   from transformers import AutoModel
   ```

2. Загрузите модель с тем же именем, что и токенизатор:

   ```python
   model = AutoModel.from_pretrained(model_name)
   ```

3. Переведите модель в режим оценки:

   ```python
   model.eval()
   ```

4. Посмотрите на архитектуру:

   ```python
   print(model)
   ```

## Задача 2 — Получение hidden states для одного текста

1. Возьмите текст и токенизируйте его:

   ```python
   text = "This movie was absolutely amazing!"
   tokens = tokenizer(text, return_tensors="pt")
   ```

2. Пропустите данные через модель без вычисления градиентов:

   ```python
   with torch.no_grad():
       outputs = model(**tokens)
   ```

3. Изучите структуру `outputs`:

   ```python
   print(type(outputs))
   print(outputs.last_hidden_state.shape)
   ```

   Ожидаемая форма: `[batch_size, sequence_length, hidden_size]`.

4. Извлеките CLS-токен — первый токен последовательности:

   ```python
   cls_embedding = outputs.last_hidden_state[:, 0, :]
   print(f"CLS embedding shape: {cls_embedding.shape}")
   print(f"CLS embedding: {cls_embedding[0][:5]}...")  # первые 5 значений
   ```

## Задача 3 — Функция получения эмбеддингов

Напишите функцию, которая принимает тексты и возвращает эмбеддинги:

```python
import numpy as np


def get_embeddings(texts, tokenizer, model, batch_size=32):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        # Токенизируем тексты, используя подход из Дня 1.
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        # Получаем hidden states.
        with torch.no_grad():
            outputs = model(**tokens)

        # Извлекаем CLS-токены.
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())

    # Объединяем все батчи.
    return np.vstack(all_embeddings)
```

## Задача 4 — Тестирование на нескольких текстах

1. Создайте тестовый список:

   ```python
   texts = [
       "This movie was absolutely amazing!",
       "Terrible movie, waste of time.",
       "Pretty good, I liked it.",
       "Boring and too long.",
   ]
   ```

2. Получите эмбеддинги:

   ```python
   embeddings = get_embeddings(texts, tokenizer, model)
   print(f"Embeddings shape: {embeddings.shape}")
   print("Ожидается: (4, 768) для DistilBERT")
   ```

## Задача 5 — Сходство текстов

1. Напишите функцию для вычисления косинусного сходства:

   ```python
   from sklearn.metrics.pairwise import cosine_similarity


   def similarity(text1, text2, tokenizer, model):
       emb = get_embeddings([text1, text2], tokenizer, model)
       sim = cosine_similarity(emb[0:1], emb[1:2])[0][0]
       return sim
   ```

2. Проверьте сходство похожих и разных текстов:

   ```python
   sim1 = similarity("Great movie!", "Amazing film!", tokenizer, model)
   sim2 = similarity("Great movie!", "Terrible film!", tokenizer, model)

   print(f"Сходство похожих: {sim1:.3f}")
   print(f"Сходство разных: {sim2:.3f}")
   ```

## Чекпоинт

К концу дня у вас должны быть:

- [ ] загруженная модель `AutoModel`;
- [ ] функция `get_embeddings` для получения эмбеддингов;
- [ ] понимание hidden states и CLS-токена;
- [ ] функция для вычисления сходства текстов.

> Этот код понадобится в День 3 для визуализации и в День 4 для классификации.
