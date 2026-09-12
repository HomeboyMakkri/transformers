# День 1 — Архитектура трансформеров и токенизация

> Идентификатор: `transformers_day01`

## Цель дня

Понять, как работают трансформеры, и научиться превращать текст в токены.

## Задача 1 — Установка библиотек

1. Установите `transformers` и `torch`:

   ```bash
   pip install transformers torch
   ```

2. Создайте новый Python-файл или Jupyter Notebook.

3. Импортируйте необходимые библиотеки:

   ```python
   from transformers import AutoTokenizer
   import torch
   ```

## Задача 2 — Загрузка токенизатора

1. Выберите модель.

   Для английского языка:

   ```python
   model_name = "distilbert-base-uncased"
   ```

   Для русского языка:

   ```python
   model_name = "distilbert-base-multilingual-cased"
   ```

2. Загрузите токенизатор:

   ```python
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   ```

3. Посмотрите на его параметры:

   ```python
   print(tokenizer.vocab_size)
   print(tokenizer.model_max_length)
   ```

## Задача 3 — Токенизация текста

1. Возьмите пример текста:

   ```python
   text = "This movie was absolutely amazing!"
   ```

2. Токенизируйте его:

   ```python
   tokens = tokenizer(text)
   print(tokens)
   ```

3. Посмотрите на идентификаторы токенов:

   ```python
   input_ids = tokens["input_ids"]
   print(f"Количество токенов: {len(input_ids)}")
   ```

4. Декодируйте токены обратно в текст:

   ```python
   decoded = tokenizer.decode(input_ids)
   print(f"Декодировано: {decoded}")
   ```

## Задача 4 — Работа с батчами

1. Напишите функцию для токенизации списка текстов:

   ```python
   def tokenize_texts(texts, max_length=128):
       return tokenizer(
           texts,
           padding=True,
           truncation=True,
           max_length=max_length,
           return_tensors="pt",
       )
   ```

2. Протестируйте её на нескольких текстах:

   ```python
   texts = [
       "This movie was great!",
       "Terrible movie, waste of time.",
   ]

   tokens = tokenize_texts(texts)
   print(f"Shape: {tokens['input_ids'].shape}")
   ```

3. Посмотрите на `attention_mask`:

   ```python
   print(f"Attention mask:\n{tokens['attention_mask']}")
   ```

## Задача 5 — Специальные токены

1. Выведите специальные токены:

   ```python
   print(f"CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
   print(f"SEP token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
   print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
   ```

2. Посмотрите, как выглядит токенизированный текст:

   ```python
   single = tokenizer(text, return_tensors="pt")
   print(f"Input IDs: {single['input_ids']}")
   print(f"Decoded: {tokenizer.decode(single['input_ids'][0])}")
   ```

## Задача 6 — Функция `explain_tokenization`

Напишите функцию, которая показывает, как текст разбился на токены:

```python
def explain_tokenization(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    print(f"Исходный текст: {text}")
    print(f"Токены: {tokens}")
    print(f"IDs: {ids}")
    print(f"Количество: {len(tokens)}")
```

Протестируйте функцию:

```python
explain_tokenization("Transformers are amazing!", tokenizer)
```

## Чекпоинт

К концу дня у вас должны быть:

- [ ] загруженный токенизатор;
- [ ] понимание того, как текст превращается в токены;
- [ ] функция для токенизации батчей текстов;
- [ ] функция `explain_tokenization`.

> Этот код понадобится в День 2 для работы с моделью.
