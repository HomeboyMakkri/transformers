# День 3 — Attention-матрицы и визуализация

> Идентификатор: `transformers_day03`

## Цель дня

Понять, как работает механизм внимания в трансформерах, и научиться его визуализировать.

Используйте код из Дней 1 и 2.

## Задача 1 — Загрузка модели с attention

1. Загрузите модель с выводом attention:

   ```python
   from transformers import AutoModel

   model = AutoModel.from_pretrained(
       model_name,
       output_attentions=True,  # важно
   )
   model.eval()
   ```

2. Токенизируйте текст:

   ```python
   text = "The amazing movie won many awards"
   tokens = tokenizer(text, return_tensors="pt")
   ```

## Задача 2 — Получение весов attention

1. Пропустите данные через модель:

   ```python
   with torch.no_grad():
       outputs = model(**tokens)
   ```

2. Изучите attention:

   ```python
   print(type(outputs.attentions))
   print(f"Количество слоёв: {len(outputs.attentions)}")
   print(f"Форма attention для слоя 0: {outputs.attentions[0].shape}")
   ```

   Ожидаемая форма: `[batch_size, num_heads, seq_len, seq_len]`.

3. Извлеките attention из первого слоя:

   ```python
   attention = outputs.attentions[0]  # первый слой
   print(f"Attention shape: {attention.shape}")

   # Первый элемент батча, первая голова.
   attn_single = attention[0, 0]  # [seq_len, seq_len]
   print(f"Single head shape: {attn_single.shape}")
   ```

## Задача 3 — Визуализация attention

1. Установите `matplotlib` и `seaborn`:

   ```bash
   pip install matplotlib seaborn
   ```

2. Напишите функцию визуализации:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   import seaborn as sns


   def visualize_attention(tokens, attention, layer=0, head=0):
       """
       tokens: токенизированный текст
       attention: attention weights модели
       layer: номер слоя для визуализации
       head: номер головы для визуализации
       """
       # Получаем attention-матрицу [seq_len, seq_len].
       attn = attention[layer][0, head]

       # Получаем токены для подписей.
       token_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])

       # Рисуем heatmap.
       plt.figure(figsize=(10, 8))
       sns.heatmap(
           attn.cpu().numpy(),
           xticklabels=token_list,
           yticklabels=token_list,
           cmap="viridis",
           cbar=True,
       )
       plt.title(f"Attention - Layer {layer}, Head {head}")
       plt.xlabel("Keys")
       plt.ylabel("Queries")
       plt.tight_layout()
       plt.savefig(f"attention_layer{layer}_head{head}.png")
       plt.show()
   ```

3. Используйте функцию:

   ```python
   visualize_attention(tokens, outputs.attentions, layer=0, head=0)
   ```

## Задача 4 — Анализ attention на разных слоях

1. Визуализируйте attention из разных слоёв:

   ```python
   # Первый слой.
   visualize_attention(tokens, outputs.attentions, layer=0, head=0)

   # Средний слой.
   visualize_attention(tokens, outputs.attentions, layer=3, head=0)

   # Последний слой.
   visualize_attention(tokens, outputs.attentions, layer=5, head=0)
   ```

2. Сравните результаты: на последних слоях внимание обычно более сфокусировано.

## Задача 5 — Attention для разных голов

1. Визуализируйте разные головы одного слоя:

   ```python
   for head in range(8):  # DistilBERT имеет 8 голов
       visualize_attention(tokens, outputs.attentions, layer=0, head=head)
   ```

2. Проанализируйте результаты: разные головы фокусируются на разных связях.

## Задача 6 — Анализ внимания к ключевым словам

1. Возьмите текст с явно выраженным эмоционально окрашенным словом:

   ```python
   text = "This movie was absolutely terrible and I hated it"
   tokens = tokenizer(text, return_tensors="pt")

   with torch.no_grad():
       outputs = model(**tokens)

   visualize_attention(tokens, outputs.attentions, layer=5, head=0)
   ```

2. Посмотрите, на какое слово фокусируется внимание при обработке слова `terrible`.

## Чекпоинт

К концу дня у вас должны быть:

- [ ] понимание attention weights;
- [ ] функция `visualize_attention`;
- [ ] несколько графиков attention из разных слоёв и голов;
- [ ] понимание того, как внимание меняется по слоям.

> В День 4 вы будете использовать эмбеддинги для классификации.
