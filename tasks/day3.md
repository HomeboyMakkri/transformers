📝 День 3: transformers_day03

День 3 — Attention-матрицы и визуализация

ЦЕЛЬ ДНЯ:
Понять как работает механизм внимания в трансформерах и научиться его визуализировать.

Используйте код из Дня 1 и День 2.

ЗАДАЧА 1: Загрузка модели с attention

1. Для получения attention нужно использовать специальный вывод:
from transformers import AutoModel

model = AutoModel.from_pretrained(
    model_name,
    output_attentions=True  # важно!
)
model.eval()

2. Токенизируйте текст:
text = "The amazing movie won many awards"
tokens = tokenizer(text, return_tensors="pt")

ЗАДАЧА 2: Получение attention весов

1. Прогоните через модель:
with torch.no_grad():
    outputs = model(**tokens)

2. Изучите attention:
print(type(outputs.attentions))
print(f&apos;Количество слоёв: {len(outputs.attentions)}&apos;)
print(f&apos;Форма attention для слоя 0: {outputs.attentions[0].shape}&apos;)

Форма: [batch_size, num_heads, seq_len, seq_len]

3. Извлеките attention из первого слоя:
attention = outputs.attentions[0]  # первый слой
print(f&apos;Attention shape: {attention.shape}&apos;)

# Для первого батча, первой головы
attn_single = attention[0, 0]  # [seq_len, seq_len]
print(f&apos;Single head shape: {attn_single.shape}&apos;)

ЗАДАЧА 3: Визуализация attention

1. Установите matplotlib и seaborn:
pip install matplotlib seaborn

2. Напишите функцию для визуализации:
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_attention(tokens, attention, layer=0, head=0):
    """
    tokens: токенизированный текст
    attention: attention weights от модели
    layer: номер слоя для визуализации
    head: номер головы для визуализации
    """
    # Получаем attention матрицу
    attn = attention[layer][0, head]  # [seq_len, seq_len]

    # Получаем токены для подписей
    token_list = tokenizer.convert_ids_to_tokens(tokens[&apos;input_ids&apos;][0])

    # Рисуем heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn.cpu().numpy(),
        xticklabels=token_list,
        yticklabels=token_list,
        cmap=&apos;viridis&apos;,
        cbar=True
    )
    plt.title(f&apos;Attention - Layer {layer}, Head {head}&apos;)
    plt.xlabel(&apos;Keys&apos;)
    plt.ylabel(&apos;Queries&apos;)
    plt.tight_layout()
    plt.savefig(f&apos;attention_layer{layer}_head{head}.png&apos;)
    plt.show()

# Используйте:
visualize_attention(tokens, outputs.attentions, layer=0, head=0)

ЗАДАЧА 4: Анализ attention для разных слоёв

1. Визуализируйте attention из разных слоёв:
# Первый слой
visualize_attention(tokens, outputs.attentions, layer=0, head=0)

# Средний слой
visualize_attention(tokens, outputs.attentions, layer=3, head=0)

# Последний слой
visualize_attention(tokens, outputs.attentions, layer=5, head=0)

2. Сравните — на последних слоях внимание обычно более сфокусировано

ЗАДАЧА 5: Attention для разных голов

1. Визуализируйте разные головы одного слоя:
for head in range(8):  # DistilBERT имеет 8 голов
    visualize_attention(tokens, outputs.attentions, layer=0, head=head)

2. Проанализируйте — разные головы фокусируются на разных связях

ЗАДАЧА 6: Анализ внимания к ключевым словам

1. Возьмите текст с явно выраженным sentimental словом:
text = "This movie was absolutely terrible and I hated it"
tokens = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**tokens)

# Визуализируйте
visualize_attention(tokens, outputs.attentions, layer=5, head=0)

2. Посмотрите — на какое слово фокусируется внимание при обработке "terrible"

ЧЕКПОИНТ:
К концу дня вы должны иметь:
• Понимание что такое attention weights
• Функцию visualize_attention для визуализации
• Несколько графиков attention из разных слоёв/голов
• Понимание как внимание меняется по слоям

В День 4 вы будете использовать эмбеддинги для классификации.