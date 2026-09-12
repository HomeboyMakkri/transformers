📝 День 2: transformers_day02

День 2 — Получение эмбеддингов (hidden states)

ЦЕЛЬ ДНЯ:
Научиться извлекать скрытые состояния (hidden states) из трансформерной модели.

Используйте код из Дня 1 (функцию tokenize_texts).

ЗАДАЧА 1: Загрузка модели

1. Импортируйте AutoModel:
from transformers import AutoModel

2. Загрузите модель с тем же именем, что и токенизатор:
model = AutoModel.from_pretrained(model_name)

3. Переведите в режим оценки:
model.eval()

4. Посмотрите на архитектуру:
print(model)

ЗАДАЧА 2: Получение hidden states для одного текста

1. Возьмите текст и токенизируйте его:
text = "This movie was absolutely amazing!"
tokens = tokenizer(text, return_tensors="pt")

2. Прогоните через модель без градиентов:
with torch.no_grad():
    outputs = model(**tokens)

3. Изучите структуру outputs:
print(type(outputs))
print(outputs.last_hidden_state.shape)

Форма будет [batch_size, sequence_length, hidden_size]

4. Извлеките CLS-токен (первый токен):
cls_embedding = outputs.last_hidden_state[:, 0, :]
print(f&apos;CLS embedding shape: {cls_embedding.shape}&apos;)
print(f&apos;CLS embedding: {cls_embedding[0][:5]}...&apos;)  # первые 5 значений

ЗАДАЧА 3: Функция для получения эмбеддингов

Напишите функцию, которая принимает тексты и возвращает эмбеддинги:

def get_embeddings(texts, tokenizer, model, batch_size=32):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        # Токенизируем (используем функцию из Дня 1)
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        # Получаем hidden states
        with torch.no_grad():
            outputs = model(**tokens)

        # Извлекаем CLS-токены
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())

    # Объединяем все батчи
    import numpy as np
    return np.vstack(all_embeddings)

ЗАДАЧА 4: Тестирование на нескольких текстах

1. Создайте тестовый список:
texts = [
    "This movie was absolutely amazing!",
    "Terrible movie, waste of time.",
    "Pretty good, I liked it.",
    "Boring and too long."
]

2. Получите эмбеддинги:
embeddings = get_embeddings(texts, tokenizer, model)
print(f&apos;Embeddings shape: {embeddings.shape}&apos;)
print(f&apos;Ожидается: (4, 768) для DistilBERT&apos;)

ЗАДАЧА 5: Сходство текстов

1. Напишите функцию для вычисления косинусного сходства:
from sklearn.metrics.pairwise import cosine_similarity

def similarity(text1, text2, tokenizer, model):
    emb = get_embeddings([text1, text2], tokenizer, model)
    sim = cosine_similarity(emb[0:1], emb[1:2])[0][0]
    return sim

2. Проверьте сходство похожих текстов:
sim1 = similarity("Great movie!", "Amazing film!", tokenizer, model)
sim2 = similarity("Great movie!", "Terrible film!", tokenizer, model)

print(f&apos;Сходство похожих: {sim1:.3f}&apos;)
print(f&apos;Сходство разных: {sim2:.3f}&apos;)

ЧЕКПОИНТ:
К концу дня вы должны иметь:
• Загруженную модель AutoModel
• Функцию get_embeddings для получения эмбеддингов
• Понимание что такое hidden states и CLS-токен
• Функцию для вычисления сходства текстов

Этот код понадобится в День 3 для визуализации и в День 4 для классификации.