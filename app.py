import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Загрузите обученную модель и токенизатор
model = BertForSequenceClassification.from_pretrained('my_trained_model')
tokenizer = BertTokenizer.from_pretrained('my_trained_model')

# Убедитесь, что модель в режиме предсказания
model.eval()

# Список ценностей для декодирования меток
themes = [
    "Жизнь", "Искренность", "Вера", "Честность", "Честь", "Справедливость",
    "Надёжность", "Достаток", "Гуманизм", "Результат", "Альтруизм",
    "Любовь", "Комфорт", "Самореализация", "Радость", "Просветленность",
    "Красота", "Достоинство", "План", "Труд", "Надежда", "Истина",
    "Свобода", "Правда", "Всеединство", "Здоровье", "Совесть", "Человеческий капитал",
    "Творчество", "Нравственность", "Обязательность", "Концентрация",
    "Разумность", "Целостность", "Представление", "Онтичность",
    "Чувство Рода", "Возрождение", "Вдохновение", "Сущность",
    "Рассуждение", "Продуктивность", "Взаимопонимание", "Прекрасное",
    "Вечное", "Мудрость", "Верность", "Гений", "Понимание",
    "Гармония", "Наследование", "Сознание", "Ценность", "Негэнтропийность",
    "Счастье", "Тринитарность", "Взаимосодействие", "Уверенность",
    "Жизнеутверждение", "Милосердие", "Уникальность", "Сложностность",
    "Предвидение", "Благодарность", "Благодать", "Откровение"
]

# Функция для классификации текста
def classify_text(text):
    # Токенизация текста
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)

    # Получение предсказаний
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Аргмакс для одной ценности
        predicted_class_id = torch.argmax(logits, dim=1).item()
        predicted_value = themes[predicted_class_id]

        # Второй вариант: получение нескольких вероятных ценностей
        probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze()
        topk_probs, topk_indices = torch.topk(probabilities, k=3)
        topk_values = [themes[idx] for idx in topk_indices.tolist()]

    return predicted_value, list(zip(topk_values, topk_probs.tolist()))

# Создание интерфейса Streamlit
st.title("Value Recognition App")
st.write("Введите текст, чтобы получить соответствующие ценности.")

user_input = st.text_area("Введите ваш текст здесь:", "")

if st.button("Определить ценности"):
    if user_input:
        primary_value, top_values = classify_text(user_input)
        st.write(f"Основная предсказанная ценность: {primary_value}")
        st.write("Другие подходящие ценности:")
        for value, prob in top_values:
            st.write(f"{value}: {prob:.2f}")
    else:
        st.write("Пожалуйста, введите текст для анализа.")