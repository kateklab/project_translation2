import streamlit as st
from transformers import MBartForConditionalGeneration, MBartTokenizer


# Загрузка модели и токенизатора
@st.cache_resource
def load_model():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBartTokenizer.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


# Функция перевода текста
def translate_text(input_text, src_lang, tgt_lang, tokenizer, model):
    tokenizer.src_lang = src_lang
    encoded_input = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    generated_tokens = model.generate(**encoded_input, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_text


# Интерфейс Streamlit
st.title("Мультиязычный переводчик")
st.write(
    "Используем модель [facebook/mbart-large-50-many-to-many-mmt](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt)")

# Загружаем модель
tokenizer, model = load_model()

# Инициализация session_state
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
if "reversed_text" not in st.session_state:
    st.session_state.reversed_text = ""

# Поле для ввода текста
input_text = st.text_area("Введите текст для перевода:", placeholder="Введите текст здесь...")

# Выбор языков
src_lang = st.selectbox("Исходный язык:", ["en_XX", "ru_RU", "fr_XX", "de_DE", "es_XX"])
tgt_lang = st.selectbox("Целевой язык:", ["en_XX", "ru_RU", "fr_XX", "de_DE", "es_XX"])

# Кнопка перевода
if st.button("Перевести"):
    if input_text.strip():
        st.session_state.translated_text = translate_text(input_text, src_lang, tgt_lang, tokenizer, model)
        st.session_state.reversed_text = ""
    else:
        st.error("Пожалуйста, введите текст для перевода.")

# Отображение переведенного текста
if st.session_state.translated_text:
    st.write("**Перевод:**")
    st.write(st.session_state.translated_text)

    # Кнопка для обратного перевода
    if st.button("Обратный перевод"):
        st.session_state.reversed_text = translate_text(st.session_state.translated_text, tgt_lang, src_lang, tokenizer,
                                                        model)

# Отображение обратного перевода
if st.session_state.reversed_text:
    st.write("**Обратный перевод:**")
    st.write(st.session_state.reversed_text)
