from pathlib import Path
from html import escape

import streamlit as st

from next_word_predictor import generate_text, load_artifacts, predict_next_words


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "lstm_model.h5"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"
MAX_LEN_PATH = BASE_DIR / "max_len.pkl"


st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🧠",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def load_resources():
    return load_artifacts(MODEL_PATH, TOKENIZER_PATH, MAX_LEN_PATH)


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7fafc 0%, #eef3f8 100%);
    }
    .main .block-container {
        max-width: 900px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    .hero {
        border-left: 5px solid #2563eb;
        padding: 0.2rem 0 0.4rem 1rem;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        color: #111827;
        font-size: 2.6rem;
        line-height: 1.1;
        margin: 0;
        letter-spacing: 0;
    }
    .hero p {
        color: #4b5563;
        font-size: 1.05rem;
        margin-top: 0.7rem;
    }
    .prediction-card {
        background: #ffffff;
        border: 1px solid #dbe3ec;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.7rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
    }
    .prediction-word {
        color: #111827;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .prediction-score {
        color: #526071;
        font-size: 0.95rem;
    }
    .generated-text {
        background: #0f172a;
        color: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        line-height: 1.7;
        margin-top: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">
        <h1>Next Word Prediction</h1>
        <p>Type the start of a quote or sentence and the LSTM model will rank likely next words.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Predictions", min_value=1, max_value=5, value=3)
    generated_words = st.slider("Generate words", min_value=1, max_value=12, value=5)
    st.divider()
    st.caption("Model artifact: `lstm_model.h5`")
    st.caption("Dataset: `qoute_dataset.csv`")


examples = [
    "the world as we",
    "it is our choices",
    "there are only two",
]

selected_example = st.selectbox("Try an example", [""] + examples, format_func=lambda value: value or "Choose one")
default_text = selected_example or "the world as we"
user_input = st.text_area(
    "Input text",
    value=default_text,
    height=120,
    placeholder="Start typing a quote or sentence...",
)

predict_clicked = st.button("Predict", type="primary", use_container_width=True)

if predict_clicked:
    try:
        with st.spinner("Loading model and predicting..."):
            model, tokenizer, max_len = load_resources()
            predictions = predict_next_words(model, tokenizer, user_input, max_len=max_len, top_k=top_k)
            generated = generate_text(
                model,
                tokenizer,
                user_input,
                max_len=max_len,
                word_count=generated_words,
            )

        if not predictions:
            st.warning("The model could not produce a known next-word prediction for this input.")
        else:
            st.subheader("Ranked predictions")
            for prediction in predictions:
                safe_word = escape(prediction.word)
                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <div class="prediction-word">{safe_word}</div>
                        <div class="prediction-score">Confidence: {prediction.probability:.2%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.subheader("Generated continuation")
            st.markdown(f'<div class="generated-text">{escape(generated)}</div>', unsafe_allow_html=True)
    except ValueError as error:
        st.warning(str(error))
    except Exception as error:
        st.error("The model could not be loaded in this Python environment.")
        st.caption(str(error))
