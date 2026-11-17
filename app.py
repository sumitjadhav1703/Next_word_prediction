"""Streamlit page for the quote next-word demo.

The page asks DatasetPredictor for the same counts the tests cover. It does not
load the LSTM files and it does not change those counts.
"""

from html import escape
from pathlib import Path

import streamlit as st

from next_word_predictor import DatasetPredictor, PredictionResult


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "qoute_dataset.csv"
DEFAULT_TEXT = "the world as we"
EXAMPLES = (
    "the world as we",
    "it is our choices",
    "there are only two",
)


st.set_page_config(
    page_title="Next Word Prediction",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def load_resources() -> DatasetPredictor:
    """Build the quote predictor once per server process."""
    return DatasetPredictor.from_csv(DATASET_PATH)


def apply_page_style() -> None:
    """Set the page colors, type, and control size. Does not change prediction results."""
    st.markdown(
        """
        <style>
        .stApp {
            background: #f7fafc;
        }
        .main .block-container {
            max-width: 42rem;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }
        .hero {
            border-left: 4px solid #1d4ed8;
            padding: 0.1rem 0 0.2rem 1rem;
            margin-bottom: 1.75rem;
        }
        .hero h1 {
            color: #111827;
            font-size: 2.4rem;
            line-height: 1.15;
            margin: 0;
            letter-spacing: -0.02em;
            text-wrap: balance;
        }
        .hero p {
            color: #374151;
            font-size: 1.05rem;
            line-height: 1.5;
            margin: 0.75rem 0 0;
            max-width: 65ch;
        }
        .prediction-card {
            background: #ffffff;
            border: 1px solid #dbe3ec;
            border-radius: 8px;
            padding: 1rem 1rem 0.9rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
        }
        .prediction-rank {
            color: #374151;
            font-size: 0.85rem;
            margin: 0 0 0.2rem;
        }
        .prediction-word {
            color: #111827;
            font-size: 1.45rem;
            font-weight: 700;
            line-height: 1.3;
            margin: 0;
        }
        .prediction-score {
            color: #374151;
            font-size: 1rem;
            margin: 0.35rem 0 0.55rem;
        }
        .confidence-track {
            height: 8px;
            background: #e5e7eb;
            border-radius: 999px;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            background: #1d4ed8;
        }
        .stButton > button {
            min-height: 44px;
            cursor: pointer;
        }
        .stButton > button:focus-visible {
            outline: 2px solid #1d4ed8;
            outline-offset: 2px;
        }
        ::selection {
            background: #dbeafe;
            color: #111827;
        }
        @media (prefers-reduced-motion: reduce) {
            .stButton > button {
                transition: none;
            }
        }
        @media (max-width: 640px) {
            .main .block-container {
                padding-top: 1.25rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .hero h1 {
                font-size: 2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def missing_prediction_note(found: int, requested: int) -> str | None:
    """Explain a short result list. Returns None when every requested word was found."""
    if found >= requested:
        return None
    if found == 1:
        return f"1 of {requested} requested next words appears after this phrase."
    return f"{found} of {requested} requested next words appear after this phrase."


def prediction_card(rank: int, prediction: PredictionResult) -> str:
    """HTML for one next word. The percentage is the predictor's probability, unchanged."""
    width = f"{prediction.probability * 100:.2f}%"
    return f"""
        <div class="prediction-card">
            <p class="prediction-rank">{rank}</p>
            <p class="prediction-word">{escape(prediction.word)}</p>
            <p class="prediction-score">Confidence: {prediction.probability:.2%}</p>
            <div class="confidence-track">
                <div class="confidence-fill" style="width: {width}"></div>
            </div>
        </div>
        """


def use_example(phrase: str) -> None:
    """Replace the sentence in the input with one of the demo phrases."""
    st.session_state.draft = phrase


apply_page_style()

if "draft" not in st.session_state:
    st.session_state.draft = DEFAULT_TEXT

st.markdown(
    """
    <div class="hero">
        <h1>Next Word Prediction</h1>
        <p>Type the start of a quote. The page ranks the next words that follow that phrase in the quote file.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Predictions", min_value=1, max_value=5, value=3)
    generated_words = st.slider("Generate words", min_value=1, max_value=12, value=5)
    st.divider()
    st.caption("Counts come from the quote file. This page does not load the saved LSTM model.")

st.caption("Try an example")
for index, phrase in enumerate(EXAMPLES):
    st.button(
        phrase,
        key=f"example-{index}",
        on_click=use_example,
        args=(phrase,),
        use_container_width=True,
    )

with st.form("predict-form"):
    user_input = st.text_area(
        "Input text",
        key="draft",
        height=120,
        placeholder="Start typing a quote or sentence...",
    )
    predict_clicked = st.form_submit_button("Predict", type="primary", use_container_width=True)

if predict_clicked:
    try:
        with st.spinner("Predicting..."):
            predictor = load_resources()
            predictions = predictor.predict(user_input, top_k=top_k)
            generated = predictor.generate(user_input, word_count=generated_words)
    except ValueError as error:
        st.warning(str(error))
    except Exception as error:
        st.error("Prediction failed.")
        st.caption(str(error))
    else:
        if not predictions:
            st.warning("No recorded next word follows this phrase. Try a shorter or more familiar quote.")
        else:
            note = missing_prediction_note(len(predictions), top_k)
            st.subheader("Ranked predictions")
            if note:
                st.caption(note)
            for rank, prediction in enumerate(predictions, start=1):
                st.markdown(prediction_card(rank, prediction), unsafe_allow_html=True)
            st.subheader("Generated continuation")
            st.code(generated, language=None, wrap_lines=True)
