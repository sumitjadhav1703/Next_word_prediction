from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PredictionResult:
    word: str
    probability: float


def index_word_lookup(tokenizer: Any) -> dict[int, str]:
    return {index: word for word, index in tokenizer.word_index.items()}


def pad_sequence(sequence: list[int], max_len: int) -> np.ndarray:
    padded = np.zeros((1, max_len), dtype=np.int32)
    trimmed = sequence[-max_len:]
    if trimmed:
        padded[0, -len(trimmed) :] = trimmed
    return padded


def predict_next_words(
    model: Any,
    tokenizer: Any,
    text: str,
    max_len: int,
    top_k: int = 3,
) -> list[PredictionResult]:
    cleaned_text = text.strip().lower()
    if not cleaned_text:
        raise ValueError("Enter some text before predicting the next word.")

    sequence = tokenizer.texts_to_sequences([cleaned_text])[0]
    padded = pad_sequence(sequence, max_len=max_len)
    predictions = np.asarray(model.predict(padded, verbose=0))[0]
    lookup = index_word_lookup(tokenizer)

    ranked_indexes = np.argsort(predictions)[::-1]
    results: list[PredictionResult] = []
    for token_index in ranked_indexes:
        word = lookup.get(int(token_index))
        if not word:
            continue
        results.append(PredictionResult(word=word, probability=float(predictions[token_index])))
        if len(results) == top_k:
            break
    return results


def generate_text(
    model: Any,
    tokenizer: Any,
    seed_text: str,
    max_len: int,
    word_count: int = 5,
) -> str:
    generated = seed_text.strip()
    for _ in range(word_count):
        predictions = predict_next_words(model, tokenizer, generated, max_len=max_len, top_k=1)
        if not predictions:
            break
        generated = f"{generated} {predictions[0].word}".strip()
    return generated


def load_artifacts(
    model_path: str | Path = "lstm_model.h5",
    tokenizer_path: str | Path = "tokenizer.pkl",
    max_len_path: str | Path = "max_len.pkl",
) -> tuple[Any, Any, int]:
    from tensorflow.keras.models import load_model

    model = load_model(model_path)
    with open(tokenizer_path, "rb") as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    with open(max_len_path, "rb") as max_len_file:
        max_len = pickle.load(max_len_file)
    return model, tokenizer, int(max_len)
