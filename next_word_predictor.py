from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import csv
import pickle
import re
from typing import Any


WORD_RE = re.compile(r"[a-zA-Z']+")


@dataclass(frozen=True)
class PredictionResult:
    """One ranked next word and the share of the chosen context it accounts for."""

    word: str
    probability: float


class DatasetPredictor:
    """Next-word counts learned from plain text. This is the model the page uses."""

    def __init__(self, transitions: dict[tuple[str, ...], Counter[str]]):
        """Store the finished transition table. Keys are prior words, values are next-word counts."""
        self.transitions = transitions

    @classmethod
    def from_csv(cls, dataset_path: str | Path) -> "DatasetPredictor":
        """Read the `quote` column of a CSV and build a predictor from those rows."""
        with open(dataset_path, newline="", encoding="utf-8", errors="replace") as dataset_file:
            rows = csv.DictReader(dataset_file)
            return cls.from_texts(row["quote"] for row in rows if row.get("quote"))

    @classmethod
    def from_texts(cls, texts) -> "DatasetPredictor":
        """Count how often each word follows the 1 to 5 words immediately before it."""
        transitions: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        for text in texts:
            words = tokenize(text)
            for position in range(1, len(words)):
                next_word = words[position]
                for context_size in range(1, min(5, position) + 1):
                    context = tuple(words[position - context_size : position])
                    transitions[context][next_word] += 1
        return cls(dict(transitions))

    def predict(self, text: str, top_k: int = 3) -> list[PredictionResult]:
        """Return up to `top_k` next words for `text`, most common first.

        Probability is that word's count divided by every count in the same context.
        Raises ValueError when `text` has no words. Returns an empty list when no
        context in the table matches.
        """
        words = tokenize(text)
        if not words:
            raise ValueError("Enter some text before predicting the next word.")

        counter = self._counter_for_context(words)
        if not counter:
            return []

        total = sum(counter.values())
        return [
            PredictionResult(word=word, probability=count / total)
            for word, count in counter.most_common(top_k)
        ]

    def generate(self, seed_text: str, word_count: int = 5) -> str:
        """Append up to `word_count` top predictions, one word at a time, and return the sentence."""
        generated = seed_text.strip()
        for _ in range(word_count):
            predictions = self.predict(generated, top_k=1)
            if not predictions:
                break
            generated = f"{generated} {predictions[0].word}".strip()
        return generated

    def _counter_for_context(self, words: list[str]) -> Counter[str]:
        """Use the longest suffix of `words`, from 5 down to 1, that exists in the table."""
        for context_size in range(min(5, len(words)), 0, -1):
            context = tuple(words[-context_size:])
            if context in self.transitions:
                return self.transitions[context]
        return Counter()


def tokenize(text: str) -> list[str]:
    """Lowercase letter-and-apostrophe tokens. Other characters are ignored."""
    return [match.group(0).lower().strip("'") for match in WORD_RE.finditer(text) if match.group(0).strip("'")]


def index_word_lookup(tokenizer: Any) -> dict[int, str]:
    """Reverse a Keras tokenizer's word_index so an output index can be turned back into a word."""
    return {index: word for word, index in tokenizer.word_index.items()}


def pad_sequence(sequence: list[int], max_len: int) -> list[list[int]]:
    """Keep the last `max_len` ids and pad on the left with zeros. The page does not call this."""
    trimmed = sequence[-max_len:]
    return [[0] * (max_len - len(trimmed)) + trimmed]


def predict_next_words(
    model: Any,
    tokenizer: Any,
    text: str,
    max_len: int,
    top_k: int = 3,
) -> list[PredictionResult]:
    """Rank the LSTM's next-word probabilities. The Streamlit page does not call this."""
    cleaned_text = text.strip().lower()
    if not cleaned_text:
        raise ValueError("Enter some text before predicting the next word.")

    sequence = tokenizer.texts_to_sequences([cleaned_text])[0]
    padded = pad_sequence(sequence, max_len=max_len)
    predictions = list(model.predict(padded, verbose=0)[0])
    lookup = index_word_lookup(tokenizer)

    ranked_indexes = sorted(range(len(predictions)), key=lambda index: predictions[index], reverse=True)
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
    """Append up to `word_count` LSTM predictions. The Streamlit page does not call this."""
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
    """Load the saved Keras model, tokenizer, and sequence length. Imports TensorFlow when called."""
    from tensorflow.keras.models import load_model

    model = load_model(model_path)
    with open(tokenizer_path, "rb") as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    with open(max_len_path, "rb") as max_len_file:
        max_len = pickle.load(max_len_file)
    return model, tokenizer, int(max_len)
