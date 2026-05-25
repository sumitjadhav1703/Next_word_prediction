import unittest

from next_word_predictor import (
    DatasetPredictor,
    PredictionResult,
    index_word_lookup,
    predict_next_words,
)


class FakeTokenizer:
    def __init__(self):
        self.word_index = {"hello": 1, "world": 2, "friend": 3}

    def texts_to_sequences(self, texts):
        return [[self.word_index[word] for word in texts[0].lower().split() if word in self.word_index]]


class FakeModel:
    def predict(self, sequence, verbose=0):
        return [[0.0, 0.1, 0.7, 0.2]]


class PredictorTests(unittest.TestCase):
    def test_index_word_lookup_reverses_tokenizer_word_index(self):
        self.assertEqual(index_word_lookup(FakeTokenizer()), {1: "hello", 2: "world", 3: "friend"})

    def test_predict_next_words_returns_ranked_results(self):
        results = predict_next_words(FakeModel(), FakeTokenizer(), "hello", max_len=4, top_k=2)

        self.assertEqual(
            results,
            [
                PredictionResult(word="world", probability=0.7),
                PredictionResult(word="friend", probability=0.2),
            ],
        )

    def test_predict_next_words_rejects_blank_input(self):
        with self.assertRaisesRegex(ValueError, "Enter some text"):
            predict_next_words(FakeModel(), FakeTokenizer(), "   ", max_len=4)

    def test_dataset_predictor_returns_next_words_from_quote_context(self):
        predictor = DatasetPredictor.from_texts(
            [
                "the world as we know it",
                "the world as we see it",
                "the world as we know ourselves",
            ]
        )

        results = predictor.predict("the world as we", top_k=2)

        self.assertEqual(
            results,
            [
                PredictionResult(word="know", probability=2 / 3),
                PredictionResult(word="see", probability=1 / 3),
            ],
        )

    def test_dataset_predictor_generates_continuation(self):
        predictor = DatasetPredictor.from_texts(["the world as we know it"])

        self.assertEqual(predictor.generate("the world as we", word_count=2), "the world as we know it")


if __name__ == "__main__":
    unittest.main()
