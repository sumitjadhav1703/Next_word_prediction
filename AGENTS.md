# Next Word Prediction

A small Streamlit demo that ranks the next word of a quote fragment. The page is the thing a reviewer opens. The scoring code is already the product, so later edits should make the page clearer and leave the scores alone.

## What runs in the app

`app.py` is the only page. It builds one `DatasetPredictor` from `qoute_dataset.csv` and calls `predict` and `generate`. That predictor counts word transitions in the quote column. It does not load `lstm_model.h5`.

`next_word_predictor.py` has two separate tools:

- `DatasetPredictor` is the live model. `from_csv` reads the `quote` column. `from_texts` counts how often each word follows the 1 to 5 words before it. `predict` uses the longest context that exists in that table and returns the most common next words, with probability equal to that word's count divided by the context total. `generate` appends the top word, one at a time.
- `tokenize` keeps letters and apostrophes and lowercases them. Other characters, including the curly quotes in the CSV, are dropped.
- `predict_next_words`, `generate_text`, `pad_sequence`, `index_word_lookup`, and `load_artifacts` are the LSTM helpers. `load_artifacts` imports TensorFlow when it is called. The page never calls these functions. `tests/test_next_word_predictor.py` covers them with a fake model, so TensorFlow is not required to run the tests.

## Files

| File | Role |
| --- | --- |
| `app.py` | Streamlit page. Input, example phrases, ranked words, generated continuation. |
| `next_word_predictor.py` | N-gram predictor used by the page, plus unused LSTM helpers. |
| `qoute_dataset.csv` | Quotes. Columns are `quote` and `Author`. The filename spelling is intentional. The page only reads `quote`. |
| `RNN_2_proj.ipynb` | Colab notebook that loads the CSV with pandas and trains the LSTM. It is not part of the running page. |
| `lstm_model.h5` | Saved Keras model from that notebook. |
| `tokenizer.pkl` | Keras tokenizer saved beside the model. |
| `max_len.pkl` | Sequence length saved beside the model. |
| `tests/test_next_word_predictor.py` | Ranking, blank input, and a small hand-built quote table. |
| `tests/test_app.py` | Page flows. Skips when Streamlit is not installed. |
| `tests/test_deployment_config.py` | Fails if `requirements.txt` or `runtime.txt` is added. |
| `.streamlit/config.toml` | Page colors. Not a dependency list. |
| `assets/demo.png` | Screenshot used by the README. |
| `.python-version` | `3.11`. The page also runs on the system Python 3.12 used in this environment. |

The CSV has 3,038 quote rows. Do not rename `qoute_dataset.csv`. The deployed app and the notebook both open that name.

## Hard rules

- Do not change transition counts, the 1-to-5 context window, ranking, or the LSTM helpers' math. A UI change may display the same probability. It may not recompute it.
- Do not edit `lstm_model.h5`, `tokenizer.pkl`, or `max_len.pkl` to chase a different word.
- Do not add `requirements.txt` or `runtime.txt`. Streamlit Community Cloud installs Streamlit itself. A requirements file that pulls TensorFlow or NumPy is what `tests/test_deployment_config.py` is there to prevent.
- There is no HTTP client and no model provider. Do not add a network fetch or a cache layer for the CSV. The file is local and small.
- Keep new page dependencies inside Streamlit and the Python standard library.

## Commands

```bash
python -m unittest discover -s tests -t .
streamlit run app.py
```

The page listens on port 8501. Examples that match the current quote table:

- `the world as we` -> `have`
- `it is our choices` -> `harry`
- `there are only two` -> `ways`

An empty input raises `Enter some text before predicting the next word.` A phrase with no recorded next word returns an empty list.

## Docs this map was written from

- `README.md` sections Project Structure, Installation, Requirements, Inference, Usage Examples, and FAQ. Usage Examples lists the words returned by `DatasetPredictor` on `qoute_dataset.csv`: `have`, `harry`, and `ways`.
- `app.py`, `next_word_predictor.py`, both test modules, the CSV header, and the first cells of `RNN_2_proj.ipynb`.
- UI constraints used for the page: accessible contrast, 44px targets, visible labels, and errors next to the field. The page is a tool, so the layout stays one column and the existing light editorial colors.
