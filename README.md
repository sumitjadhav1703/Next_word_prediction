# Next Word Prediction

A Streamlit web app that predicts likely next words from a sentence fragment using a quote dataset. The deployed version uses a lightweight n-gram predictor so it runs cleanly on Streamlit Community Cloud without TensorFlow installation issues.

## Live App

https://nextwordprediction-gj6q3h2okjbqevqbmzuw43.streamlit.app/

## Features

- Predicts the most likely next words for a typed phrase.
- Shows ranked predictions with confidence percentages.
- Generates a short continuation from the selected seed text.
- Includes example prompts for quick testing.
- Runs on Streamlit Cloud with only Streamlit plus Python standard-library code.

## How It Works

The deployed app reads `qoute_dataset.csv`, tokenizes the quote text, and builds word-transition counts from one-word through five-word contexts. At prediction time, it uses the longest matching context from the user input and returns the most frequent next words from the dataset.

The repository also keeps the original LSTM training notebook and saved model artifacts for reference, but they are not installed or loaded by the deployed Streamlit app because Streamlit Cloud currently uses Python 3.14 and TensorFlow does not provide compatible wheels for that runtime.

## Project Structure

```text
.
├── app.py                         # Streamlit UI
├── next_word_predictor.py         # Dataset predictor and legacy LSTM helpers
├── qoute_dataset.csv              # Quote dataset used by the deployed app
├── RNN_2_proj.ipynb               # Original model-training notebook
├── lstm_model.h5                  # Saved LSTM model artifact, reference only
├── tokenizer.pkl                  # Saved tokenizer artifact, reference only
├── max_len.pkl                    # Saved sequence length artifact, reference only
└── tests/                         # Unit and deployment-safety tests
```

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install streamlit
streamlit run app.py
```

## Test

```bash
python -m unittest discover -s tests -t .
```

## Deployment Notes

There is intentionally no `requirements.txt` file. Streamlit Community Cloud includes Streamlit by default, and adding TensorFlow or NumPy requirements currently breaks installation on its Python 3.14 environment. The deployment tests check that no incompatible dependency file is reintroduced.

## Tech Stack

- Python
- Streamlit
- Standard-library CSV, regex, and collections utilities
- Unittest

## Example Inputs

```text
the world as we
it is our choices
there are only two
```
