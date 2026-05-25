# Next Word Prediction

A Streamlit app that uses a trained LSTM model to predict the next word for a sentence fragment.

## Project Files

- `app.py` - Streamlit interface.
- `next_word_predictor.py` - prediction and artifact-loading helpers.
- `lstm_model.h5` - trained LSTM model.
- `tokenizer.pkl` - saved Keras tokenizer.
- `max_len.pkl` - maximum sequence length used by the model.
- `qoute_dataset.csv` - training dataset.
- `RNN_2_proj.ipynb` - training notebook.

## Setup

The deployed app runs with Streamlit and Python's standard library only.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install streamlit
```

## Streamlit Community Cloud

Community Cloud is currently deploying this app with Python 3.14. TensorFlow does not provide Python 3.14 wheels, so the deployed app uses a lightweight dataset-backed n-gram predictor instead of installing TensorFlow.

## Run

```bash
streamlit run app.py
```

## Test

```bash
python -m unittest discover -s tests -t .
```
