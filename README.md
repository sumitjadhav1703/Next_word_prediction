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

Use Python 3.10, 3.11, or 3.12. TensorFlow is not currently available for Python 3.14 in this environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Test

```bash
python -m unittest discover -s tests -t .
```
