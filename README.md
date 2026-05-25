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

Use Python 3.11, 3.12, or 3.13. TensorFlow is not currently available for Python 3.14 in this environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Streamlit Community Cloud

Community Cloud selects the Python version from the app deployment settings, not from `runtime.txt`. The app pins `tensorflow==2.20.0` because that release has Python 3.13 wheels and installs on current Community Cloud runtimes.

## Run

```bash
streamlit run app.py
```

## Test

```bash
python -m unittest discover -s tests -t .
```
