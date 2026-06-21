# Next Word Prediction

A dual-model architecture for Next Word Prediction trained on a quotes dataset. This project features a robust **Deep Learning LSTM model** intended for research and advanced text generation, alongside a lightweight, high-performance **n-gram fallback predictor** explicitly designed to run seamlessly on restricted production environments like Streamlit Community Cloud.

The primary use case is ranking and predicting the most likely next words from a given sentence fragment, and continuously generating text continuations based on the dataset's stylistic patterns.

## Features

- **Dual Prediction Engine**: Uses an n-gram model in production to bypass Streamlit latency and memory bottlenecks, while providing an LSTM model for robust representation learning.
- **Variable Sequence Length Handling**: The LSTM accepts sequences up to a parameterized maximum context window (e.g., 50 tokens) rather than the absolute maximum quote length, preventing vanishing gradients and OOM errors.
- **Ranked Confidence Scores**: Returns predictions alongside probability/confidence scores.
- **Recursive Generation**: Capable of generating complete continuations by recursively feeding predicted words back into the model's context window.

## Data Preprocessing Pipeline

The model ingests a dataset of famous quotes (`qoute_dataset.csv`). The pipeline applies the following steps:
1. **Cleaning**: Quotation marks and special symbols are stripped, and text is lowercased.
2. **Tokenization**: Uses Keras `Tokenizer` to map the most frequent words to integer indices. The vocabulary size is configurable but bounded by the dataset (e.g., ~10,000 words).
3. **N-gram Sequence Generation**: Sentences are split into incremental n-gram sequences to generate multiple `[context] -> [target]` pairs per quote.
4. **Padding & Truncation**: Sequences are pre-padded to a strict maximum length (e.g., `seq_len = 50`). Truncating the context window rather than using the absolute maximum quote length (745 tokens) ensures efficient training and mitigates vanishing gradients.

## Model Architecture

The repository maintains two distinct models:

### 1. Research Model: LSTM (Deep Learning)
The LSTM model is designed for sophisticated sequence modeling and pattern recognition.
- **Embedding Layer**: Transforms integer token indices into dense vectors of fixed size (default: 100).
- **LSTM Layer**: A recurrent neural network layer (default: 150 units) capable of learning long-term dependencies in the sequence data.
- **Dense Layer**: The output layer using a `softmax` activation function, emitting a probability distribution over the entire vocabulary size.

### 2. Production Model: N-gram Predictor
A fast, memory-efficient predictor built with Python's standard library (`collections.Counter`). It calculates exact transition probabilities across sliding 1-to-5 word contexts. This model is deployed to `app.py` as it reliably fits within the restrictive resource constraints of Streamlit Community Cloud.

## Installation & Setup

We recommend using a virtual environment. Note that `tensorflow-cpu` is specified instead of `tensorflow` to ensure lightweight, crash-free installations on constrained environments.

```bash
git clone https://github.com/sumitjadhav1703/Next_word_prediction.git
cd Next_word_prediction
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Training the Model

The deep learning model is parameterized via a standalone script, `train.py`. Do not rely on the legacy Jupyter notebook for production builds.

To train the model with default parameters:
```bash
python train.py
```

To override hyperparameters, pass CLI arguments:
```bash
python train.py --data_path qoute_dataset.csv --epochs 50 --batch_size 64 --seq_len 30
```

### Hyperparameters & Evaluation

| Parameter | Default Value |
| :--- | :--- |
| Epochs | 100 |
| Batch Size | 128 |
| Sequence Length (Context Window) | 50 |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| **Final Training Accuracy** | 0.0405 |
| **Final Validation Loss** | 6.6263 |

*(Note: The metrics above reflect a 1-epoch test run. For optimal results, train for 50-100 epochs).*

## Inference / Usage

To utilize the trained LSTM model in a Python script, you can rely on the provided helper functions inside `next_word_predictor.py`:

```python
from next_word_predictor import load_artifacts, generate_text, predict_next_words

# 1. Load the compiled artifacts
model, tokenizer, max_len = load_artifacts(
    model_path="lstm_model.keras",
    tokenizer_path="tokenizer.pkl",
    max_len_path="max_len.pkl"
)

# 2. Get the top K next word predictions
seed = "the world as we"
predictions = predict_next_words(model, tokenizer, seed, max_len, top_k=3)
for p in predictions:
    print(f"Word: {p.word}, Confidence: {p.probability:.2%}")

# 3. Generate a multi-word continuation
continuation = generate_text(model, tokenizer, seed, max_len, word_count=5)
print(f"Generated text: {continuation}")
```

## Directory Structure

```text
.
├── app.py                         # Streamlit UI (Production App)
├── train.py                       # Parameterized LSTM training script
├── next_word_predictor.py         # Helper utilities for both n-gram and LSTM
├── qoute_dataset.csv              # Primary text corpus
├── requirements.txt               # Pinned dependencies (tensorflow-cpu)
├── lstm_model.keras               # Saved LSTM model weights (generated via train.py)
├── tokenizer.pkl                  # Pickled Keras Tokenizer (generated via train.py)
├── max_len.pkl                    # Pickled max sequence integer (generated via train.py)
├── RNN_2_proj.ipynb               # Legacy exploratory notebook
└── tests/                         # Unit tests
```

## Future Scope / Scalability

While the current architecture is fully functional, scaling it to larger datasets introduces a few technical limitations:
1. **Basic Tokenization Strategy**: The current pipeline uses standard word-level Keras Tokenization. Moving to Subword Tokenization (like Byte-Pair Encoding or WordPiece) would drastically reduce the vocabulary size, handle Out-Of-Vocabulary (OOV) words natively, and improve memory efficiency.
2. **Context Window Limitations**: LSTMs struggle to retain context beyond ~100 tokens. Replacing the LSTM block with a Transformer/Attention-based architecture (like GPT) would allow for significantly larger context windows and better generation coherency.
3. **Deployment Constraints**: The LSTM model is currently disabled in the Streamlit app due to the memory overhead of the TensorFlow runtime. Exporting the model to ONNX or TensorFlow Lite could allow the Deep Learning model to run efficiently in constrained web environments without relying on the n-gram fallback.
