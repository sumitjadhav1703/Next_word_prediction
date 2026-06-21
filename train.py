import argparse
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical

def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM for Next Word Prediction")
    parser.add_argument("--data_path", type=str, default="qoute_dataset.csv", help="Path to the dataset CSV file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--seq_len", type=int, default=50, help="Maximum sequence length (context window)")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Maximum vocabulary size")
    parser.add_argument("--embedding_dim", type=int, default=100, help="Dimension of embedding layer")
    parser.add_argument("--lstm_units", type=int, default=150, help="Number of units in LSTM layer")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)

    # Preprocessing
    # Assuming the column name is 'quote' based on earlier dataset exploration
    quotes = df['quote'].dropna().astype(str).tolist()

    print("Tokenizing text...")
    tokenizer = Tokenizer(num_words=args.vocab_size)
    tokenizer.fit_on_texts(quotes)

    # Save tokenizer early
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    # Calculate actual vocabulary size (bounded by args.vocab_size)
    actual_vocab_size = min(len(tokenizer.word_index) + 1, args.vocab_size)
    print(f"Vocabulary Size: {actual_vocab_size}")

    print("Generating sequences...")
    input_sequences = []
    for line in quotes:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    print(f"Total Sequences: {len(input_sequences)}")

    print(f"Padding sequences to length {args.seq_len}...")
    # Pad sequences to a maximum length (args.seq_len)
    padded_sequences = pad_sequences(input_sequences, maxlen=args.seq_len, padding='pre')

    # Save max_len
    with open("max_len.pkl", "wb") as f:
        pickle.dump(args.seq_len, f)

    print("Splitting into X and y...")
    X = padded_sequences[:, :-1]
    y = padded_sequences[:, -1]

    # For training the LSTM we need the target variable to be one-hot encoded
    y_one_hot = to_categorical(y, num_classes=actual_vocab_size)

    print("Building model...")
    model = Sequential([
        Embedding(input_dim=actual_vocab_size, output_dim=args.embedding_dim, input_length=args.seq_len - 1),
        LSTM(units=args.lstm_units),
        Dense(units=actual_vocab_size, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    print("Starting training...")
    history = model.fit(
        X, y_one_hot,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1
    )

    print("Saving model...")
    model.save("lstm_model.keras")
    print("Training complete! Artifacts saved.")

if __name__ == "__main__":
    main()
