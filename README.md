Sentiment Analysis System — Airline Reviews

A deep-learning based sentiment classifier for airline customer reviews using RNN / LSTM / GRU (TensorFlow / Keras). Achieved ~97% accuracy with proper preprocessing and model tuning.

🔍 Project Summary

This project classifies airline reviews into positive, negative, or neutral sentiment. It includes a full preprocessing pipeline (cleaning with regex + NLTK/Pandas), model experiments (RNN, LSTM, GRU), training/evaluation code, and inference utilities. The repo contains code to reproduce training and to run predictions on new text.

⭐ Key Features

Text cleaning & normalization (Regex, lowercasing, stopwords removal)

Tokenization & sequence padding (Keras Tokenizer)

Model experiments: RNN, LSTM, GRU (TensorFlow/Keras)

Model saving & loading for inference

Train/validation split, accuracy + confusion matrix evaluation

Simple inference script to predict sentiment for new input text

🧰 Tech Stack

Python 3.8+

TensorFlow / Keras

Numpy, Pandas, scikit-learn

NLTK (stopwords, tokenizers)

Matplotlib / Seaborn (optional for plots)

(My) sample commands use pip / venv

📁 Suggested Repository Structure
sentiment-analysis/
├─ data/
│  ├─ airline_reviews.csv         # raw dataset (do not commit large files)
│  └─ processed/                  # optional processed files
├─ notebooks/
│  └─ EDA-and-Model-Experiments.ipynb
├─ src/
│  ├─ preprocess.py               # cleaning/tokenization/padding routines
│  ├─ models.py                    # model architectures (RNN, LSTM, GRU)
│  ├─ train.py                     # training loop, callbacks, save model
│  ├─ evaluate.py                  # evaluation & metrics
│  └─ infer.py                     # load model + predict for new text
├─ requirements.txt
├─ README.md
├─ .gitignore
└─ LICENSE

🧾 Dataset

Type: Labeled airline customer reviews (text + sentiment)

Columns (example): review_text, sentiment (values: positive/negative/neutral)

If you used a public dataset, add citation/link here. If it’s proprietary, note that in the README and do not commit raw data.

🔧 Preprocessing (what I did)

Main steps implemented in src/preprocess.py:

Lowercase the text.

Remove special characters & punctuation using regex: re.sub(r'[^a-zA-Z0-9 ]', '', text)

Remove URLs / emails if present.

Remove extra whitespace.

Tokenize using tensorflow.keras.preprocessing.text.Tokenizer.

Convert words to sequences and pad using pad_sequences(maxlen=MAX_LEN).

Encode sentiment labels (e.g., LabelEncoder or integer mapping).

Example cleaning function

import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

🏗️ Model Architectures (in src/models.py)

I experimented with three sequence models:

1. Simple RNN

Embedding → SimpleRNN(units=128) → Dense(3, activation='softmax')

Good baseline; suffers on long sequences.

2. LSTM

Embedding → LSTM(units=128, return_sequences=False) → Dense(3, activation='softmax')

Gates help learn long-range dependencies; best accuracy in my experiments.

3. GRU

Embedding → GRU(units=128) → Dense(3, activation='softmax')

Lighter than LSTM, trains faster with similar performance.

Common hyperparameters

Embedding dim: 100 or 128

Max sequence length: 100–200 (tune based on dataset)

Optimizer: Adam (lr 1e-3)

Loss: sparse_categorical_crossentropy for integer labels

Metrics: accuracy (+ precision/recall/f1 via sklearn)

▶️ Training (in src/train.py)

Train/validation split (e.g., 80/20).

Callbacks: ModelCheckpoint, EarlyStopping (monitor val_loss).

Train for ~10–30 epochs (use early stopping).

Save best model with model.save('models/lstm_best.h5').

Sample command

python src/train.py --model lstm --epochs 20 --batch_size 64

📊 Evaluation

Use confusion matrix to analyze class-wise errors.

Compute precision, recall, F1-score for each class (with sklearn.metrics.classification_report).

In my run: LSTM/GRU → ~97% accuracy (after tuning and proper preprocessing). Mention this result in your repo if reproducible.
