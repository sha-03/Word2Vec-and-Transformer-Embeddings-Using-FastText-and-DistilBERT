# Word2Vec and Transformer Embeddings Using FastText and DistilBERT

This Python project demonstrates two approaches to representing words and text as numerical vectors:

1. **Pre-trained FastText word embeddings** loaded through Gensim.
2. **Transformer embeddings** generated using the DistilBERT model from Hugging Face Transformers.

It also uses Principal Component Analysis (PCA) to reduce high-dimensional embeddings to two dimensions for visualization.

## Features

### FastText / Word Embeddings
- Loads pre-trained 300-dimensional FastText word vectors.
- Displays the vector for the word `machine`.
- Retrieves vectors for selected words.
- Uses PCA to visualize word embeddings in 2D.
- Finds the five words most similar to `machine`.

### DistilBERT / Transformer Embeddings
- Loads the pre-trained `distilbert-base-uncased` tokenizer and model.
- Tokenizes a sample sentence.
- Extracts contextual token embeddings.
- Creates a sentence embedding using mean pooling.
- Uses PCA to visualize token embeddings in 2D.

## Project Structure

```text
word2vec-transformer-embeddings/
├── word2vec_transformer_embeddings.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.9 or later
- gensim
- transformers
- torch
- scikit-learn
- matplotlib
- numpy

Install the required packages:

```bash
pip install -r requirements.txt
```

## How to Run

Run the program from the project folder:

```bash
python word2vec_transformer_embeddings.py
```

On some systems, use `python3` instead of `python`.

The code is written as a regular Python script; the Jupyter/Colab-only `%pip` command has been removed.

## Model Downloads

The first run downloads the pre-trained models:

- `fasttext-wiki-news-subwords-300` from Gensim: approximately 959 MB.
- `distilbert-base-uncased` from Hugging Face: model files are also downloaded and cached locally.

A stable internet connection and sufficient disk space are required for the first run. Later runs can reuse the cached model files.

## Concepts Used

- **Word embeddings:** Numerical representations of words that capture aspects of their meaning and relationships.
- **FastText:** Learns word representations using character subword information, which can help represent words not seen exactly during training.
- **Transformer embeddings:** Context-sensitive representations produced by a transformer model. A token's representation can depend on the surrounding sentence.
- **DistilBERT:** A smaller, faster transformer model based on BERT.
- **PCA:** Principal Component Analysis reduces the number of dimensions while retaining as much variance as possible.
- **Sentence embedding:** This project uses mean pooling of the token embeddings as a simple sentence representation.

## Output

The program displays:

1. The 300-dimensional vector for `machine`.
2. A 2D plot of selected FastText word embeddings.
3. The five most similar words to `machine`.
4. The shape of the DistilBERT token embedding matrix.
5. The shape of the sentence embedding.
6. A 2D plot of DistilBERT token embeddings.

Exact similarity results and numerical values may vary with library/model versions.

## Author

Add your name here.
