"""
NLP - WORD2VEC (FASTTEXT) AND TRANSFORMER EMBEDDINGS
Word embeddings, similar-word search, PCA visualization,
and DistilBERT token/sentence embeddings.
"""

import gensim.downloader as api
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel


def word2vec_embeddings():
    # Load pretrained FastText word vectors (300 dimensions).
    # The first download is large and may take some time.
    print("Loading pretrained FastText word vectors...")
    word2vec_model = api.load("fasttext-wiki-news-subwords-300")

    # Get vector for a sample word.
    vector = word2vec_model["machine"]
    print("\nMachine Vector:")
    print(vector)

    # Select words and get their vectors.
    words = ["king", "queen", "man", "woman", "apple", "orange"]
    word_vectors = np.array([word2vec_model[word] for word in words])

    # Reduce 300 dimensions to 2 using PCA.
    pca = PCA(n_components=2)
    word_vecs_2d = pca.fit_transform(word_vectors)

    # Visualize word embeddings.
    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words):
        plt.scatter(word_vecs_2d[i, 0], word_vecs_2d[i, 1])
        plt.text(word_vecs_2d[i, 0], word_vecs_2d[i, 1], word)

    plt.title("Pre-trained FastText Word Embeddings")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Find similar words.
    print("\nSimilar words to machine:")
    print(word2vec_model.most_similar("machine", topn=5))


def transformer_embeddings():
    # Load pretrained DistilBERT tokenizer and model.
    model_name = "distilbert-base-uncased"
    print("\nLoading DistilBERT model...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Create and tokenize a sentence.
    sentence = "Natural Language Processing is a fascinating field."
    inputs = tokenizer(sentence, return_tensors="pt")

    # Extract embeddings without calculating gradients.
    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.squeeze(0)

    print("\nToken Embedding Shape:")
    print(embeddings.shape)

    # Mean pooling to create a sentence embedding.
    sentence_embedding = torch.mean(embeddings, dim=0)

    print("\nSentence Embedding Shape:")
    print(sentence_embedding.shape)

    # Convert token IDs to readable tokens.
    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"].squeeze(0).tolist()
    )

    # Reduce token embeddings to 2D using PCA.
    token_embeddings = embeddings.cpu().numpy()
    token_embeddings_2d = PCA(n_components=2).fit_transform(token_embeddings)

    # Visualize DistilBERT token embeddings.
    plt.figure(figsize=(12, 8))
    for i, token in enumerate(tokens):
        plt.scatter(token_embeddings_2d[i, 0], token_embeddings_2d[i, 1])
        plt.annotate(
            token,
            (token_embeddings_2d[i, 0], token_embeddings_2d[i, 1])
        )

    plt.title("DistilBERT Token Embeddings")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    word2vec_embeddings()
    transformer_embeddings()
