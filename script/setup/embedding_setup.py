import json
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
import torch
# import fasttext
from tqdm import tqdm


# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "preprocessed" / "train.csv"
FASTTEXT_BIN = PROJECT_ROOT / "model" / "cc.en.300.bin"
OUTPUT_PATH = PROJECT_ROOT / "model" / "embedding.pt"

EMBED_DIM = 300
MIN_FREQ = 1

# =========================
# BUILD WORD2IDX
# =========================
def build_word2idx(texts, min_freq=1):
    counter = Counter()
    for sent in texts:
        for w in sent.split():
            counter[w] += 1

    word2idx = {
        "<pad>": 0,
        "<unk>": 1,
    }

    idx = 2
    for word, freq in counter.items():
        if freq >= min_freq:
            word2idx[word] = idx
            idx += 1

    return word2idx


# =========================
# BUILD EMBEDDING MATRIX
# =========================
def build_embedding(ft_model, word2idx, embed_dim=300):
    vocab_size = len(word2idx)
    emb = np.zeros((vocab_size, embed_dim), dtype=np.float32)

    for word, idx in tqdm(word2idx.items(), desc="Building embedding"):
        if word in ("<pad>", "<unk>"):
            continue
        emb[idx] = ft_model.get_word_vector(word)

    return torch.tensor(emb)


# =========================
# MAIN
# =========================
def main():
    # Load data
    train_df = pd.read_csv(DATA_PATH)
    texts = train_df["text"].astype(str).tolist()

    # Build vocab
    word2idx = build_word2idx(texts, min_freq=MIN_FREQ)

    # Load FastText
    ft = fasttext.load_model(str(FASTTEXT_BIN))

    # Build embedding
    embedding = build_embedding(ft, word2idx, EMBED_DIM)

    # Save
    torch.save(
        {
            "embedding": embedding,
            "word2idx": word2idx,
        },
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
