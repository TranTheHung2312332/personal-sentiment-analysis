# Setup Guide

This document explains how to install dependencies, download FastText, and run `embedding_setup.py`.

---

## 1. Install Dependencies

Make sure Python (>= 3.8) is installed.

```bash
pip install pandas numpy torch fasttext tqdm
```

## 2. Download FastText Model

Run the following commands from the project root:

```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gunzip cc.en.300.bin.gz
```

### 3. Run the Script

Run the embedding setup script:
```bash
python script/setup/embedding_setup.py
```

### 4. Output

After completion, the following file will be created:

```
model/embedding.pt
```
