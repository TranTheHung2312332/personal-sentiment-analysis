# 🔍 Sentiment Analysis – End-to-End ML System

An **end-to-end sentiment analysis project** covering **data processing, model training, inference API, and interactive frontend visualization**.

The system combines **machine learning + rule-based logic** to produce **stable, interpretable sentiment predictions**.

---

## 🚀 Key Features

- **End-to-End ML Pipeline**
  - Data preprocessing → embedding → training → inference
- **Hybrid Decision Logic**
  - ML logits + **rule-based neutral fallback**
  - Improves robustness on low-confidence predictions
- **Explainable Outputs**
  - Cleaned text
  - Extra linguistic features:
    - `ex_intensity`
    - `emoji_score`
    - `all_uppercase`
    - `uppercase_ratio`
- **Interactive Frontend**
  - Toggle preprocessing steps
  - Adjust neutral threshold
  - Visualize probabilities & intermediate features

---

## 🧠 Model Overview

- **Task**: Sentiment Analysis (3-class)
- **Labels**:
  - `Positive`
  - `Neutral`
  - `Negative`
- **Embedding**: FastText (pretrained, not fine-tuned)
- **Model Output**: logits → softmax probabilities
- **Decision Strategy**:
  ```python
  if idxmax == 1 and probs[idxmax] < neutral_threshold:
      idxmax = 0 if probs[0] >= probs[2] else 2

---

## 🛠 Tech Stack

- **Backend**: FastAPI, PyTorch
- **Frontend**: React, TypeScript, Vite
- **ML**: FastText embedding, custom feature engineering

## 📦 Project Structure (Simplified)

- **ml/** – data processing, training, experiments
- **backend/** – inference pipeline & FastAPI service
- **frontend/** – interactive sentiment visualization UI

---

## ▶️ Quick Start

```
# Backend
pip install -r requirements.txt
python backend/main.py
```

```
# Frontend
cd frontend
npm install
npm run dev
```

---

## 🎯 Why This Project?

This project demonstrates:
 - Practical ML system design
 - Deployment-ready inference pipeline
 - Thoughtful hybrid ML + rule-based reasoning
 - Strong focus on interpretability & UX
Built as a personal project to showcase applied machine learning beyond model training.