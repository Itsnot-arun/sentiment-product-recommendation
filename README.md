# 🛍️ Sentiment-Based Product Recommendation System

> An end-to-end machine learning system that blends **collaborative filtering** with **sentiment analysis** to recommend products that are both relevant *and* well-reviewed.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white" alt="sklearn">
  <img src="https://img.shields.io/badge/NLTK-NLP-green" alt="NLTK">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/Status-Complete-success" alt="Status">
</p>

**Final Capstone — MSc Data Science & AI**

---

## 📌 Overview

Traditional recommenders suggest products based on what similar users liked — but ratings alone don't tell the full story. A product with a high average rating can still carry a heavy load of negative reviews.

This project closes that gap by layering **sentiment analysis** on top of **collaborative filtering**:

```
User  →  Collaborative Filtering  →  Top 20 candidates  →  Sentiment Filter  →  Top 5 products
```

The result is a shortlist of products that are not only relevant to the user, but also genuinely well-received by the community.

---

## 📊 Dataset

`sample30.csv` — **~30,000 product reviews** from the e-commerce platform *Ebuss*, spanning household essentials, personal care, and health products.

| Attribute | Detail |
|-----------|--------|
| Records | ~30,000 reviews |
| Key fields | review text, rating, username, product name, sentiment |
| Target | `user_sentiment` (Positive / Negative) |
| Class balance | ⚖️ ~89% Positive · ~11% Negative |

---

## 🔧 Methodology

| # | Stage | Description |
|---|-------|-------------|
| 1️⃣ | **Data Cleaning** | Removed sparse and non-informative columns, treated missing values, encoded the sentiment target |
| 2️⃣ | **Text Processing** | Lowercasing, URL/HTML removal, punctuation stripping, stopword removal, lemmatisation |
| 3️⃣ | **Feature Extraction** | TF-IDF features with unigrams + bigrams; stratified train-test split |
| 4️⃣ | **Sentiment Modelling** | Logistic Regression, Naive Bayes & Random Forest with class balancing; best selected by weighted F1 |
| 5️⃣ | **Recommendation System** | User-based & item-based collaborative filtering (adjusted cosine); lower-RMSE system retained |
| 6️⃣ | **Top-20 Generation** | Produced top 20 candidate products for a specified user |
| 7️⃣ | **Sentiment Filtering** | Scored sentiment across each candidate's reviews; returned top 5 by positive % |

---

## 💡 Key Decisions

- **🎯 TF-IDF over Bag-of-Words** — down-weights frequent but uninformative terms, sharpening performance on sparse review text.
- **⚖️ Class imbalance handled** — `class_weight='balanced'` corrects the 89/11 split without resampling.
- **📈 Model selection by weighted F1** — accuracy is misleading under imbalance; weighted F1 gives a fairer comparison.
- **🔍 Recommender selection by RMSE** — both CF approaches evaluated on held-out ratings; the lower-error one wins.

---

## 📁 Repository Structure

```
.
├── 📓 Sentiment_Based_Product_Recommendation_Arun_K.ipynb   # Full project notebook
├── 📄 sample30.csv                                          # Dataset
└── 📘 README.md
```

---

## ⚙️ Requirements

```bash
pandas
numpy
matplotlib
seaborn
nltk
scikit-learn
```

---

## 🚀 Usage

```bash
# 1. Clone the repository
git clone https://github.com/Itsnot-arun/sentiment-product-recommendation.git

# 2. Keep sample30.csv in the same directory as the notebook

# 3. Launch and run the notebook end to end
jupyter notebook Sentiment_Based_Product_Recommendation_Arun_K.ipynb
```

---

## 📝 Assumptions

> The system assumes **no new users or products** are introduced during prediction. It operates over the fixed set of users and products present in the dataset.

---

## 👤 Author

**Arun K**
*MSc Data Science & AI*
