# Sentiment-Based Product Recommendation System

An end-to-end machine learning project that combines collaborative filtering with sentiment analysis to deliver product recommendations that are both relevant and well-reviewed. Built as the capstone for the MSc Data Science & AI programme.

## Overview

Traditional recommendation systems suggest products based on what similar users have liked. However, ratings alone do not tell the full story — a product with a high average rating can still attract a significant number of negative reviews. This project addresses that gap by adding a sentiment layer on top of collaborative filtering.

The system first generates a set of candidate products for a given user through collaborative filtering, then applies a sentiment classifier to the reviews of those candidates, returning only the products with the highest proportion of positive sentiment.

## Dataset

The dataset (`sample30.csv`) contains approximately 30,000 product reviews from the e-commerce platform Ebuss, spanning multiple product categories including household essentials, personal care, and health products. Each record includes the review text, rating, reviewer username, product name, and a pre-labelled sentiment.

The target variable (`user_sentiment`) is imbalanced, with roughly 89% positive and 11% negative reviews.

## Methodology

| Stage | Description |
|-------|-------------|
| Data Cleaning | Removed sparse and non-informative columns, treated missing values, and encoded the sentiment target |
| Text Processing | Applied lowercasing, URL and HTML removal, punctuation stripping, stopword removal, and lemmatisation |
| Feature Extraction | Converted text to TF-IDF features using unigrams and bigrams; applied a stratified train-test split |
| Sentiment Modelling | Trained Logistic Regression, Naive Bayes, and Random Forest classifiers with class balancing; selected the best by weighted F1-score |
| Recommendation System | Built user-based and item-based collaborative filtering using adjusted cosine similarity; selected the system with the lower RMSE |
| Top-20 Generation | Produced the top 20 candidate products for a specified user |
| Sentiment Filtering | Predicted sentiment across all reviews of each candidate and returned the top 5 products by positive sentiment percentage |

## Key Decisions

- **TF-IDF over Bag-of-Words:** TF-IDF down-weights frequent but uninformative terms, improving classifier performance on sparse review text.
- **Class imbalance handling:** Models use `class_weight='balanced'` to correct for the 89/11 distribution without resampling.
- **Model selection by weighted F1:** Accuracy alone is misleading under class imbalance; weighted F1 provides a more reliable comparison.
- **Recommendation system selection by RMSE:** Both collaborative filtering approaches were evaluated on held-out ratings, and the lower-error system was retained.

## Repository Structure

```
.
├── Sentiment_Based_Product_Recommendation_Arun_K.ipynb   # Full project notebook
├── sample30.csv                                          # Dataset
└── README.md
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
nltk
scikit-learn
```

## Usage

1. Clone the repository.
2. Ensure `sample30.csv` is in the same directory as the notebook.
3. Open and run `Sentiment_Based_Product_Recommendation_Arun_K.ipynb` end to end.

## Assumptions

In line with the project brief, the system assumes that no new users or products are introduced during prediction. It operates over the fixed set of users and products present in the dataset.

## Author

**Arun K** — MSc Data Science & AI
