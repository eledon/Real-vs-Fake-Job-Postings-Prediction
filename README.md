# 💼 Real vs Fake Job Postings Classification

This project investigates the classification of job postings into **real vs. fake** categories using Natural Language Processing (NLP) and machine learning. The aim is to support job platforms and users in identifying fraudulent job postings by analyzing both structured and unstructured data.

---

## 📦 Project Summary

We trained and evaluated three supervised learning models:

- **Logistic Regression** — a linear and interpretable baseline.
- **Support Vector Machine (SVM)** — a linear model optimized for margin-based class separation.
- **XGBoost** — a non-linear, ensemble-based classifier capable of capturing feature interactions.

While all three models performed well, Logistic Regression offered the best balance of predictive accuracy, recall, and interpretability. SVM achieved the highest AP score, and XGBoost delivered perfect precision but at the cost of lower recall.

---

## 📘 Table of Contents
- [Technologies](#technologies)
- [Research Questions](#research questions)
- [Dataset](#dataset)
- [EDA](#eda)
- [Text Preprocessing](#text-preprocessing)
- [Modeling and Evaluation](#modeling-and-evaluation)
- [Model Explainability](#model-explainability)
- [Conclusion](#conclusion)
- [Resources](#resources)

---

## ⚙️ Technologies

- Python
- pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn
- spaCy (lemmatization)
- SHAP (model explainability)
- TF-IDF vectorization

---

## ❓ Research Questions

This project investigates the following core questions:

1. **Can we accurately classify job postings as real or fake?**  
   Using natural language and structured features, can machine learning models reliably distinguish fraudulent listings?

2. **What patterns and features differentiate fake from real job posts?**  
   Does word count matter? Are there specific terms, tones, or structural traits that are more common in fake postings?

3. **How do the models make their predictions?**  
   By applying SHAP explainability techniques, can we understand which features most influence the model's decision and validate that the logic aligns with human intuition?

---

## 🗃 Dataset

- **Source**: [Kaggle - Fake Job Postings](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Size**: 17,880 job postings
- **Target Variable**: `fraudulent` (0 = Real, 1 = Fake)
- **Key Features Used**: `description`, `requirements`, `telecommuting`, `has_company_logo`, `has_questions`, `employment_type`, `industry`

---

## 📊 EDA

The dataset is highly imbalanced with only **4.8%** labeled as fake.

- **Text length**: Real job postings are generally longer.
- **Word clouds**: Fake postings use sales-like or vague language (e.g., "hiring," "opportunity") vs. real postings that include more specific job-related terms.

<img src="plots/word_clouds.png" width="500"/>

---

## 🧹 Text Preprocessing

- HTML unescaping and punctuation removal
- Custom stopword filtering
- Lemmatization with spaCy
- Word count features
- TF-IDF vectorization (unigrams and bigrams, top 5,000 terms)

---

## 🤖 Modeling and Evaluation

We trained **Logistic Regression**, **LinearSVC**, and **XGBoost** using the same feature set (TF-IDF + word count). All models were evaluated using test set metrics and precision-recall tradeoffs.

### 🔢 Performance Overview

| Metric             | Logistic Regression | SVM (LinearSVC) | XGBoost |
|--------------------|---------------------|------------------|---------|
| Accuracy           | 98%                 | 96%              | 96%     |
| Fake Precision     | 98%                 | 98%              | 100%    |
| Fake Recall        | 64%                 | 59%              | 55%     |
| Fake F1-score      | 0.77                | 0.74             | 0.71    |
| Avg. Precision (Test) | 0.83             | 0.85             | 0.67    |

### 📉 Precision-Recall Curves

We plotted PR curves for both validation and test datasets to assess generalization and overfitting. While SVM had the **highest Average Precision (AP)**, Logistic Regression was selected for its better overall **recall**, **balance**, and **interpretability**.

<img src="plots/prec_recall_lr.png" width="250"/>
<img src="plots/prec_recall_svm.png" width="250"/>
<img src="plots/prec_recall_xgb.png" width="250"/>

### 📊 Confusion Matrices

Confusion matrices help visualize classification errors:
- **TP**: correctly identified fake postings
- **FP**: real postings wrongly flagged as fake
- **FN**: fake postings missed by the model
- **TN**: correctly identified real postings

<img src="plots/conf_matrix_lr.png" width="250"/>
<img src="plots/conf_matrix_svm.png" width="250"/>
<img src="plots/conf_matrix_xgb.png" width="250"/>

---

## 🧠 Model Explainability

We used SHAP to understand how individual features influenced each model’s output.

- **SHAP value**: quantifies how much a feature contributes to a prediction.
- Features like `word_count`, `team`, `look`, and `user` showed high impact.
- Red points = high feature value; Blue = low.

### SHAP Summary Plots

<img src="plots/shap_lr.png" width="300"/>
<img src="plots/shap_svm.png" width="300"/>

**Interpretation highlights:**
- Logistic Regression favored longer, more descriptive job posts as real.
- Fake posts often had shorter content and used generic terms.
- SHAP plots validated linguistic patterns captured during EDA.

---

## ✅ Conclusion

- All models achieved high accuracy, but performance varied for the **minority class (fake)**.
- **Logistic Regression** offered the best combination of **recall**, **stability**, and **interpretability**.
- **SVM** slightly outperformed in AP but had lower recall.
- **XGBoost** overfit and yielded low recall despite perfect precision.
- SHAP analysis confirmed meaningful and intuitive linguistic patterns.

---

## 📎 Resources

- [Notebook](Real_Fake_Job_Postings_update.ipynb)
- [Report (PDF)](Real_Fale_Job_Postings_Report.pdf)
