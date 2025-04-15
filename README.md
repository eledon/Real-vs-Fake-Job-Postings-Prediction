# 💼 Real vs Fake Job Postings Classification

Classifying job postings as real or fraudulent using NLP techniques and machine learning models including **Logistic Regression**, **SVM**, and **XGBoost**.

<img src="plots/pexels-ron-lach-9832718.jpg" width="500" height="300"/>

![Python](https://img.shields.io/badge/Python-TextProcessing-blue?logo=python)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Model](https://img.shields.io/badge/Model-Logistic%20Regression%20%7C%20SVM%20%7C%20XGBoost-yellowgreen)
![Data](https://img.shields.io/badge/Data-Kaggle-orange)

---

## 📘 Table of Contents
- [Overview](#-overview)
- [Technologies](#-technologies)
- [Research Question](#-research-question)
- [Dataset](#-dataset)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Text Preprocessing](#-text-preprocessing)
- [Modeling](#-modeling)
- [Model Evaluation](#-model-evaluation)
- [Explainability](#-explainability)
- [Conclusion](#-conclusion)

---

## 🧱 Overview

This project investigates the classification of job postings into **real vs. fake** categories using machine learning and natural language processing (NLP). The goal is to help platforms and users identify fraudulent job offers by leveraging structured and unstructured data from job listings.

---

## 🧪 Technologies

- **Language:** Python  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `wordcloud`, `spacy`, `shap`  
- **Vectorization:** TF-IDF (unigrams and bigrams)  
- **Explainability:** SHAP for XGBoost, Logistic Regression, and LinearSVC

---

## ❓ Research Question

> Can we reliably identify fake job postings based on the job description and requirements using NLP features and machine learning classifiers?

---

## 📊 Dataset

- **Source:** [Kaggle - Fake Job Postings Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Size:** 17,880 job postings
- **Label:** `fraudulent` (0 = Real, 1 = Fake)
- **Key Features Used:** `description`, `requirements`, `telecommuting`, `has_company_logo`, `has_questions`, `employment_type`, `industry`

---

## 🔍 Exploratory Data Analysis

- **Only 4.8%** of job postings are fraudulent — heavy class imbalance.
- Real postings tend to have **longer descriptions**, while fake postings often repeat generic content.

**Visuals:**

- ![Confusion Matrix - LR](plots/conf_matrix_lr.png)
- ![Confusion Matrix - SVM](plots/conf_matrix_svm.png)
- ![Confusion Matrix - XGBoost](plots/conf_matrix_xgb.png)
- ![Word Clouds](plots/word_clouds.png)

---

## 🧹 Text Preprocessing

- Custom stopword handling and lemmatization via spaCy
- Word count computed as additional feature
- TF-IDF (1- and 2-grams) using top 5,000 features

---

## ⚙️ Modeling

We trained and compared three models:

### 🔹 Logistic Regression
- Tuned `C` parameter, balanced class weights
- Accuracy: **98%**, Fake F1: **0.77**

### 🔹 Support Vector Machine (SVM)
- Linear SVM with balanced class weights
- Accuracy: **96%**, Fake F1: **0.74**

### 🔹 XGBoost
- Tuned with `RandomizedSearchCV`
- Accuracy: **96%**, Fake F1: **0.71**

---

## 📈 Model Evaluation

| Metric             | Logistic Regression | SVM (LinearSVC) | XGBoost |
|--------------------|---------------------|------------------|---------|
| Accuracy           | 98%                 | 96%              | 96%     |
| Fake Precision     | 98%                 | 98%              | 100%    |
| Fake Recall        | 64%                 | 59%              | 55%     |
| Fake F1-score      | 0.77                | 0.74             | 0.71    |
| Avg. Precision (Test) | 0.85             | 0.83             | 0.67    |

**Precision-Recall Curves:**
- ![PR - LR](plots/prec_recall_lr.png)
- ![PR - SVM](plots/prec_recall_svm.png)
- ![PR - XGB](plots/prec_recall_xgb.png)

---

## 🧠 Explainability

We used SHAP to explain model decisions. SHAP values represent the **contribution** of each feature to the prediction:

- Positive SHAP → pushes toward "Fake"
- Negative SHAP → pushes toward "Real"

**Top Influential Words:** `team`, `look`, `user`, `client`, `marketing`, `responsibility`, `word_count`

**SHAP Summary Plots:**
- ![SHAP - Logistic Regression](plots/shap_lr.png)
- ![SHAP - SVM](plots/shap_svm.png)

---

## ✅ Conclusion

- Logistic Regression performed best in terms of **F1-score**, **precision-recall tradeoff**, and **interpretability**.
- SVM was close and confirmed model stability.
- XGBoost achieved high precision but struggled with recall despite tuning.
- SHAP explainability provided insights into **language patterns of fake vs. real** postings.
- **Word count and specific words** were highly indicative of job posting authenticity.

---

✉️ **Notebook & Report**:  
- [Full notebook](Real_Fake_Job_Postings_update.ipynb)  
- [Report (PDF)](Real_Fale_Job_Postings_Report.pdf)
