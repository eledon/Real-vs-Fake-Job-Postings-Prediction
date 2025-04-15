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
- [Results and Evaluation](#-results-and-evaluation)
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
- **Word clouds** revealed distinct linguistic patterns between real and fake postings. Fake postings use more generic or salesy terms ("opportunity," "apply," "hiring"), while real ones use more technical or specific job-related vocabulary.

**Word Cloud Visual:**
<img src="plots/word_clouds.png" width="500"/>

---

## 🧹 Text Preprocessing

- Custom stopword handling and lemmatization via spaCy
- Word count computed as additional feature
- TF-IDF (1- and 2-grams) using top 5,000 features

---

## ⚙️ Modeling

We trained and compared three models to ensure robustness and representativeness. This aligns with data science best practices, where multiple models are evaluated before final selection. Each model was chosen for its unique learning mechanism:

- **Logistic Regression**: Interpretable baseline model using linear decision boundaries.
- **Support Vector Machine (SVM)**: Optimizes class separation by maximizing the margin.
- **XGBoost**: A powerful tree-based ensemble capable of capturing nonlinearities and feature interactions.

---

## 📊 Results and Evaluation

### ✏️ Confusion Matrices

These illustrate classification performance on the test set:

<div align="center">
  <img src="plots/conf_matrix_lr.png" width="250"/>
  <img src="plots/conf_matrix_svm.png" width="250"/>
  <img src="plots/conf_matrix_xgb.png" width="250"/>
</div>

### 🔄 Performance Metrics

| Metric             | Logistic Regression | SVM (LinearSVC) | XGBoost |
|--------------------|---------------------|------------------|---------|
| Accuracy           | 98%                 | 96%              | 96%     |
| Fake Precision     | 98%                 | 98%              | 100%    |
| Fake Recall        | 64%                 | 59%              | 55%     |
| Fake F1-score      | 0.77                | 0.74             | 0.71    |
| Avg. Precision (Test) | 0.85             | 0.83             | 0.67    |

### 🔢 Precision-Recall Curves

These curves help visualize trade-offs between precision and recall at different thresholds. Logistic Regression consistently showed the best balance.

<div align="center">
  <img src="plots/prec_recall_lr.png" width="300"/>
  <img src="plots/prec_recall_svm.png" width="300"/>
  <img src="plots/prec_recall_xgb.png" width="300"/>
</div>

---

## 🧠 Explainability

SHAP analysis was applied to all models to interpret individual feature contributions:

- **SHAP values** represent the impact each feature has on the prediction (positive = more fake, negative = more real).
- In Logistic Regression and SVM, `word_count`, `team`, `look`, and `user` were highly influential.
- Visuals below show how individual feature values affect predictions.

<div align="center">
  <img src="plots/shap_lr.png" width="300"/>
  <img src="plots/shap_svm.png" width="300"/>
</div>

### ✍ Interpretation (from report):
- SHAP values confirmed that longer job descriptions tend to be real.
- Generic marketing language and brevity were strong indicators of fake postings.
- Logistic Regression provided the clearest and most interpretable decision boundaries.

---

## ✅ Conclusion

- Logistic Regression emerged as the best-performing model overall in terms of F1-score and balance between precision and recall.
- SVM was close and offered good generalization, while XGBoost struggled with recall despite having perfect precision.
- SHAP explainability was key to understanding model decisions and validating linguistic intuition.
- Word count and specific vocabulary patterns are reliable indicators of job posting authenticity.
- Multiple models were evaluated to ensure result robustness, confirming that a simpler linear model (Logistic Regression) is both effective and interpretable for this NLP classification task.

---

📄 **Notebook & Report**:  
- [Full notebook](Real_Fake_Job_Postings_update.ipynb)  
- [Report (PDF)](Real_Fale_Job_Postings_Report.pdf)

