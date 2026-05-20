# 📰 Fake News Detection using Machine Learning

> A comparative study of classical supervised machine learning algorithms for fake news detection using the ISOT dataset — achieving **99.9% accuracy** with a Passive-Aggressive Classifier on TF-IDF Uni+Bigram features.

---

## 👤 Team Details

| Field | Details |
|---|---|
| **Name** | Akshat Mittal |
| **Roll No.** | 2210991228 |
| **Institution** | Chitkara University, Rajpura, Punjab, India |
| **Email** | akshat1228.be22@chitkara.edu.in |

---

## 📋 Project Overview

This project investigates whether classical supervised machine learning — focused on simple linguistic patterns — can reliably detect fake news articles at scale. Five popular classifiers are benchmarked across four TF-IDF/BoW vectorization schemes on the **ISOT Fake News Dataset** (44,898 articles) under fully controlled, reproducible experimental conditions.

The **Passive-Aggressive Classifier** combined with **TF-IDF Unigram + Bigram** features delivers standout results: **99.9% accuracy** and an **F1-score of 0.999** — training in approximately **0.2 seconds** — matching or exceeding fine-tuned transformer benchmarks at a fraction of the computational cost.

---

## 📄 IPR Deliverable

| Field | Details |
|---|---|
| **Deliverable Type** | Research Paper |
| **Title** | *Fake News Detection using Machine Learning: A Comparative Study of Classical Supervised Learning Algorithms* |
| **Current Status** | ✅ Submitted to Journal |

---

## 🗂️ Repository Structure

```
Fake-News-Detection-2210991228/
│
├── IPR_Submission_Proof/       # Journal submission confirmation & proof of submission
│
├── PPT_and_Report/             # Presentation slides and full research paper/report
│
├── Source_Code/                # Jupyter notebook with all experiments and analysis
│
└── README.md                   # You are here
```

---

## 🔬 Research Summary

### Dataset
- **Name:** ISOT Fake News Dataset (University of Victoria)
- **Size:** 44,898 articles — 23,481 fake + 21,417 real (≈52/48 split)
- **Source:** Real articles from Reuters; fake articles from sources flagged by PolitiFact & Wikipedia
- **Coverage:** U.S. political and international news, 2015–2017

### Classifiers Evaluated
| Classifier | Accuracy | F1-Score | Train Time |
|---|---|---|---|
| Logistic Regression | 99.7% | 0.997 | 3.8s |
| Support Vector Machine | 99.8% | 0.998 | 1.1s |
| Random Forest | 99.7% | 0.997 | 138.5s |
| Naïve Bayes | 96.3% | 0.963 | 0.1s |
| **Passive-Aggressive (PAC)** ⭐ | **99.9%** | **0.999** | **0.2s** |

*All results under TF-IDF Unigram + Bigram vectorization on a held-out test set of 8,980 articles.*

### Vectorization Schemes Compared
| Scheme | Best Accuracy | Feature Dimensions |
|---|---|---|
| BoW Count — Unigrams | 99.6% | 58,420 |
| TF-IDF — Unigrams only | 99.8% | 58,420 |
| TF-IDF — Bigrams only | 99.4% | 311,680 |
| **TF-IDF — Unigrams + Bigrams** ⭐ | **99.9%** | **370,100** |

### Key Findings
- **PAC is Pareto-optimal** — highest accuracy at the lowest meaningful training time among competitive classifiers.
- **Vectorization scheme matters as much as classifier choice** — the performance gap across schemes (0.5 pp) is larger than the gap among the top 4 classifiers (0.2 pp).
- **Bigrams add meaningful signal** — sequences like `washington reuters`, `york reuters`, and `featured image` are not captured by unigrams alone.
- **Feature interpretability reveals a clear rhetorical divide** — real news is grounded in source-attribution language (`reuters`, `said`, datelines); fake news is media-embedding and politically charged (`via`, `video`, `breaking`, `obama`).

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python |
| Data Handling | pandas, NumPy |
| NLP & Preprocessing | NLTK (tokenization, stopwords, lemmatization) |
| Feature Engineering | scikit-learn (TF-IDF, CountVectorizer) |
| Classifiers | scikit-learn (LR, LinearSVC, RF, MultinomialNB, PAC) |
| Evaluation | scikit-learn (accuracy, precision, recall, F1, confusion matrix) |
| Visualisation | Matplotlib, Seaborn, WordCloud |
| Notebook | Jupyter Notebook |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install scikit-learn nltk pandas numpy matplotlib seaborn wordcloud tqdm
```

### Dataset Setup
Download the **ISOT Fake News Dataset** from Kaggle:
[https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

Place `Fake.csv` and `True.csv` in the same directory as the notebook.

### Running the Notebook
```bash
cd Source_Code/
jupyter notebook Fake_News_Detection_Experiments.ipynb
```
Run all cells sequentially — the notebook is structured in 7 phases from environment setup through to feature interpretability analysis.

---

## 📚 References

1. Ahmed, H., Traore, I., & Saad, S. — ISOT Fake News Dataset benchmark (original TF-IDF baseline).
2. Wang, W. Y. — LIAR Dataset: multi-class truthfulness classification.
3. Sharma, K. et al. — Cross-classifier SVM comparison on sparse TF-IDF spaces.
4. Devlin, J. et al. — BERT: Pre-training of deep bidirectional transformers.
5. Kaliyar, R. K. et al. — FakeBERT: Fake news detection in social media with a BERT-based deep learning approach.
6. Monti, F. et al. — Fake news detection on social media using geometric deep learning.

---

## ⚠️ Limitations

- Results are specific to the ISOT corpus (U.S. political news, 2015–2017) and may not generalize to other domains or languages.
- Vocabulary-based classifiers may be vulnerable to adversarial evasion if feature weights are exposed.
- Cross-paper accuracy comparisons with transformer models are indicative only; a controlled head-to-head benchmark is a identified future direction.

---

## 📃 License

This repository is submitted for academic purposes under Chitkara University's IPR submission framework. The ISOT Fake News Dataset is publicly available via the University of Victoria.
