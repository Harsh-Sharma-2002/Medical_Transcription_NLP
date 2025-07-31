# 🩺 Medical Text Classification using RoBERTa + Random Forest

This project aims to **automatically classify medical transcriptions into their respective clinical specialties** using a combination of **transformer-based embeddings** (RoBERTa) and a **Random Forest classifier**. Medical transcription classification is crucial for organizing electronic health records, automating workflow, and aiding clinical decision support systems.

---

## 🧠 Project Highlights

- **Transformer-based Text Embedding**: We leverage the pre-trained `roberta-base` model from Hugging Face to convert raw medical text into high-dimensional semantic embeddings.
- **Traditional ML Classifier**: Instead of using deep neural classifiers, we apply a **Random Forest** model for classification—balancing performance with interpretability and efficiency.
- **Imbalanced Class Handling**: Healthcare datasets often suffer from class imbalance. We apply **Random Oversampling** to augment minority classes during training.
- **Performance Visualization**: We visualize class distributions, confusion matrix, and other evaluation metrics to better understand model behavior.

---
## 🧪 Problem Statement

Given a corpus of medical transcriptions (free-form text), the goal is to classify each document into one of several **clinical specialties**, such as cardiology, neurology, radiology, orthopedics, etc.

Accurate and **automated classification of medical text** is a critical task in modern healthcare systems, with wide-reaching practical benefits:

- **Improved Electronic Health Record (EHR) indexing**: Categorizing transcriptions by specialty enables faster data retrieval, efficient organization, and enhanced interoperability between systems.
- **Automated document routing**: Clinical notes can be directed to the appropriate department or specialist without manual intervention, improving hospital workflow and reducing administrative load.
- **Faster insurance claim processing**: Specialty-level classification helps insurance systems rapidly identify relevant documentation, accelerating billing, claims approval, and reimbursement workflows.
- **Enhanced Clinical Decision Support (CDS)**: Structured inputs empower AI systems for smarter triage, risk stratification, and diagnosis assistance.
- **Foundation for advanced NLP tasks**: Specialty-labeled data improves the accuracy of downstream models for **ICD coding**, **medical named entity recognition (NER)**, **clinical summarization**, and **compliance auditing**.

By automating this process, we enable **faster clinical workflows**, **reduced costs**, and **better patient outcomes**—while unlocking valuable insights from previously unstructured medical text.


---

## 🧰 Technologies Used

| Component                  | Tool / Library                         |
|---------------------------|----------------------------------------|
| Text Embedding            | 🤗 Hugging Face `roberta-base`         |
| Classifier                | 🌲 Scikit-learn `RandomForestClassifier` |
| Class Balancing           | `imblearn.over_sampling.RandomOverSampler` |
| Evaluation Metrics        | `sklearn.metrics`                      |
| Visualizations            | Matplotlib, Seaborn                    |
| Environment               | Python 3.x                             |

---

---

## 🚀 Workflow Overview

### 1. Data Preprocessing (`preprocessing.py`)
- Loads and cleans the transcription dataset
- Handles missing values and unwanted columns
- Tokenizes and prepares data for embedding

### 2. Embedding with RoBERTa (`embedding.py`)
- Uses Hugging Face’s `roberta-base` model
- Extracts CLS token or mean pooled embeddings for each transcription
- Stores embeddings for reuse to save compute time

### 3. Random Oversampling (`main.py`)
- Balances the dataset using `RandomOverSampler` from `imblearn`
- Ensures the Random Forest model sees sufficient examples from minority classes

### 4. Model Training (`model.py`)
- Trains a Random Forest with hyperparameter tuning via `GridSearchCV`
- Evaluates using precision, recall, F1-score, and confusion matrix
- Saves the best model for deployment/inference

### 5. Visualization (`main.py`)
- Plots class distribution before/after oversampling
- Generates confusion matrix and per-class performance metrics

---

## 📈 Results & Observations

- **RoBERTa Embeddings Boost Accuracy**: The use of pre-trained RoBERTa embeddings led to a significant performance improvement over traditional methods like bag-of-words or TF-IDF, thanks to its ability to capture rich semantic context from clinical text.
- **Random Forest Advantage**: Combining these embeddings with a Random Forest classifier yielded strong results—**achieving 86% accuracy** on the test set—while maintaining faster training time and interpretability.
- **Oversampling Impact**: Random oversampling effectively addressed class imbalance, improving recall on minority specialties without sacrificing overall performance.

---



