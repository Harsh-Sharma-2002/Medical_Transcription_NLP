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

Given a corpus of medical transcriptions (free-form text), classify each document into one of several **clinical specialties**, such as cardiology, neurology, radiology, etc.

Such classification can:
- Improve EHR (Electronic Health Record) indexing
- Help route documents to the right departments
- Serve as a preprocessing step for downstream tasks like ICD coding or named entity recognition

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

- **Impact of RoBERTa**: Pre-trained embeddings significantly improve classification over bag-of-words or TF-IDF approaches by capturing deeper semantic context.
- **Random Forest Advantage**: Faster training, easier interpretability, and less overfitting compared to deep classifiers with small data.
- **Oversampling**: Mitigates skew from dominant classes and improves recall on underrepresented specialties.

---



