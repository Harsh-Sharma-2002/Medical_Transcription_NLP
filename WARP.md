# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Setup and environment

- Python project with dependencies listed in `requirements.txt` (pandas, numpy, scikit-learn, imbalanced-learn, nltk, matplotlib, seaborn).
- RoBERTa embeddings are generated via Hugging Face `roberta-base` (loaded inside the preprocessing/embedding logic). The first run will download model weights and tokenizer.
- Recommended setup:
  - Create a virtual environment (example):
    - `python -m venv .venv`
    - `source .venv/bin/activate` (macOS/Linux)
  - Install dependencies:
    - `pip install -r requirements.txt`

## Common commands

> All commands below assume the working directory is the repository root.

- **Install dependencies**
  - `pip install -r requirements.txt`

- **Run the full training & evaluation pipeline**
  - `python main.py`
  - Expects a `Data.csv` file in the project root with at least:
    - `text`: raw medical transcription text
    - `label`: clinical specialty label
  - Outputs:
    - `classification_report.txt` – best hyperparameters and detailed metrics
    - `confusion_matrix.png` – confusion matrix plot
    - `best_roberta_rf_model.pkl` – persisted best model (GridSearchCV wrapper) via `joblib.dump`

- **Tests / linting**
  - There is currently **no test suite** or dedicated linting/formatting configuration checked into this repository.
  - If you introduce tests (e.g., with `pytest`), typical usage patterns will be:
    - Run all tests: `pytest`
    - Run a single test file: `pytest path/to/test_file.py`
    - Run a single test: `pytest path/to/test_file.py::TestClass::test_name`

## High-level architecture

The project implements a medical transcription **specialty classifier** using **RoBERTa embeddings + Random Forest** with class balancing via random oversampling.

### Conceptual pipeline

1. **Input data**
   - `Data.csv` contains free-text medical transcriptions and their associated clinical specialty labels.
2. **Embedding & class selection (preprocessing layer)**
   - A RoBERTa-based embedder converts each transcription into a dense vector representation.
   - Only the top-*k* most frequent classes (default `top_k=5`) are retained.
   - Class imbalance is addressed using `RandomOverSampler` to oversample minority classes after embedding.
3. **Train/validation split & evaluation utilities (training utilities layer)**
   - Data is split into train/test sets with stratification over labels.
   - Helpers exist for computing and plotting classification reports and confusion matrices.
4. **Model definition & hyperparameter search (model layer)**
   - A `RandomForestClassifier` is wrapped in `GridSearchCV` over a predefined hyperparameter grid.
   - Cross-validation scoring metric defaults to accuracy.
5. **Orchestration (pipeline entrypoint)**
   - The main script orchestrates:
     - Reading the CSV
     - Generating embeddings & oversampled labels
     - Plotting class distributions before/after oversampling
     - Splitting data
     - Running the grid search
     - Printing/saving the classification report
     - Plotting/saving the confusion matrix
     - Saving the fitted best model to disk

### Key modules (big-picture roles)

- `Preprocess.py`
  - Defines the **RoBERTaEmbedder** abstraction responsible for:
    - Tokenizing input texts with `RobertaTokenizer` (Hugging Face).
    - Running `RobertaModel` in evaluation mode and extracting embeddings (mean pooling over token representations).
  - Provides data-level utilities such as:
    - Selecting the top-*k* most frequent label classes.
    - Generating embeddings for those examples.
    - Encoding labels using `LabelEncoder` and applying `RandomOverSampler` to balance the dataset.
  - Includes visualization logic to compare class distributions before and after oversampling.

- `traint.py`
  - Houses **training and evaluation utilities** rather than the model definition itself:
    - `split_data` – stratified train/test split over embeddings and labels.
    - `train_model` – fits either a raw estimator or a `GridSearchCV` wrapper and returns the fitted object.
    - `evaluate_model` – convenience wrapper for computing and printing a classification report (optionally decoding labels).
    - `plot_confusion_matrix` – builds and saves a confusion matrix visualization.

- `Model.py`
  - Encapsulates **model configuration** and hyperparameter search:
    - `get_param_grid` – returns the hyperparameter grid for the Random Forest (n_estimators, max_depth, min_samples_split, etc.).
    - `get_grid_search_model` – wraps `RandomForestClassifier` in `GridSearchCV` with configurable CV folds, scoring metric, and parallelism.

- `main.py`
  - Acts as the **single entrypoint** for running the end-to-end experiment:
    - Loads `Data.csv`.
    - Calls preprocessing utilities to generate RoBERTa-based embeddings and oversampled labels.
    - Plots pre/post-oversampling class distributions.
    - Splits data into train/test.
    - Instantiates the grid-search model from `Model.py`.
    - Trains it and obtains the best estimator.
    - Evaluates performance, prints and saves a detailed classification report.
    - Plots and saves the confusion matrix.
    - Serializes the trained model to `best_roberta_rf_model.pkl`.

### File naming and imports

- The README and code refer to logical modules like `preprocessing`, `train`, and `model`.
- In the current repository snapshot, the corresponding files are named:
  - `Preprocess.py` (preprocessing + embedding + oversampling + plotting)
  - `traint.py` (training utilities & evaluation helpers)
  - `Model.py` (Random Forest + GridSearchCV configuration)
- If you encounter import errors (e.g., `from preprocessing import ...` or `from train import ...`), align the module names by either:
  - Renaming the files to match the imports, or
  - Updating the import statements to match the actual filenames.

## Notes for future modifications

- The current pipeline trains from scratch each time `main.py` is run, including recomputing RoBERTa embeddings. If you extend the project, consider:
  - Caching embeddings to disk for reuse across runs.
  - Adding a separate inference script that loads `best_roberta_rf_model.pkl` and `LabelEncoder` to classify new transcriptions without retraining.
- Before adding new CLI entrypoints or scripts, keep the existing single-entrypoint design (`main.py`) in mind to avoid duplicating orchestration logic.
