import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the data into training and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_model(model, X_train, y_train):
    """
    Fits the model on training data and returns the best estimator.
    Works with both GridSearchCV and standard estimators.
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, label_encoder=None):
    """
    Evaluates the model using classification report and prints results.
    """
    y_pred = model.predict(X_test)

    if label_encoder:
        target_names = label_encoder.inverse_transform(sorted(set(y_test)))
    else:
        target_names = None

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=target_names))


def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix"):
    """
    Plots and saves a confusion matrix image.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation='vertical')
    plt.title(title)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()
