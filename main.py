import pandas as pd
from preprocessing import (
    select_top_classes_and_oversample,
    plot_class_distributions
)
from train import split_data, train_model, evaluate_model
from model import get_grid_search_model
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("your_transcriptions.csv")

embeddings, labels, label_encoder = select_top_classes_and_oversample(
    df,
    text_col='text',
    label_col='label',
    top_k=5,
    return_label_encoder=True
)

plot_class_distributions(
    df,
    label_col='label',
    resampled_labels=labels,
    top_k=5
)

X_train, X_test, y_train, y_test = split_data(embeddings, labels)

model = get_grid_search_model()

best_model = train_model(model, X_train, y_train)

evaluate_model(best_model, X_test, y_test, label_encoder)
