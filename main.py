import pandas as pd
import joblib
from preprocessing import (
    select_top_classes_and_oversample,
    plot_class_distributions
)
from train import (
    split_data,
    train_model,
    evaluate_model,
    plot_confusion_matrix
)
from model import get_grid_search_model
from sklearn.metrics import classification_report

#  Load dataset
df = pd.read_csv("Data.csv")  

embeddings, labels, label_encoder = select_top_classes_and_oversample(
    df,
    text_col='text',
    label_col='label',
    top_k=5,
    return_label_encoder=True
)

plot_class_distributions(
    df=df,
    label_col='label',
    resampled_labels=labels,
    top_k=5
)

X_train, X_test, y_train, y_test = split_data(embeddings, labels)

model = get_grid_search_model(cv=3, scoring='accuracy')
best_model = train_model(model, X_train, y_train)


print("Best Hyperparameters:\n", best_model.best_params_)
y_pred = best_model.predict(X_test)


target_names = label_encoder.inverse_transform(sorted(set(y_test)))
report = classification_report(y_test, y_pred, target_names=target_names)
print("\nClassification Report:\n", report)

# Save classification report to file
with open("classification_report.txt", "w") as f:
    f.write("Best Parameters:\n")
    f.write(str(best_model.best_params_))
    f.write("\n\nClassification Report:\n")
    f.write(report)

# Confusion matrix
plot_confusion_matrix(y_test, y_pred, class_names=target_names)

# Save trained model
joblib.dump(best_model, "best_roberta_rf_model.pkl")
print(" Model saved as best_roberta_rf_model.pkl")

