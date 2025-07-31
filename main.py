import pandas as pd
from sklearn.preprocessing import LabelEncoder
from preprocessing import preprocess_dataframe, balance_and_plot
from model import get_grid_search_model
from train import split_data, train_model, evaluate_model

def main():
    # Load dataset
    df = pd.read_csv("your_transcriptions.csv")  # Replace with your actual CSV path

    # Preprocess text and labels
    X, y, label_encoder = preprocess_dataframe(df, text_column='text', label_column='label')

    # Undersample and oversample top 5 classes, visualize distributions
    X_balanced, y_balanced = balance_and_plot(X, y, label_encoder)

    # Train/test split
    X_train, X_test, y_train, y_test = split_data(X_balanced, y_balanced)

    # Initialize model with grid search
    model = get_grid_search_model()

    # Train model and get best estimator
    best_model = train_model(model, X_train, y_train)

    # Evaluate
    evaluate_model(best_model, X_test, y_test, label_encoder)

if __name__ == "__main__":
    main()
