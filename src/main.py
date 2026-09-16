from data_loader import load_data
from preprocessing import preprocess_data
from model import train_model, make_predictions
from evaluation import evaluate_model


def main():
    # Load dataset
    X, y, feature_names, target_names = load_data()

    print("=== Data Classification Using AI ===")
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Classes: {target_names}")

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Train KNN model
    model = train_model(X_train, y_train, k=5)

    # Make predictions
    predictions = make_predictions(model, X_test)

    # Evaluate model
    results = evaluate_model(y_test, predictions, target_names)

    print("\n=== Model Evaluation ===")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1_score']:.4f}")

    print("\n=== Confusion Matrix ===")
    print(results["confusion_matrix"])


if __name__ == "__main__":
    main()