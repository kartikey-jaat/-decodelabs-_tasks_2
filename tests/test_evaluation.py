from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import train_model, make_predictions
from src.evaluation import evaluate_model


def test_evaluation():
    X, y, _, target_names = load_data()

    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    model = train_model(X_train, y_train, k=5)

    predictions = make_predictions(model, X_test)

    results = evaluate_model(
        y_test,
        predictions,
        target_names
    )

    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1_score" in results
    assert "confusion_matrix" in results