from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import train_model, make_predictions


def test_model():
    X, y, _, _ = load_data()

    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    model = train_model(X_train, y_train, k=5)

    predictions = make_predictions(model, X_test)

    assert len(predictions) == len(y_test)