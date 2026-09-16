from src.data_loader import load_data
from src.preprocessing import preprocess_data


def test_preprocess_data():
    X, y, _, _ = load_data()

    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    assert X_train.shape[0] == 120
    assert X_test.shape[0] == 30
    assert X_train.shape[1] == 4
    assert X_test.shape[1] == 4
    assert len(y_train) == 120
    assert len(y_test) == 30