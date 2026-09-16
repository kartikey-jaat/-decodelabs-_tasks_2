from sklearn.neighbors import KNeighborsClassifier


def train_model(X_train, y_train, k=5):
    """
    Train a K-Nearest Neighbors classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        k: Number of neighbors

    Returns:
        Trained KNN model
    """

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train, y_train)

    return model


def make_predictions(model, X_test):
    """
    Generate predictions using the trained model.

    Args:
        model: Trained KNN model
        X_test: Testing features

    Returns:
        Predicted labels
    """

    predictions = model.predict(X_test)

    return predictions