from sklearn.datasets import load_iris


def load_data():
    """
    Load the Iris dataset.

    Returns:
        X: Feature data
        y: Target labels
        feature_names: Names of the features
        target_names: Names of the classes
    """

    iris = load_iris()

    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    return X, y, feature_names, target_names