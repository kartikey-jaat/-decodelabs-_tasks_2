from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(X, y):
    """
    Split the dataset into training and testing sets
    and standardize the features.

    Returns:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled testing features
        y_train: Training labels
        y_test: Testing labels
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test