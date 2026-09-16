from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import matplotlib.pyplot as plt


def evaluate_model(y_test, predictions, target_names):
    """
    Evaluate the classification model and save the confusion matrix.
    """

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    cm = confusion_matrix(y_test, predictions)

    # Create confusion matrix image
    plt.figure(figsize=(7, 6))

    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix - KNN Classifier")
    plt.colorbar()

    plt.xticks(
        range(len(target_names)),
        target_names,
        rotation=45
    )

    plt.yticks(
        range(len(target_names)),
        target_names
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Add numbers inside the matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    # Save the image
    plt.savefig(
        "results/confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }