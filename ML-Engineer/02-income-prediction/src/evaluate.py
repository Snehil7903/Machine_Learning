from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

OUTPUT_DIR = Path("outputs")


def evaluate_model(model, X_test, y_test, model_name):

    OUTPUT_DIR.mkdir(exist_ok=True)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred)

    print("=" * 60)
    print(f"{model_name.upper()} RESULTS")
    print("=" * 60)

    print(f"Test Accuracy : {accuracy:.4f}\n")

    print(report)

    with open(
        OUTPUT_DIR / f"{model_name}_metrics.txt",
        "w",
    ) as f:
        f.write(f"Accuracy : {accuracy:.4f}\n")

    with open(
        OUTPUT_DIR / f"{model_name}_classification_report.txt",
        "w",
    ) as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot(cmap="Blues")

    plt.title(f"{model_name} Confusion Matrix")

    plt.savefig(
        OUTPUT_DIR / f"{model_name}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nResults saved inside outputs/")