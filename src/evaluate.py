import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def main():
    test_data = np.load("data/processed/test.npz")

    x_test = test_data["images"]
    y_test = test_data["labels"]

    model = tf.keras.models.load_model("models/model.h5")

    loss, accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    predictions = model.predict(
        x_test,
        verbose=0
    )

    y_pred = np.argmax(predictions, axis=1)

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()
    plt.title("Fashion-MNIST Confusion Matrix")
    plt.savefig("confusion_matrix.png", bbox_inches="tight")
    plt.close()

    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy)
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("Metrics saved to metrics.json")
    print("Confusion matrix saved to confusion_matrix.png")


if __name__ == "__main__":
    main()