import os
import csv
import yaml
import numpy as np
import tensorflow as tf


def main():
    os.makedirs("models", exist_ok=True)

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    train_params = params["train"]

    train_data = np.load("data/processed/train.npz")
    val_data = np.load("data/processed/val.npz")

    x_train = train_data["images"]
    y_train = train_data["labels"]

    x_val = val_data["images"]
    y_val = val_data["labels"]

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(
            train_params["dense_units"],
            activation="relu"
        ),
        tf.keras.layers.Dropout(
            train_params["dropout_rate"]
        ),
        tf.keras.layers.Dense(
            10,
            activation="softmax"
        )
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=train_params["learning_rate"]
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=train_params["epochs"],
        batch_size=train_params["batch_size"],
        verbose=1
    )

    model.save("models/model.h5")

    with open("models/history.csv", "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "loss",
            "accuracy",
            "val_loss",
            "val_accuracy"
        ])

        for i in range(len(history.history["loss"])):
            writer.writerow([
                i + 1,
                history.history["loss"][i],
                history.history["accuracy"][i],
                history.history["val_loss"][i],
                history.history["val_accuracy"][i]
            ])

    print("Model saved to models/model.h5")
    print("Training history saved to models/history.csv")


if __name__ == "__main__":
    main()