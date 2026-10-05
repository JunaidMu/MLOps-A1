import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def main():
    os.makedirs("data/processed", exist_ok=True)

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    test_size = params["preprocess"]["test_size"]
    seed = params["preprocess"]["seed"]

    train_data = np.load("data/raw/train.npz")
    test_data = np.load("data/raw/test.npz")

    x_train = train_data["images"].astype("float32") / 255.0
    y_train = train_data["labels"]

    x_test = test_data["images"].astype("float32") / 255.0
    y_test = test_data["labels"]

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=test_size,
        random_state=seed,
        stratify=y_train
    )

    np.savez_compressed(
        "data/processed/train.npz",
        images=x_train,
        labels=y_train
    )

    np.savez_compressed(
        "data/processed/val.npz",
        images=x_val,
        labels=y_val
    )

    np.savez_compressed(
        "data/processed/test.npz",
        images=x_test,
        labels=y_test
    )

    print("Preprocessing complete.")
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")


if __name__ == "__main__":
    main()