import os
import numpy as np
from tensorflow.keras.datasets import fashion_mnist


def main():
    os.makedirs("data/raw", exist_ok=True)

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    np.savez_compressed(
        "data/raw/train.npz",
        images=x_train,
        labels=y_train
    )

    np.savez_compressed(
        "data/raw/test.npz",
        images=x_test,
        labels=y_test
    )

    print("Fashion-MNIST downloaded and saved to data/raw/")


if __name__ == "__main__":
    main()