import os
import glob
import random
import time
import numpy as np
from neural_network.network import NeuralNetwork, cross_entropy_loss, accuracy

from neural_network.preprocessing import preprocess_image


DATASET_DIR = "numbers"
MODEL_PATH = "models/digit_model.npz"

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.0001

TRAIN_RATIO = 0.8
PATIENCE = 3

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def find_dataset():
    samples = []
    writers = []

    ignored = {"tools", "example", "UNCATEGORIZED"}

    for writer in sorted(os.listdir(DATASET_DIR)):
        writer_path = os.path.join(DATASET_DIR, writer)

        if not os.path.isdir(writer_path):
            continue

        if writer in ignored:
            continue

        writers.append(writer)

        for digit in range(10):
            digit_path = os.path.join(writer_path, str(digit))

            if not os.path.isdir(digit_path):
                continue

            files = glob.glob(os.path.join(digit_path, "number-*.png"))

            for path in files:
                samples.append((path, digit, writer))

    return samples, writers


def split_dataset(samples, writers):
    writers = writers.copy()

    random.shuffle(writers)

    split = int(len(writers) * TRAIN_RATIO)

    train_writers = set(writers[:split])

    test_writers = set(writers[split:])

    train_samples = []
    test_samples = []

    for sample in samples:
        if sample[2] in train_writers:
            train_samples.append(sample)
        else:
            test_samples.append(sample)

    return (train_samples, test_samples, train_writers, test_writers)


def load_samples(samples):
    X = np.empty((len(samples), 4096), dtype=np.float32)

    y = np.empty(len(samples), dtype=np.int64)

    for i, (path, label, _) in enumerate(samples):
        try:
            X[i] = preprocess_image(path)
            y[i] = label
        except Exception as e:
            print(f"\nFailed: {path}: {e}")
            X[i] = 0
            y[i] = label

        if (i + 1) % 500 == 0:
            print(f"\rLoaded {i + 1}/{len(samples)}", end="", flush=True)

    print()

    return X, y


def evaluate(network, X, y):
    predictions = network.forward(X)

    return (cross_entropy_loss(predictions, y), accuracy(predictions, y))


def print_digit_accuracy(predictions, labels):
    print()

    for digit in range(10):
        mask = labels == digit

        if not np.any(mask):
            continue

        score = np.mean(predictions[mask] == labels[mask])

        print(f"{digit}: {score * 100:.2f}% ({np.sum(mask)})")


def train():
    print("Finding dataset...")

    samples, writers = find_dataset()

    print(f"Found {len(samples)} images from {len(writers)} writers")

    (train_samples, test_samples, train_writers, test_writers) = split_dataset(
        samples, writers
    )

    print(f"Training images: {len(train_samples)}")

    print(f"Test images: {len(test_samples)}")

    print()
    print("Loading training data...")

    X_train, y_train = load_samples(train_samples)

    print("Loading test data...")

    X_test, y_test = load_samples(test_samples)

    print()
    print("Train:", X_train.shape)

    print("Test:", X_test.shape)

    network = NeuralNetwork()

    os.makedirs("models", exist_ok=True)

    best_accuracy = -1.0
    patience_counter = 0

    print()
    print("Starting training...")
    print()

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        indices = np.random.permutation(len(X_train))

        X_train = X_train[indices]
        y_train = y_train[indices]

        total_loss = 0.0
        batches = 0

        total_batches = (len(X_train) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_index, start in enumerate(
            range(0, len(X_train), BATCH_SIZE), start=1
        ):
            end = min(start + BATCH_SIZE, len(X_train))

            X_batch = X_train[start:end]

            y_batch = y_train[start:end]

            predictions = network.forward(X_batch)

            loss = cross_entropy_loss(predictions, y_batch)

            network.backward(y_batch)

            network.update(LEARNING_RATE)

            total_loss += loss
            batches += 1

            if (
                batch_index == 1
                or batch_index % 10 == 0
                or batch_index == total_batches
            ):
                elapsed = time.time() - epoch_start

                print(
                    f"\rEpoch "
                    f"{epoch + 1}/{EPOCHS} "
                    f"[{batch_index}/{total_batches}] "
                    f"loss={loss:.4f} "
                    f"time={elapsed:.1f}s",
                    end="",
                    flush=True,
                )

        train_loss = total_loss / batches

        test_loss, test_accuracy = evaluate(network, X_test, y_test)

        epoch_time = time.time() - epoch_start

        print()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- train loss: {train_loss:.4f} "
            f"- test loss: {test_loss:.4f} "
            f"- accuracy: "
            f"{test_accuracy * 100:.2f}% "
            f"- {epoch_time:.1f}s"
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

            network.save(MODEL_PATH)

            patience_counter = 0

            print(f"  New best model: {best_accuracy * 100:.2f}%")

        else:
            patience_counter += 1

            print(f"  No improvement ({patience_counter}/{PATIENCE})")

            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

    network.load(MODEL_PATH)

    predictions = network.predict(X_test)

    final_accuracy = np.mean(predictions == y_test)

    print()
    print(f"Best test accuracy: {best_accuracy * 100:.2f}%")

    print(f"Loaded best model accuracy: {final_accuracy * 100:.2f}%")

    print_digit_accuracy(predictions, y_test)

    print()
    print(f"Saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
