import matplotlib.pyplot as plt
import pickle

from dataset import train_loader, CLASS_NAMES
from pathlib import Path

def visualize_sample_batch():

    images, labels = next(iter(train_loader))

    plt.figure(figsize=(10,10))

    for i in range(16):
        plt.subplot(4,4,i+1)

        image = images[i].permute(1,2,0)

        plt.imshow(image)

        plt.title(f"Label: {CLASS_NAMES[labels[i]]}")

        plt.axis("off")

    plt.tight_layout()

    plt.savefig("outputs/sample_batch.png", dpi=300)

    plt.show()

def plot_training_curves():
    history_path = Path("outputs") / "history.pkl"

    with open(history_path, "rb") as f:
        history = pickle.load(f)

    train_losses = history["train_loss"]
    val_losses = history["val_loss"]

    train_accuracies = history["train_accuracy"]
    val_accuracies = history["val_accuracy"]

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(
        epochs,
        train_losses,
        label="Train Loss"
    )
    plt.plot(
        epochs,
        val_losses,
        label="Validation Loss"
    )
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(
        epochs,
        train_accuracies,
        label="Train Accuracy"
    )
    plt.plot(
        epochs,
        val_accuracies,
        label="Validation Accuracy"
    )
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "assets/training_curves_v1.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

if __name__ == "__main__":

    plot_training_curves()