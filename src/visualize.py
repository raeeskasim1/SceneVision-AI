import matplotlib.pyplot as plt
import pickle
import torch

from PIL import Image
from dataset import train_loader, CLASS_NAMES, transform, NUM_CLASSES
from model.cnn import SceneVisionCNN
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

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = SceneVisionCNN(NUM_CLASSES).to(device)

model.load_state_dict(
    torch.load(
        "outputs/checkpoints/best_model.pth",
        map_location=device
    )
)

model.eval()

image_path = Path("data") / "intel" / "seg_pred" / "10004.jpg"

image = Image.open(image_path).convert("RGB")
image = transform(image)
image = image.unsqueeze(0).to(device)

with torch.no_grad():
    outputs, fm1, fm2, fm3 = model(image)


def visualize_feature_maps(feature_maps, block_name):
    plt.figure(figsize=(10,10))
    num_maps = min(16, feature_maps.shape[1])
    for i in range(num_maps):
        plt.subplot(4,4,i+1)
        feature_map = feature_maps[0, i].cpu()

        plt.imshow(feature_map, cmap="gray")
        plt.axis("off")
        plt.suptitle(
                    f"Feature Maps - {block_name}",
                    fontsize=16
                )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.title(f"FM {i+1}", fontsize=8)
    plt.savefig(
            f"assets/{block_name.lower().replace(' ', '_')}_v1.png",
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


if __name__ == "__main__":

    # plot_training_curves()
    visualize_feature_maps(fm1, "block1")
    visualize_feature_maps(fm2, "block2")
    visualize_feature_maps(fm3, "block3")