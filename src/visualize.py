import matplotlib.pyplot as plt
import pickle
import torch
import random

from PIL import Image
from dataset import train_loader, CLASS_NAMES, test_transform, NUM_CLASSES,test_loader
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
    history_path = Path("outputs") / "history_e4.pkl"

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
        "assets/curves/training_curves_e4.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def get_feature_maps():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SceneVisionCNN(NUM_CLASSES).to(device)

    model.load_state_dict(
        torch.load(
            "outputs/checkpoints/best_model_e4.pth",
            map_location=device
        )
    )

    model.eval()

    image_path = Path("data") / "intel" / "seg_pred" / "10004.jpg"

    image = Image.open(image_path).convert("RGB")

    image = test_transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        _, fm1, fm2, fm3, fm4 = model(image)

    return fm1, fm2, fm3, fm4


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
            f"assets/feature_maps/{block_name.lower().replace(' ', '_')}_v2.png",
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

def visualize_misclassified_images():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SceneVisionCNN(NUM_CLASSES).to(device)

    model.load_state_dict(
        torch.load(
            "outputs/checkpoints/best_model_e4.pth",
            map_location=device
        )
    )

    model.eval()

    misclassified_images = []
    actual_labels = []
    predicted_labels = []
    confidences = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs, _, _, _,_ = model(images)

            probabilities = torch.softmax(outputs, dim=1)

            confidence, predicted = torch.max(
                probabilities,
                dim=1
            )

            mask = predicted != labels

            misclassified_images.extend(
                images[mask].cpu()
            )

            actual_labels.extend(
                labels[mask].cpu()
            )

            predicted_labels.extend(
                predicted[mask].cpu()
            )

            confidences.extend(
                confidence[mask].cpu()
            )

    plt.figure(figsize=(12, 12))

    indices = random.sample(
        range(len(misclassified_images)),
        min(16, len(misclassified_images))
    )

    for plot_idx, sample_idx in enumerate(indices):

        plt.subplot(4, 4, plot_idx + 1)

        image = misclassified_images[sample_idx].permute(1, 2, 0)

        plt.imshow(image)

        actual = CLASS_NAMES[actual_labels[sample_idx]]

        predicted = CLASS_NAMES[predicted_labels[sample_idx]]

        confidence_score = confidences[sample_idx].item() * 100

        plt.title(
            f"A: {actual}\n"
            f"P: {predicted}\n"
            f"{confidence_score:.1f}%",
            fontsize=8
        )

        plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        "assets/misclassified_images_v2.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

if __name__ == "__main__":

    # plot_training_curves()

    # plot_training_curves()
    fm1, fm2, fm3, fm4 = get_feature_maps()
    visualize_feature_maps(fm1, "feature_map1")
    visualize_feature_maps(fm2, "feature_map2")
    visualize_feature_maps(fm3, "feature_map3")
    visualize_feature_maps(fm4, "feature_map4")

    visualize_misclassified_images()