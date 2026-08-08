import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataset import test_loader, NUM_CLASSES, CLASS_NAMES
from model.cnn import SceneVisionCNN

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = SceneVisionCNN(NUM_CLASSES).to(device)

model.load_state_dict(
    torch.load(
        "outputs/checkpoints/best_model_e2.pth",
        map_location=device
    )
)

model.eval()

criterion = nn.CrossEntropyLoss()

running_loss = 0.0
correct = 0
total = 0

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        running_loss += loss.item()

        predicted = torch.argmax(outputs, dim=1)

        correct += (predicted == labels).sum().item()

        total += labels.size(0)

        all_predictions.extend(predicted.cpu().numpy())

        all_labels.extend(labels.cpu().numpy())

test_loss = running_loss / len(test_loader)

test_accuracy = 100 * correct / total

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES
    )
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

plt.figure(figsize=(8, 8))

disp.plot(
    cmap="Blues",
    xticks_rotation=45
)

plt.tight_layout()

plt.savefig(
    "assets/reports/confusion_matrix_e2.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
print("Confusion matrix saved!")