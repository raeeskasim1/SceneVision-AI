import torch
from PIL import Image
from pathlib import Path

from torchvision import transforms

from dataset import CLASS_NAMES, NUM_CLASSES,transform
from model.cnn import SceneVisionCNN

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

IMAGE_PATH = Path("data") / "intel" / "seg_pred" / "5.jpg"

image = Image.open(IMAGE_PATH)
image = image.convert("RGB")
image = transform(image)
image = image.unsqueeze(0)

image = image.to(device)

with torch.no_grad():
    outputs = model(image)
    probabilities = torch.softmax(outputs, dim=1)
    predicted = torch.argmax(probabilities, dim=1)
    predicted_class = predicted.item()
    predicted_label = CLASS_NAMES[predicted_class]
    confidence = probabilities[0][predicted_class].item() * 100

print(f"Predicted Class: {predicted_label}")
print(f"Confidence: {confidence:.2f}%")