import torch
import torch.nn as nn
import pickle

from dataset import train_loader,val_loader,NUM_CLASSES
from model.cnn import SceneVisionCNN
from utils import set_seed

device=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = SceneVisionCNN(NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=5
)

print(f"Using device: {device}")
print(f"Training samples: {len(train_loader.dataset)}")
print(f"Validation samples: {len(val_loader.dataset)}")
print("-" * 60)

EPOCHS = 50

train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []

best_val_loss = float("inf")
best_train_loss = 0
best_train_accuracy = 0
best_val_accuracy = 0
best_epoch = 0

patience = 10
counter = 0

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predicted = torch.argmax(outputs, dim=1)

        correct += (predicted == labels).sum().item()

        total += labels.size(0)

    train_accuracy = 100 * correct /total
    train_loss = running_loss / len(train_loader)

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    model.eval()

    running_val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_val_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)

            correct += (predicted == labels).sum().item()

            total += labels.size(0)

    val_loss = running_val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy) 

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f} | "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )           

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch + 1
        best_train_loss = train_loss
        best_train_accuracy = train_accuracy
        best_val_accuracy = val_accuracy

        counter = 0

        torch.save(
            model.state_dict(),
            "outputs/checkpoints/best_model_e4.pth"
        )
        print(
            f" Best model saved! "
            f"(Epoch {epoch + 1}, Best Val Loss: {best_val_loss:.4f})")

    else:

        counter += 1
        print(f"Early Stopping Counter: {counter}/{patience}")

    scheduler.step(val_loss)

    if counter >= patience:

        print("Early stopping triggered!")

        break

print("\n" + "=" * 60)
print("BEST MODEL RESULTS")
print("=" * 60)

print(f"Best Epoch:        {best_epoch}")
print(f"Train Loss:        {best_train_loss:.4f}")
print(f"Train Accuracy:    {best_train_accuracy:.2f}%")
print(f"Validation Loss:   {best_val_loss:.4f}")
print(f"Validation Accuracy: {best_val_accuracy:.2f}%")
print("=" * 60)

history = {
    "train_loss": train_losses,
    "train_accuracy": train_accuracies,
    "val_loss": val_losses,
    "val_accuracy": val_accuracies
}

with open("outputs/history_e4.pkl","wb") as f:
    pickle.dump(history, f)

print("Training history saved!")
