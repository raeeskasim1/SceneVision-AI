from pathlib import Path

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

DATA_DIR = Path("data") / "intel"

TRAIN_DIR = DATA_DIR / "seg_train"
TEST_DIR = DATA_DIR / "seg_test"

# print(TRAIN_DIR.exists())
# print(TEST_DIR.exists())
# print(list(TRAIN_DIR.iterdir()))

transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor()
])

full_train_dataset=datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=transform
)

dataset_size=len(full_train_dataset)

print(dataset_size)

train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

print(train_size)
print(val_size)

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size , val_size]
)

print(len(train_dataset))
print(len(val_dataset))


