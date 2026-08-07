from pathlib import Path
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Subset

DATA_DIR = Path("data") / "intel"

TRAIN_DIR = DATA_DIR / "seg_train"
TEST_DIR = DATA_DIR / "seg_test"

# print(TRAIN_DIR.exists())
# print(TEST_DIR.exists())
# print(list(TRAIN_DIR.iterdir()))

train_transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor()
])



train_full_dataset=datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=train_transform
)

val_full_dataset=datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=test_transform
)

CLASS_NAMES = train_full_dataset.classes
NUM_CLASSES = len(CLASS_NAMES)

dataset_size=len(train_full_dataset)

train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

generator = torch.Generator().manual_seed(42)

train_indices, val_indices = random_split(
    train_full_dataset,
    [train_size , val_size],
    generator=generator
)


train_dataset = Subset(
    train_full_dataset,
    train_indices.indices
)

val_dataset = Subset(
    val_full_dataset,
    val_indices.indices
)


BATCH_SIZE = 32

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)



