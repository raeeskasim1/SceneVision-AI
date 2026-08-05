import matplotlib.pyplot as plt
import torchvision

from dataset import train_loader, CLASS_NAMES

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
