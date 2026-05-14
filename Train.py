import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pathlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt


# configuration ------------------------------------------------------------------------
TRAIN_DIR = pathlib.Path("./data")

CLASSES = sorted(
    [
        "Anaheim",
        "Bakersfield",
        "Los_Angeles",
        "Riverside",
        "SLO",
        "San_Diego",
    ]
)
CLASS_TO_NUMBER = {name: i for i, name in enumerate(CLASSES)}

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 10

VALIDATION_FRACTION = 0.2


# dataset ------------------------------------------------------------------------------

class SoCalDataset(Dataset):
    """Loads images from the training set."""

    def __init__(self, root, transform=None):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.samples = [] 

        for path in sorted(self.root.glob("*.jpg")):
            label = path.name.rsplit("-", 1)[0]
            self.samples.append((path, CLASS_TO_NUMBER[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# training -----------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1) Transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), # Upgrade 2: Rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
        transforms.ToTensor(),
        normalize,
    ])

    # Step 2) Dataset and DataLoaders
    dataset_train_transformed = SoCalDataset(TRAIN_DIR, transform=train_transform)
    dataset_val_transformed = SoCalDataset(TRAIN_DIR, transform=val_transform)

    num_samples = len(dataset_train_transformed)
    indices = torch.randperm(num_samples).tolist()
    val_size = int(num_samples * VALIDATION_FRACTION)

    train_dataset = Subset(dataset_train_transformed, indices[val_size:])
    val_dataset = Subset(dataset_val_transformed, indices[:val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Step 3) Model setup
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    num_ftrs = model.fc.in_features
    # Upgrade 1: Dropout Layer
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, len(CLASSES))
    )
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Upgrade 3: Learning Rate Scheduler
    # This cuts the learning rate in half every 4 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    # Step 4) The training loop.
    epoch_losses = []
    best_val_accuracy = 0.0 # Track the high score

    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0

        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        epoch_losses.append(avg_loss)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device) 
                
                outputs = model(images)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += images.size(0)
                
        val_accuracy = val_correct / val_total

        print(
            f"Epoch {epoch + 1}/{EPOCHS}  "
            f"loss: {avg_loss:.4f}  "
            f"accuracy: {accuracy:.4f}  "
            f"val_accuracy: {val_accuracy:.4f}"
        )

        # Save the model ONLY if it beats our high score
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "model.pt")
            print(f" ---> 🌟 New best model saved! ({best_val_accuracy:.4f})")

        # Step the scheduler so it knows an epoch finished
        scheduler.step()

    # Generate the training curve graph
    print("Generating training curve...")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, EPOCHS + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title('Training Curve (Epoch vs. Loss)')
    plt.xlabel('Epoch / Iteration')
    plt.ylabel('Empirical Risk (Loss)')
    plt.grid(True)
    plt.savefig('training_curve.png')
    print("Saved graph to training_curve.png!")

if __name__ == "__main__":
    main()
