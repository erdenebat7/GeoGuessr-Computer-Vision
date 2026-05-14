import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pathlib
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# these must match the values used during training in train.py
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

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


def load_and_transform_image(path):
    image = Image.open(path).convert("RGB")
    pipeline = transforms.Compose(
        [
            transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return pipeline(image).unsqueeze(0)  



def predict(test_dir):
    test_dir = pathlib.Path(test_dir)

    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, len(CLASSES))
    )

    device = torch.device("cpu")
    model.load_state_dict(torch.load("model.pt", map_location=device, weights_only=True))
    model.eval()

    predictions = {}
    with torch.no_grad():
        for path in sorted(test_dir.glob("*.jpg")):
            image = load_and_transform_image(path)
            output = model(image)
            predicted_index = output.argmax(dim=1).item()
            predictions[path.name] = CLASSES[predicted_index]

    return predictions


if __name__ == "__main__":
    preds = predict("./testdata")
    print("Predictions:")
    for filename, label in sorted(preds.items()):
        print(f"{filename}: {label}")
