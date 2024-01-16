import sys
import torch
import torch.nn as nn
from torchvision import transforms as T, datasets
from PIL import Image

with open("config.cfg", "r") as f:
    cfg = []
    for line in f.readlines():
        if line[0] != "#":
            cfg.append(line.split("=")[1].strip("\n").strip('"'))
    try:
        global acc
        acc = float(cfg[0])
    except ValueError:
        print(
            "Config file is corrupted. Please use the GUI to recalculate the accuracy."
        )
    model_file = cfg[1]


class pneumonia(nn.Module):
    def __init__(self, num_classes=2):
        super(pneumonia, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1
        )

        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(
            in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1
        )
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(in_features=32 * 112 * 112, out_features=num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        output = output.view(-1, 32 * 112 * 112)
        output = self.fc(output)

        return output


def detect(image_path):
    try:
        model = torch.load(model_file)
        model.eval()
    except FileNotFoundError:
        print(
            "Model file was not found. Please make sure it's in the same directory as the application. Make sure its name is the same one as specified in config file.",
        )
    data_T = T.Compose(
        [
            T.Resize(size=(224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    img_tensor = data_T(img)
    img_new = img_tensor.view(1, 3, 224, 224)
    with torch.no_grad():
        logps = model(img_new)
    ps = torch.exp(logps)
    probab = list(ps.cpu()[0])
    pred_label = probab.index(max(probab))
    if pred_label:
        print(f"Pneunomia detected. Accuracy = {round(acc*100, 2)}%")
    else:
        print(f"Pneunomia not detected. Accuracy = {round(acc*100, 2)}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cli.py <image_path>")
    else:
        image_path = sys.argv[1]
        detect(image_path)
