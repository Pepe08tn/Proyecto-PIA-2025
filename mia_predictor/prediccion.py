import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import os

# ======== RUTA DEL MODELO ======== #

# Directorio raíz del proyecto (uno arriba de /mia_predictor)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ruta absoluta al archivo cnn_model.pth dentro de /api
MODEL_PATH = os.path.join(BASE_DIR, "api", "cnn_model.pth")


# ======== MODELO REAL ENTRENADO ======== #

class CNN_simple(nn.Module):
    def __init__(self):
        super(CNN_simple, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ======== CARGAR MODELO ======== #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_simple().to(device)

# Cargar pesos entrenados usando la ruta absoluta
model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=True)

model.eval()


# ======== TRANSFORMACIÓN DE LA IMAGEN ======== #

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


# ======== FUNCIÓN PÚBLICA PARA LA API ======== #

def predict(image_path: str):

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    class_map = {0: "normal", 1: "pancreatic_tumor"}

    return {
        "predicted_class": class_map[predicted.item()],
        "probabilities": {
            "normal": float(probs[0]),
            "pancreatic_tumor": float(probs[1])
        }
    }
