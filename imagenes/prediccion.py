import torch
from torchvision import transforms
from PIL import Image
import sys
import torch.nn as nn
#definir la arquitectura del modelo CNN(la misma de antes)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# Cargar el modelo entrenado
# Asegurarse de que el modelo esté en el mismo dispositivo que la imagen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Cargar el modelo
model = CNN().to(device)
# Cargar los pesos del modelo entrenado
model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
# poner el modelo en modo evaluación
model.eval()

#Definir las transformaciones (igual a los que use antes)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

#Cargar imagen a predecir
image_path = sys.argv[1]  # se espera que se pase por consola
image = Image.open(image_path).convert("RGB") # Convertir a RGB si es necesario
image = transform(image).unsqueeze(0).to(device) # Añadir una dimensión para el batch 

#Predecir
with torch.no_grad():
    # Realizar la predicción
    output = model(image)
# Obtener la clase con mayor probabilidad
    _, predicted = torch.max(output, 1)


# Definir el mapeo de clases
class_map = {0: "normal", 1: "pancreatic_tumor"}
# Imprimir la clase predicha
print(class_map[predicted.item()])
