# serve.py
from fastapi import FastAPI
import torch
from train import SimpleCNN
from PIL import Image
import torchvision.transforms as transforms

app = FastAPI()

model = SimpleCNN()
model.load_state_dict(torch.load("cnn_model.pth"))
model.eval()

transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

@app.post("/predict/")
async def predict(file: bytes):
    img = Image.open(io.BytesIO(file))
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        prediction = model(img)
    return {"prediction": prediction.argmax().item()}
