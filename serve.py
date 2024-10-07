# serve.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from train import ImprovedCNN  # Import the new model class
from prometheus_client import make_asgi_app, Counter

app = FastAPI()

# Add prometheus asgi middleware to route /metrics requests
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Create a metric to track predictions
PREDICTIONS = Counter('predictions_total', 'Total number of predictions made')

model = ImprovedCNN()
model.load_state_dict(torch.load("best_cnn_model.pth", map_location=torch.device('cpu')))
model.eval()

class ImageData(BaseModel):
    image: list

@app.post("/predict")
async def predict(image_data: ImageData):
    try:
        image = torch.tensor(image_data.image, dtype=torch.float32).reshape(1, 3, 32, 32)
        with torch.no_grad():
            output = model(image)
        prediction = output.argmax(dim=1).item()
        PREDICTIONS.inc()  # Increment the predictions counter
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
