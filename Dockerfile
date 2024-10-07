# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY best_cnn_model.pth .
COPY serve.py .
COPY train.py .

CMD ["python", "serve.py"]
