# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY cnn_model.pth /app/cnn_model.pth
COPY serve.py /app/serve.py
COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
