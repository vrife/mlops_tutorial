# experiment.py
import mlflow
import mlflow.pytorch
from train import train_model

if __name__ == "__main__":
    mlflow.set_experiment("cifar10-classification")
    
    with mlflow.start_run():
        train_model()
        mlflow.log_artifact("cnn_model.pth")
        mlflow.pytorch.log_model(model, "cnn_model")
