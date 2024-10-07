# End-to-End MLOps Tutorial with PyTorch and CIFAR-10

This project demonstrates an end-to-end MLOps pipeline using PyTorch, FastAPI, Docker, Prometheus, and Grafana. We train a Convolutional Neural Network (CNN) on the CIFAR-10 dataset, deploy it as a REST API, and monitor its performance.

## Project Structure

- `train.py`: Script for training the CNN model
- `serve.py`: FastAPI application for serving predictions
- `test.py`: Script for testing the deployed model
- `Dockerfile`: Instructions for building the Docker image
- `requirements.txt`: Python dependencies
- `prometheus.yml`: Prometheus configuration file

## Setup and Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd end_to_end_tutorial
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

## Project Components

### DVC (.dvc directory)

This project uses DVC (Data Version Control) for managing and versioning datasets and machine learning models. The `.dvc` directory contains DVC-specific files that help track changes to data and ensure reproducibility.

To use DVC:
1. Initialize DVC in your project: `dvc init`
2. Create the data dire
2. Add files to track: `dvc add data/large_dataset.csv`
3. Commit the changes: `git commit -m "Add dataset"`
4. Push to remote storage: `dvc push`

### MLflow (mlruns directory) - to be added

MLflow is used in this project for experiment tracking and model management. The `mlruns` directory contains metadata about your machine learning experiments, including parameters, metrics, and artifacts.

To use MLflow:
1. Start tracking an experiment with the script:
   ```python
   import mlflow

   with mlflow.start_run():
       mlflow.log_param("learning_rate", 0.01)
       mlflow.log_metric("accuracy", 0.85)
       mlflow.pytorch.log_model(model, "model")
   ```
2. View the MLflow UI: `mlflow ui`
3. Access the UI in your browser at `http://localhost:5000`

These tools enhance the reproducibility, traceability, and manageability of your machine learning project.

## Model Architecture and Evolution

### Initial Model (old_train.py)

The initial model was a simple Convolutional Neural Network (CNN) with the following architecture:

```
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

This model had limitations in its capacity to learn complex features and was prone to overfitting.

### Improved Model (current train.py)

The current model, `ImprovedCNN`, addresses these limitations with the following enhancements:

1. Additional Convolutional Layer: Increases the model's capacity to learn more complex features.
2. Dropout: Reduces overfitting by randomly setting a fraction of input units to 0 during training.
3. Batch Normalization: Normalizes the input layer by adjusting and scaling the activations, which can lead to faster learning and higher overall accuracy.

Here's the architecture of the improved model:

```
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
```

### Training Process Improvements

Along with the model architecture changes, we've also enhanced the training process:

1. Data Augmentation: We now apply random crops and horizontal flips to the training data, increasing the effective size of our dataset and improving generalization.

2. Learning Rate Scheduler: We've implemented a ReduceLROnPlateau scheduler, which reduces the learning rate when the validation loss plateaus, allowing for finer adjustments as training progresses.

3. Early Stopping: We now stop training if the validation loss doesn't improve for a set number of epochs, preventing overfitting and unnecessary computation.

4. Validation Set: We've introduced a separate validation set to better monitor the model's performance on unseen data during training.

These improvements collectively result in a more robust model with better generalization capabilities, reflected in improved accuracy on the CIFAR-10 dataset.

## Training the Model

1. Run the training script:
   ```
   python train.py
   ```
   This will train the CNN on the CIFAR-10 dataset and save the best model as `best_cnn_model.pth`.

## Building and Running the Docker Container

1. Build the Docker image:
   ```
   docker build -t cnn_model_api .
   ```

2. Run the Docker container:
   ```
   docker run -p 8000:8000 cnn_model_api
   ```
   This starts the FastAPI application on `http://localhost:8000`.

## Testing the Model

1. Run the test script:
   ```
   python test.py
   ```
   This will send sample images to the API and display the predictions.

## Monitoring with Prometheus and Grafana

1. Start Prometheus:
   ```
   docker run -d --name prometheus -p 9090:9090 -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
   ```

2. Start Grafana:
   ```
   docker run -d --name grafana -p 3000:3000 grafana/grafana
   ```

3. Access Grafana at `http://localhost:3000` (default credentials: admin/admin)

4. Add Prometheus as a data source in Grafana:
   - URL: `http://host.docker.internal:9090`
   - Access: Browser

5. Create dashboards in Grafana to visualize metrics from your API.

## API Endpoints

- `/predict`: POST request for making predictions
  - Input: JSON with 'image' key containing a flattened 32x32x3 image array
  - Output: JSON with 'prediction' key containing the predicted class

- `/metrics`: GET request for Prometheus metrics

## Model Performance

The current model achieves around 60-70% accuracy on the CIFAR-10 test set. This performance can be improved by:
- Introducing early stopping in trainng
- Using a more complex model architecture (more layers)
- Fine-tuning hyperparameters:
  * Number of epochs
  * Batch size
  * Learning rate
  * Dropout rate
  * Number of filters in each convolutional layer
  * Kernel size in each convolutional layer
  * Pooling size in each pooling layer
  * Activation function
  * Optimizer
  

## Future Improvements

- Implement CI/CD pipeline for automatic training and deployment
- Add more comprehensive monitoring and alerting
- Implement A/B testing for model versions
- Add model versioning and experiment tracking

## Troubleshooting

If you encounter any issues, please check:
- All required packages are installed
- Docker is running and has sufficient resources
- Ports 8000, 9090, and 3000 are not in use by other applications

For any other problems, please open an issue in the repository.