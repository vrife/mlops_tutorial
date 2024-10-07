import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_prep import download_data

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

def train_model():
    trainset, testset = download_data()
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(2):  # 2 epochs for simplicity
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1} completed")
    
    torch.save(model.state_dict(), "cnn_model.pth")
    print("Model trained and saved as cnn_model.pth")

if __name__ == "__main__":
    train_model()