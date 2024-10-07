import requests
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Load CIFAR-10 test dataset
transform = transforms.Compose([transforms.ToTensor()])
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def test_prediction(index):
    # Get an image and its label from the test set
    image, label = testset[index]
    
    # Convert image to list for JSON serialization
    image_data = image.numpy().flatten().tolist()

    # Send POST request
    response = requests.post('http://localhost:8000/predict', 
                             json={'image': image_data})
    
    prediction = response.json()['prediction']

    # Plot the image
    plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    plt.title(f"True: {classes[label]}, Predicted: {classes[prediction]}")
    plt.axis('off')
    plt.show()

    print(f"True label: {classes[label]}")
    print(f"Predicted label: {classes[prediction]}")
    print(f"Prediction is {'correct' if label == prediction else 'incorrect'}")

# Test multiple predictions
for i in range(5):  # Test 5 random images
    print(f"\nTest {i+1}:")
    test_prediction(np.random.randint(0, len(testset)))