import torch
import torch.nn as nn
import torch.optim as optim
from config import *
from gen import *


# Create teh neural network model
class BinaryClassifier(nn.Module):
    def __init__(self) -> None:
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.fc1(x).sigmoid()
        x = self.fc2(x).sigmoid()
        return x


model = BinaryClassifier()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.02)


def train(iteration, dataset, expected_output):
    for epoch in range(iteration):
        optimizer.zero_grad()
        outputs = model(dataset)
        loss = criterion(outputs, expected_output)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.8f}")


def train_with_feedback(iteration, dataset, expected_output):
    weights_fc1 = []
    weights_fc2 = []
    losses = []
    for epoch in range(iteration):
        weights_fc1.append(model.fc1.weight.data.numpy().copy())  
        weights_fc2.append(model.fc2.weight.data.numpy().copy())  
        optimizer.zero_grad()
        outputs = model(dataset)
        loss = criterion(outputs, expected_output)
        loss.backward()
        optimizer.step()

        # if (epoch + 1) % 10 == 0:
            # print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.8f}")
        
        losses.append(loss.item())
    weights_fc1.append(model.fc1.weight.data.numpy().copy())  
    weights_fc2.append(model.fc2.weight.data.numpy().copy())  
    return weights_fc1, weights_fc2, losses

def evaluate():
    with torch.no_grad():
        test_outputs = model(coordinates_test)
        test_loss = criterion(test_outputs, labels_test)
        predicted_class = (test_outputs > 0.5).float()
        accuracy = (predicted_class == labels_test).float().mean()
    print(f"Test Loss: {test_loss.item():.4f}")
    print(f"Test Accuracy: {accuracy.item() * 100:.8f}%")
