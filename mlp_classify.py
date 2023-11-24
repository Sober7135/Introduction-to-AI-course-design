import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# Generate the dataset with two concentric circles
X, y = make_circles(n_samples=500, noise=0.075, factor=0.5, random_state=2023)

# Convert the dataset to PyTorch tensors
X = torch.tensor(X, dtype=torch.float)
print(X)
y = torch.tensor(y.reshape(-1, 1), dtype=torch.float)

# Split the dataset into training and testing sets
split = int(0.99 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Create the neural network model
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Create an instance of the model
model = BinaryClassifier()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.02)

# Train the model
for epoch in range(250):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{120}, Loss: {loss.item():.4f}')

# Evaluate the model on the testing set
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    predicted_classes = (test_outputs > 0.5).float()
    accuracy = (predicted_classes == y_test).float().mean()

print(f'Test Loss: {test_loss.item():.4f}')
print(f'Test Accuracy: {accuracy.item()*100:.2f}%')

# Plot the decision boundary
x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01))
Z = model(torch.tensor(np.c_[xx1.ravel(), xx2.ravel()], dtype=torch.float))
Z = (Z > 0.5).float().reshape(xx1.shape)

plt.contourf(xx1, xx2, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), edgecolors='k')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Binary Classification for Concentric Circles')
plt.show()