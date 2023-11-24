import torch
from sklearn.datasets import make_circles

# Generate the dataset with two concentric circles
coordinates, labels = make_circles(n_samples=500, noise=0.075, factor=0.5, random_state=2023)

# Convert the dataset to PyTorch tensors
coordinates = torch.tensor(coordinates, dtype=torch.float)
labels = torch.tensor(labels.reshape(-1, 1), dtype=torch.float)

# Split the dataset into training and testing sets
split = int(0.99 * len(coordinates))
coordinates_train, labels_train = coordinates[:split], labels[:split]
coordinates_test, labels_test = coordinates[split:], labels[split:]

