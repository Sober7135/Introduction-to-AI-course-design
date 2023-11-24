import matplotlib.pyplot as plt
from model import *
import numpy as np
from gen import *


# Plot the decision boundary
def decision_boundary():
    x1_min, x1_max = coordinates[:, 0].min() - 0.5, coordinates[:, 0].max() + 0.5
    x2_min, x2_max = coordinates[:, 1].min() - 0.5, coordinates[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01)
    )
    Z = model(torch.tensor(np.c_[xx1.ravel(), xx2.ravel()], dtype=torch.float))
    Z = (Z > 0.5).float().reshape(xx1.shape)

    plt.figure()
    plt.contourf(xx1, xx2, Z, alpha=0.8)
    plt.scatter(
        coordinates[:, 0], coordinates[:, 1], c=labels.squeeze(), edgecolors="k"
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Binary Classification for Concentric Circles")


def get_hidden_layer_output():
    with torch.no_grad():
        hidden_layer_output = model.fc1(coordinates).sigmoid()
    return hidden_layer_output


def hidden_layer_output():
    hidden_outputs = get_hidden_layer_output()
    # Plotting the 3D graph for each neuron's output
    fig_3d = plt.figure(figsize=(20, 20))
    for i in range(hidden_outputs.shape[1]):  # Iterating through each neuron
        ax = fig_3d.add_subplot(2, 2, i + 1, projection="3d")
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            hidden_outputs[:, i],
            c=labels[:, 0],
            cmap="viridis",
        )
        ax.set_title(f"Neuron {i} Output")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.set_zlabel("Neuron Output")

    fig_2d = plt.figure(figsize=(20, 20))
    for i in range(hidden_outputs.shape[1]):  # hidden_outputs.shape[1] 是神经元的数量
        plt.subplot(2, 2, i+1)
        plt.scatter(coordinates[:, 0], coordinates[:, 1], c=hidden_outputs[:, i], cmap='viridis')
        plt.colorbar()
        plt.title(f'Neuron {i} Output')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')

    fig_2d.subplots_adjust(hspace=0.5)
    fig_3d.subplots_adjust(hspace=0.5)

def weights():
    weights_fc1, weights_fc2 = train_with_weights(epochs, coordinates_train, labels_train)
    weights_fc1, weights_fc2 = np.array(weights_fc1), np.array(weights_fc2)
    plt.figure(figsize=(20,20))
    for neuron_index in range(weights_fc1.shape[1]):
        for weight_index in range(weights_fc1.shape[2]): 
            plt.plot(weights_fc1[:, neuron_index, weight_index], label=f'Neuron {neuron_index} Weight {weight_index}')

    plt.title('fc1 Weight Changes During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Weight Value')
    plt.legend()
  
    plt.figure(figsize=(20,20))
    for neuron_index in range(weights_fc2.shape[1]):
        for weight_index in range(weights_fc2.shape[2]): 
            plt.plot(weights_fc2[:, neuron_index, weight_index], label=f'Neuron {neuron_index} Weight {weight_index}')
    plt.title('fc2 Weight Changes During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Weight Value')
    plt.legend()

def analysis():
    # train
    weights()
    evaluate()
    decision_boundary()
    hidden_layer_output()
    plt.show()

if __name__ == "__main__":
    analysis()
