from model import *
from gen import *
from config import *

# Create the neural network model
train(epochs, coordinates_train, labels_train)
# Evaluate the model on the testing set
evaluate()
