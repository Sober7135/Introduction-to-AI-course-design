from matplotlib import pyplot as plt
from model import *
import result_analysis
from gen import *
from config import *

# Create the neural network model
train(epochs, coordinates_train, labels_train)
# Evaluate the model on the testing set
evaluate(coordinates_test, labels_test)
