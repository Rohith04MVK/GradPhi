import numpy as np

from gradphi.autodiff import Variable


class Sigmoid(Variable):
    """
    Implements the sigmoid activation function, a ubiquitous element in neural networks.

    The sigmoid function, also known as the logistic function, squashes its input values
    to the range (0, 1) through an S-shaped curve. This characteristic makes it a popular
    choice for activation in neural networks, particularly for binary classification tasks
    where outputs need to represent probabilities. Within the broader context of the
    `Variable` class, the `Sigmoid` class transforms raw input values into activations
    suitable for subsequent layers in the network, contributing to the learning process.

    Attributes:
        None

    Methods:
        __init__(self):
            Initializes a new Sigmoid object.

        forward(self, x):
            Computes the sigmoid activation function for the given input.

        __call__(self, x):
            Alias for the forward method.

    Reference:
        https://en.wikipedia.org/wiki/Logistic_function
    """

    def __init__(self):
        pass

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def __call__(self, x):
        return self.forward(x)
