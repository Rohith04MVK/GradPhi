import numpy as np
from gradphi.autodiff import Variable


class Dense:
    """
    Dense (fully connected) layer for neural networks.

    The Dense layer is a fundamental component of a neural network, performing a linear transformation
    on the input data. It can be used as a hidden layer in a neural network to learn complex patterns.

    The Dense layer, also known as a fully connected or linear layer, is a crucial building block
    in neural networks. It performs a linear transformation on the input data, which is often followed
    by a non-linear activation function. This layer is responsible for learning complex relationships
    between input features and is widely used in various neural network architectures.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        bias (bool, optional): Whether to include a bias term (default is True).

    Attributes:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        weight (Variable): Learnable weights for the layer.
        bias (Variable or None): Learnable bias for the layer (if bias=True), or None (if bias=False).

    Methods:
        forward(self, X):
            Compute the forward pass of the Dense layer.

        backward(self, grad=1):
            Compute the backward pass of the Dense layer.

        __call__(self, X):
            Apply the Dense layer to the input.

    Example:
        >>> dense_layer = Dense(in_features=128, out_features=64, bias=True)
        >>> input_data = Variable(np.random.randn(32, 128))
        >>> output = dense_layer(input_data)

    References:
        - "Neural Networks and Deep Learning" by Michael Nielsen (http://neuralnetworksanddeeplearning.com/)
    """

    def __init__(self, in_features, out_features, bias=True):
        # Initialize the super class (Variable)
        super().__init__(None, [], None)

        self.in_features = in_features
        self.out_features = out_features

        stdv = 1.0 / np.sqrt(in_features)
        self.weight = Variable(
            np.random.uniform(-stdv, stdv, (out_features, in_features)),
            dtype=np.float32,
        )

        if bias:
            self.bias = Variable(np.zeros((1, out_features)), dtype=np.float32)
        else:
            self.bias = None

    def forward(self, X):
        self.args = [X, self.weight, self.bias]
        self.op = "linear"

        output = np.matmul(X.data, self.weight.data.T)
        if self.bias is not None:
            output += self.bias.data

        # Create a new Dense instance for backpropagation
        return self.backward()

    def backward(self, grad=1):
        if self.bias is not None:
            self.args[2].backward(np.sum(grad, axis=0, keepdims=True))
        self.args[1].backward(
            np.matmul(self.args[0].data.swapaxes(-1, -2), grad).swapaxes(-1, -2)
        )
        self.args[0].backward(np.matmul(grad, self.args[1].data))

    def __call__(self, X):
        return self.forward(X)
