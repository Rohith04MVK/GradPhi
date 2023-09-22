import numpy as np
from gradphi.autodiff import Variable


class Dense:
    """
    A class representing a Dense (fully connected) layer in a neural network.

    The Dense layer, also known as a fully connected layer, performs linear transformations
    on input data. It is typically used in neural networks to connect every input neuron
    to every output neuron with trainable weights and optional bias.

    The forward pass computes the linear transformation as follows:
    output = X * weight^T + bias

    Parameters:
    -----------
    - in_features : int
        The number of input features or input neurons.
    - out_features : int
        The number of output features or output neurons.
    - bias : bool, optional (default=True)
        Whether to include bias terms in the layer.

    Attributes:
    ----------
    - weight : Variable
        The learnable weight matrix of shape (out_features, in_features).
    - bias : Variable or None
        The optional bias vector of shape (1, out_features). None if bias is not used.

    Methods:
    --------
    - forward(self, X):
      Compute the forward pass of the Dense layer.

      Parameters:
      -----------
      X : Variable
          The input data or input features of shape (batch_size, in_features).

      Returns:
      --------
      Variable
          The result of the linear transformation applied to the input data.

    - backward(self, grad=1):
      Compute the backward pass of the Dense layer.

      Parameters:
      -----------
      grad : float or numpy.ndarray, optional (default=1)
          The gradient of the loss with respect to the output of this layer.

      Returns:
      --------
      None
          This method updates the gradients of the weight and bias if present,
          but does not return any values.

    - __call__(self, X):
      Allow using the Dense layer instance as a callable object for the forward pass.

      Parameters:
      -----------
      X : Variable
          The input data or input features of shape (batch_size, in_features).

      Returns:
      --------
      Variable
          The result of the linear transformation applied to the input data.

    Usage:
    ------
    Create an instance of this class to add a Dense layer to your neural network. For example:

    ```python
    dense_layer = Dense(in_features=64, out_features=128, bias=True)
    input_data = Variable(np.random.randn(32, 64))  # Batch size of 32, 64 input features
    output = dense_layer(input_data)
    ```
    """

    def __init__(self, in_features, out_features, bias=True):
        # Initialize the super class (Variable)
        super().__init__(None, [], None)

        self.in_features = in_features
        self.out_features = out_features

        stdv = 1. / np.sqrt(in_features)
        self.weight = Variable(
            np.random.uniform(-stdv, stdv, (out_features, in_features)), dtype=np.float32)

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
            np.matmul(self.args[0].data.swapaxes(-1, -2), grad).swapaxes(-1, -2))
        self.args[0].backward(np.matmul(grad, self.args[1].data))

    def __call__(self, X):
        return self.forward(X)
