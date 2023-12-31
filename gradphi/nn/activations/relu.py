import numpy as np
from gradphi.autodiff import Variable


class ReLU(Variable):
    """
    The Rectified Linear Unit (ReLU) activation function for neural networks.

    ReLU is a widely used activation function in neural networks, known for its simplicity and effectiveness.
    It introduces non-linearity into the network by returning the input if it's positive and zero otherwise.
    ReLU has gained popularity due to its simplicity and its ability to mitigate the vanishing gradient problem.
    It is widely used in various neural network architectures.

    Attributes:
        No attributes are specific to this class.

    Methods:
        forward(self, x):
            Compute the forward pass of the ReLU activation function.

        backward(self, grad=1):
            Compute the backward pass of the ReLU activation function.

        __call__(self, x):
            Apply the ReLU activation function to the input.

    Example:
        >>> relu = ReLU()
        >>> input_data = np.array([-2, -1, 0, 3, 4])
        >>> output = relu(input_data)
        >>> print(output)
        [0 0 0 3 4]

    References:
        - "Rectified Linear Units (ReLU)" by Vinod Nair and Geoffrey E. Hinton (https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf)
    """

    def __init__(self, x):
        self.input_data = x
        self.output_data = np.ones_like(x.data)

    def forward(self):
        self.output_data = self.input_data.maximum(0)
        return self.input_data.maximum(0)

    def backward(self):
        self.output_data.backward(np.ones_like(self.output_data.data))
        return self.input_data.grad

    def __call__(self, x):
        return self.forward(x)
