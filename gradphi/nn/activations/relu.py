import numpy as np
from gradphi.autodiff import Variable


class ReLU(Variable):
    """
    A class representing the Rectified Linear Unit (ReLU) activation function.

    The Rectified Linear Unit (ReLU) is a widely used activation function in neural networks.
    It introduces non-linearity by transforming the input data such that negative values
    are replaced with zero, while positive values remain unchanged. The ReLU activation is
    defined as follows:

    ReLU(x) = max(0, x)

    Attributes:
    ----------
    None

    Methods:
    --------
    - forward(self, x):
      Compute the forward pass of the ReLU activation function.

      Parameters:
      -----------
      x : numpy.ndarray
          The input data to which the ReLU activation will be applied.

      Returns:
      --------
      numpy.ndarray
          The result of the ReLU activation applied to the input data 'x'.

    - backward(self, grad=1):
      Compute the backward pass of the ReLU activation function.

      Parameters:
      -----------
      grad : float, optional
          An optional gradient value that can be provided. Default is 1.

      Returns:
      --------
      None
          This method prints the gradient information but does not
          perform gradient computation in this implementation.

    - __call__(self, x):
      Allow using the ReLU instance as a callable object for the forward pass.

      Parameters:
      -----------
      x : numpy.ndarray
          The input data to which the ReLU activation will be applied.

      Returns:
      --------
      numpy.ndarray
          The result of the ReLU activation applied to the input data 'x'.

    Usage:
    ------
    Create an instance of this class and use it to apply the ReLU activation function
    to your input data. For example:

    ```python
    relu = ReLU()
    input_data = numpy.array([-2.0, 0.5, 3.0, -1.5])
    output = relu(input_data)
    ```

    The output will be: `[0.  0.5 3.  0. ]`
    """

    def __init__(self):
        pass

    def forward(self, x):
        self.input_data = x
        return x.maximum(0)

    def backward(self, grad=1):
        print(self.input_data.backward(np.ones_like(self.input_data.data)))
        return self.input_data.backward()

    def __call__(self, x):
        return self.forward(x)
