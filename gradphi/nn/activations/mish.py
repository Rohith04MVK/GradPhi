from gradphi.autodiff import Variable


class Mish(Variable):
    """
    Implements the Mish activation function,A Self Regularized Non-Monotonic Activation Function.

    The Mish function, introduced in 2019, has demonstrated superior performance
    compared to ReLU and Swish in various deep learning tasks. It's defined as:

       Mish(x) = x * tanh(softplus(x))
       where softplus(x) = ln(1 + exp(x))

    Mish offers several advantages:
    - Smoothness: Unlike ReLU, Mish is smooth and non-monotonic, reducing the
       risk of vanishing gradients and allowing for better information flow.
    - Self-regularization: The function's curvature acts as a form of
       regularization, potentially improving generalization.
    - Better performance: Mish has consistently outperformed ReLU and Swish
       in image classification, object detection, and other tasks.

    Attributes:
        None

    Methods:
        __init__(self)
            Initializes a new Mish object.

        forward(self, x)
            Computes the Mish activation function for the given input.

        __call__(self, x)
            Alias for the forward method.

    Reference:
        https://arxiv.org/abs/1908.08681
    """

    def __init__(self):
        pass

    def forward(self, x):
        return x.mul(x.tanh().mul(x.exp().add(1)).log())

    def __call__(self, x):
        return self.forward(x)
