from gradphi.autodiff import Variable


class Softplus(Variable):
    """
    Implements the softplus activation function, a smooth approximation of ReLU.

    The softplus function is a smooth, non-linear function that closely resembles
    the rectified linear unit (ReLU) but has a smoother transition at x = 0. It's defined
    as:

       softplus(x) = ln(1 + exp(x))

    It offers several advantages:
    - Smoothness: The softplus function is smooth and differentiable everywhere,
       which can help prevent vanishing gradients during training.
    - Boundedness: The output of softplus is always positive, making it suitable
       for modeling non-negative quantities.
    - Closeness to ReLU: Softplus shares many of the desirable properties of ReLU,
       such as sparsity and ability to model non-linearities.

    Attributes:
        None

    Methods:
        __init__(self)
            Initializes a new Softplus object.

        forward(self, x)
            Computes the softplus activation function for the given input.

        __call__(self, x)
            Alias for the forward method.

    Reference:
        https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Softplus
    """

    def __init__(self):
        pass

    def forward(self, x):
        return (1 + x.exp(x)).log()

    def __call__(self, x):
        return self.forward(x)
