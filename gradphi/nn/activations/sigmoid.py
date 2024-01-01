import numpy as np

from gradphi.autodiff import Variable


class Sigmoid(Variable):
    """
    """

    def __init__(self):
        pass

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def __call__(self, x):
        return self.forward(x)
