import numpy as np

from gradphi.autodiff import Variable


class MSELoss(Variable):
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        return (y_pred.sub(y_true)).power(2).sum().div(np.prod(y_pred.data.shape))

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)
