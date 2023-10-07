import numpy as np

from gradphi.autodiff import Variable


class MSELoss(Variable):
    """
    Mean Squared Error (MSE) loss function for regression tasks.

    The Mean Squared Error (MSE) loss function is a widely used metric in regression tasks,
    especially in machine learning and deep learning. It quantifies the average squared difference
    between predicted and actual values. The primary goal of MSE is to evaluate how well a model's
    predictions align with the true target values.

    Attributes:
        No attributes are specific to this class.

    Methods:
        forward(self, y_pred, y_true):
            Compute the forward pass of the MSE loss function.

        __call__(self, y_pred, y_true):
            Apply the MSE loss function to predicted and true values.

    Example:
        >>> mse_loss = MSELoss()
        >>> predicted_values = Variable(np.array([1.0, 2.0, 3.0]))
        >>> true_values = Variable(np.array([1.2, 1.8, 2.9]))
        >>> loss = mse_loss(predicted_values, true_values)
        >>> print(loss)
        0.014333333333333332

    References:
        - "Mean Squared Error" on Wikipedia (https://en.wikipedia.org/wiki/Mean_squared_error)
    """

    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        return (y_pred.sub(y_true)).power(2).sum().div(np.prod(y_pred.data.shape))

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)
