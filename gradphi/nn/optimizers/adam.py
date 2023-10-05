import numpy as np


class Adam:
    """
    Adam optimizer for neural network training.

    Adam (short for Adaptive Moment Estimation) is an optimization algorithm commonly used for training
    deep learning models. It combines the advantages of both the RMSprop and momentum optimization techniques
    and has demonstrated substantial improvements in training deep neural networks.

    Adam computes adaptive learning rates for each parameter by considering both the first-order momentum
    estimation and the second-order moment estimation. This combination allows Adam to handle noisy gradients
    and non-stationary objectives effectively. The algorithm has two hyperparameters, β₁ and β₂, that control
    the decay rates of the moving averages. Adam is computationally efficient and has been widely adopted
    in the deep learning community.

    Args:
        params (list of Tensors): The list of model parameters to optimize.
        lr (float, optional): The learning rate (default is 0.01).
        betas (tuple of two floats, optional): Coefficients for the exponentially moving averages of
            gradient (beta1) and squared gradient (beta2) (default is (0.9, 0.999)).
        epsilon (float, optional): A small constant to prevent division by zero (default is 1e-8).

    Attributes:
        params (list of Tensors): The list of model parameters to optimize.
        lr (float): The learning rate.
        betas (tuple of two floats): Coefficients for the exponentially moving averages of gradient and squared gradient.
        epsilon (float): A small constant to prevent division by zero.
        m (list of Tensors): Exponentially moving average of the gradient.
        v (list of Tensors): Exponentially moving average of the squared gradient.
        t (int): Time step counter.

    Methods:
        step(self):
            Perform one optimization step using the Adam algorithm.

        zero_grad(self):
            Zero out the gradients of all the model parameters.

    Example:
        >>> model = YourNeuralNetwork()
        >>> optimizer = Adam(model.parameters(), lr=0.001)
        >>> loss_fn = YourLossFunction()
        >>> for epoch in range(num_epochs):
        >>>     for batch_data in data_loader:
        >>>         optimizer.zero_grad()
        >>>         outputs = model(batch_data)
        >>>         loss = loss_fn(outputs, batch_data.labels)
        >>>         loss.backward()
        >>>         optimizer.step()

    References:
        - Adam: A Method for Stochastic Optimization (https://arxiv.org/abs/1412.6980)
    """

    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), epsilon=1e-8):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.epsilon = epsilon

        self.m = [np.zeros_like(param.data) for param in self.params]
        self.v = [np.zeros_like(param.data) for param in self.params]

        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            self.m[i] = self.betas[0] * self.m[i] + \
                (1 - self.betas[0]) * param.grad
            self.v[i] = self.betas[1] * self.v[i] + \
                (1 - self.betas[1]) * param.grad ** 2

            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)

            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def zero_grad(self):
        for param in self.params:
            param.grad = None if param.grad is None else np.zeros_like(
                param.grad)
