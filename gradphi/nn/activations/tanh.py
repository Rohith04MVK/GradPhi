from gradphi.autodiff import Variable


class Tanh(Variable):
    def __init__(self):
        pass

    def forward(self, x):
        return x.exp().sub(x.mul(-1).exp()).div(x.exp().add(x.mul(-1).exp()))

    def __call__(self, x):
        return self.forward(x)
