from gradphi.autodiff import Variable


class Swish(Variable):
    def __init__(self, beta=1):
        self.beta = beta

    def forward(self, x):
        z = x.mul(self.beta)
        sigmoid = z.exp().div(z.exp().add(1))

        return x.mul(sigmoid)

    def __call__(self, x):
        return self.forward(x)
