import numpy as np
from gradphi.autodiff import Variable

# class ReLU():
#     def __init__(self):
#         self.input_data = None

#     def forward(self, x):
#         self.input_data = x
#         f_x = np.maximum(0, x)
#         return f_x

#     def backward(self, grad=1):
#         if self.input_data is None:
#             raise ValueError("Forward must be called before backward.")
#         return grad * (self.input_data > 0)

#     def __call__(self, x):
#         return self.forward(x)

class ReLU(Variable): #Dynamic ReLU computation (slower than static)
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
