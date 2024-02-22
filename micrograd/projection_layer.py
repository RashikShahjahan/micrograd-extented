from module import Module
from linear import Linear
from softmax import Softmax

class ProjectionLayer(Module):
    def __init__(self, d_model, d_vocab):
        self.linear = Linear(d_model, d_vocab)
        self.softmax = Softmax()

    def __call__(self, x):
        return self.softmax(self.linear(x))
    def parameters(self):
        return self.linear.parameters()

    def __repr__(self):
        return f"ProjectionLayer({self.linear})"