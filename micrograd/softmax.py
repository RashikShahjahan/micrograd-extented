from module import Module

class Softmax(Module):
    def __call__(self, x):
        e_x = (x - x.max(axis=1, keepdims=True)).exp()
        return e_x / e_x.sum(axis=1, keepdims=True)

    def __repr__(self):
        return "Softmax()"