class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
    
    def to(self, device):
        for p in self.parameters():
            p.to(device)
        return self