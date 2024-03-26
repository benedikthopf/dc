class UEMA:
    def __init__(self, alpha=0.99):
        self.alpha = alpha
        self.value = 0
        self.num_values = 0

    def update(self, x):
        x = float(x)
        self.value = self.alpha * self.value + (1 - self.alpha) * x
        self.num_values = self.alpha * self.num_values + (1 - self.alpha)

        return self.get()

    def get(self):
        return self.value / self.num_values

    def __str__(self):
        return f"{self.get():.6f}"