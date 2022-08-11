# +
import autofunc.abstract as ab


class Example(ab.Elemental, ab.Generators):
    def __init__(self):
        self.a = self.tensor(2.0)
        self.b = self.tensor(3.0)

    def __call__(self, t: ab.Tensor) -> ab.Tensor:
        return self.a * self.sin(t) + self.b
