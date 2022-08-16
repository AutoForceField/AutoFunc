# +
import autofunc.abstract as ab


class Example(ab.Abs):
    def __init__(self):
        self.a = self.tens(2.0)
        self.b = self.tens(3.0)

    def __call__(self, t: ab.Tens) -> ab.Tens:
        return self.a * self.sin(t) + self.b
