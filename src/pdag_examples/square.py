import pdag
from typing import Annotated


class SquareModel(pdag.Model):
    x = pdag.RealParameter("x")
    y = pdag.RealParameter("y")

    @pdag.relationship
    @staticmethod
    def square(*, x: Annotated[float, x.ref()]) -> Annotated[float, y.ref()]:
        return x**2
