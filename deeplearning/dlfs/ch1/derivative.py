from typing import Callable
from numpy import array as ndarray
import numpy

def deriv(func: Callable[[ndarray], ndarray], input_: ndarray, delta: float = 0.001) -> ndarray:
    return (func(input_  + delta) - func(input_ - delta)) / (2 * delta)

def square(x: ndarray) -> ndarray:
    return numpy.square(x)

print(deriv(square, numpy.array([1,2,3]), 0.001))