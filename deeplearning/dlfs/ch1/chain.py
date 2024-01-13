from typing import List
from numpy import array as ndarray
from typing import Callable
import numpy as np

# A Function takes in an ndarray as an argument and produces an ndarray
Array_Function = Callable[[ndarray], ndarray]

# A Chain is a list of functions
Chain = List[Array_Function]

def chain_length_3(chain: Chain,
                   x: ndarray) -> ndarray:
    '''
    Evaluates two functions in a row, in a "Chain".
    '''
    assert len(chain) == 3, \
    "Length of input 'chain' should be 3"

    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    return f3(f2(f1(x)))

def square(x: ndarray) -> ndarray:
    '''
    Square each element in the input Tensor.
    '''
    return np.power(x, 2)

def sigmoid(x: ndarray) -> ndarray:
    '''
    Apply the sigmoid function to each element in the input ndarray.
    '''
    return 1 / (1 + np.exp(-x))

def log(x: ndarray) -> ndarray:
    return np.log2(x)

def deriv(func: Callable[[ndarray], ndarray],
          input_: ndarray,
          diff: float = 0.001) -> ndarray:
    '''
    Evaluates the derivative of a function "func" at every element in the "input_" array.
    '''
    return (func(input_ + diff) - func(input_ - diff)) / (2 * diff)

def chain_deriv_2(chain: Chain, input_range: ndarray) -> ndarray:
    assert len(chain) == 3, \
    "This function requires 'Chain' objects of length 2"

    assert input_range.ndim == 1, \
    "Function requires a 1 dimensional ndarray as input_range"

    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    f1x = f1(input_range)
    f2_f1x = f2(f1x)
    df3_of_f2_f1x = deriv(f3, f2_f1x)
    df2_of_f1x = deriv(f2, f1x)
    df1 = deriv(f1, input_range) 

    return df3_of_f2_f1x * df2_of_f1x * df1

chain1 = [square, sigmoid, log]

derivative = chain_deriv_2(chain1, np.array([-1,0,1]))
print(derivative)

