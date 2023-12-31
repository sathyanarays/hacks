import numpy

array = numpy.array([1,2,3,0,-1,-2,-3])

def square(x: numpy.array) -> numpy.array:
    return numpy.power(x,2)

def leaky_relu(x: numpy.array) -> numpy.array:
    return numpy.maximum(0, x)

print(square(array))
print(leaky_relu(array))