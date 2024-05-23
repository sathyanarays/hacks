from numpy import array
from numpy import random
from numpy import dot
from scipy.special import softmax
import numpy as np

word_1 = array([1,0,0])
word_2 = array([0,1,0])
word_3 = array([1,1,0])
word_4 = array([0,0,1])

words = array([word_1, word_2, word_3, word_4, word_1, word_2])

random.seed(42)
W_Q = random.randint(3, size=(3,3))
W_K = random.randint(3, size=(3,3))
W_V = random.randint(3, size=(3,3))

Q = words @ W_Q
K = words @ W_K
V = words @ W_V

print("### Q ###")
print(Q)

print("### Transpose(K) ###")
print(K.transpose())

scores = Q @ K.transpose()
print("### Scores ###")
print(scores)

weights = softmax(scores / K.shape[1] ** 0.5, axis=1)
print(weights)
print(V)

attention = weights @ V
print(attention)

