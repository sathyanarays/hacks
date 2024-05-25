from numpy import array
from numpy import random
from numpy import dot
from scipy.special import softmax
import numpy as np

word_1 = array([1,0,0])
word_2 = array([0,1,0])
word_3 = array([1,1,0])
word_4 = array([0,0,1])

words = array([word_1, word_2, word_3, word_4]).transpose()

random.seed(42)
W_Q = random.randint(3, size=(3,3)).transpose()
W_K = random.randint(3, size=(3,3)).transpose()
W_V = random.randint(3, size=(3,3)).transpose()

print(words[0])

Q = np.matmul(W_Q, words)
K = np.matmul(W_K, words)
V = np.matmul(W_V, words)

print("=== Q ===")
print(Q)
print("=== K ===")
print(K)
print("=== V ===")
print(V)


scores = np.matmul(Q.transpose(), K)
print("=== scores ===")
print(scores)

weights = softmax(scores / K.shape[0] ** 0.5, axis=1)
print("=== weights ===")
print(weights)

attention = np.matmul(weights, V.transpose())
print("=== attention ===")
print(attention)
