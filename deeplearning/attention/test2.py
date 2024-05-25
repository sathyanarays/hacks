from numpy import array
from numpy import random
from numpy import dot
from scipy.special import softmax
import numpy as np

word_1 = array([1,0,0])
word_2 = array([0,1,0])
word_3 = array([1,1,0])
word_4 = array([0,0,1])

# n = number of words
# d = word embedding vector dimension
words = array([word_1, word_2, word_3, word_4]) # n * d
print("=== Words ===")
print(words)

random.seed(42)
W_Q = random.randint(3, size=(3,3)) # d * d
W_K = random.randint(3, size=(3,3)) # d * d
W_V = random.randint(3, size=(3,3)) # d * d

print("=== W_Q ===")
print(W_Q)

print()
q = np.matmul(W_Q, words[-1])

print("=== q ===")
print(q)

n = len(words)

k = []
v = []
q = []
for i in range(n):
    k.append(np.matmul(W_K, words[i]))
    v.append(np.matmul(W_V, words[i]))
    q.append(np.matmul(W_Q, words[i]))

k = array(k)
v = array(v)
q = array(q)

print("\n=== k ===")
print(k)

print("\n=== v ===")
print(v)

print("\n=== q ===")
print(q)


A = np.zeros(shape=(n,n))
sqrt_d = np.sqrt(3)
for i in range(len(words)):
    for j in range(len(words)):
        qDotKj = np.dot(q[i],k[j])
        numerator = np.exp(qDotKj / sqrt_d)
        denominator = 0.0
        for t in range(0,i+1):
            qDotKt = np.dot(q[i],k[t])
            partial_result = np.exp(qDotKt / sqrt_d)            
            denominator += partial_result
        
        res = numerator / denominator
        A[i][j] = res

print("\n=== A ===")
print(A)



