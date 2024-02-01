import numpy as np

np.random.seed(3)
LEARNING_RATE=0.1
index_list = [0,1,2,3]

x_train = [np.array([1.0, -1.0, -1.0]), np.array([1.0, -1.0, 1.0]), np.array([1.0, 1.0, -1.0]), np.array([1.0, 1.0, 1.0])]
y_train = [0.0, 1.0, 1.0, 0.0]

delta = 0.0001

def neuron1(x,w):
    return np.tanh(np.dot(x,w))

def neuron2(x,w):
    return np.tanh(np.dot(x,w))

def neuron3(x,w):
    z = np.dot(x,w)
    return 1.0 / (1.0 + np.exp(-z))

def errorFunc(y, expected):
    return (y-expected) * (y-expected)

n1_w = np.zeros(3)
n1_w[1] = np.random.uniform(-1.0,1.0)
n1_w[2] = np.random.uniform(-1.0,1.0)

n2_w = np.zeros(3)
n2_w[1] = np.random.uniform(-1.0,1.0)
n2_w[2] = np.random.uniform(-1.0,1.0)

n3_w = np.zeros(3)
n3_w[1] = np.random.uniform(-1.0,1.0)
n3_w[2] = np.random.uniform(-1.0,1.0)

for i in range(0,100):
    for j in range(0,4):
        n1_out = neuron1(x_train[j], n1_w)
        n2_out = neuron2(x_train[j], n2_w)
        n3_out = neuron3([1.0, n1_out, n2_out], n3_w)
        error = errorFunc(y_train[j], n3_out)
        print(x_train[j], "Expected", y_train[j], "Got", n3_out)

        dErrorBydN3 = (errorFunc(y_train[j], n3_out + delta) - error) / delta
        dN3BydN1 = (neuron3([1.0, n1_out+delta, n2_out], n3_w) - n3_out) / delta
        dN3BydN2 = (neuron3([1.0, n1_out, n2_out+delta], n3_w) - n3_out) / delta
        
        n3_w[0] += delta
        dN3ByN3W0 = (neuron3([1.0, n1_out, n2_out], n3_w) - n3_out) / delta
        n3_w[0] -= delta

        n3_w[1] += delta
        dN3ByN3W1 = (neuron3([1.0, n1_out, n2_out], n3_w) - n3_out) / delta
        n3_w[1] -= delta

        n3_w[2] += delta
        dN3ByN3W2 = (neuron3([1.0, n1_out, n2_out], n3_w) - n3_out) / delta
        n3_w[2] -= delta

        n2_w[0] += delta
        dN2ByN2W0 = (neuron2(x_train[j], n2_w) - n2_out) / delta
        n2_w[0] -= delta

        n2_w[1] += delta
        dN2ByN2W1 = (neuron2(x_train[j], n2_w) - n2_out) / delta
        n2_w[1] -= delta

        n2_w[2] += delta
        dN2ByN2W2 = (neuron2(x_train[j], n2_w) - n2_out) / delta
        n2_w[2] -= delta

        n1_w[0] += delta
        dN1ByN1W0 = (neuron1(x_train[j], n1_w) - n1_out) / delta
        n1_w[0] -= delta

        n1_w[1] += delta
        dN1ByN1W1 = (neuron1(x_train[j], n1_w) - n1_out) / delta
        n1_w[1] -= delta

        n1_w[2] += delta
        dN1ByN1W2 = (neuron1(x_train[j], n1_w) - n1_out) / delta
        n1_w[2] -= delta

        dErrorByN3W0 = dErrorBydN3 * dN3ByN3W0
        dErrorByN3W1 = dErrorBydN3 * dN3ByN3W1
        dErrorByN3W2 = dErrorBydN3 * dN3ByN3W2

        dErrorByN2W0 = dErrorBydN3 * dN3BydN2 * dN2ByN2W0
        dErrorByN2W1 = dErrorBydN3 * dN3BydN2 * dN2ByN2W1
        dErrorByN2W2 = dErrorBydN3 * dN3BydN2 * dN2ByN2W2

        dErrorBydN1W0 = dErrorBydN3 * dN3BydN1 * dN1ByN1W0
        dErrorBydN1W1 = dErrorBydN3 * dN3BydN1 * dN1ByN1W1
        dErrorBydN1W2 = dErrorBydN3 * dN3BydN1 * dN1ByN1W2

        n1_w = n1_w - (LEARNING_RATE * np.array([dErrorBydN1W0, dErrorBydN1W1, dErrorBydN1W2]))
        n2_w = n2_w - (LEARNING_RATE * np.array([dErrorByN2W0, dErrorByN2W1, dErrorByN2W2]))
        n3_w = n3_w - (LEARNING_RATE * np.array([dErrorByN3W0, dErrorByN3W1, dErrorByN3W2]))        
