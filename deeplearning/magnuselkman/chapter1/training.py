from perceptron import *
import random

def show_learning(w):
    print('w0=', '%5.2f' % w[0], 'w1=', '%5.2f' % w[1], 'w2=', '%5.2f' % w[2])

random.seed(7)
LEARNING_RATE=0.1
index_list = [0,1,2,3]

x_train = [(1.0, -1.0, -1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0), (1.0, 1.0, 1.0)]
y_train = [1.0, 1.0, 1.0, 1.0]

w = [0.2, -0.6, 0.25]

show_learning(w)

all_correct = False
while not all_correct:
    all_correct = True
    random.shuffle(index_list)
    for i in index_list:
        x = x_train[i]
        y = y_train[i]
        p_out = compute_output(w,x)

        if y != p_out:
            for j in range(0, len(w)):
                w[j] += (y * LEARNING_RATE * x[j])
            all_correct = False
            show_learning(w)