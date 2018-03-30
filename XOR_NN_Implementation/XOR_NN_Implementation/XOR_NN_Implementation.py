import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoidDeriv(x):
    return x*(1-x) 

input_size = 3
output_size = 1
hidden_size = 6

X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
Y = np.array([[0],
			[1],
			[1],
			[0]])

Test_Set = np.array([[0, 0, 0],
[1, 1, 0],
[0, 1, 0],
[1, 0, 0]
])

Test_Set_Y = np.array([[0],
[0],
[1],
[1]
])

indices = np.arange(4)

np.random.seed(1)

alpha = 10

weights_0_1 = 2 * np.random.random((input_size, hidden_size)) - 1
weights_1_2 = 2 * np.random.random((hidden_size, output_size)) - 1

for i in range(90000):
    np.random.shuffle(indices)
    global_error = 0
    for j in range(indices.size):
        
        l0 = np.array(X[indices[j]],  ndmin=2)
        Y_hat = np.array(Y[indices[j]],  ndmin=2)
        l1 = sigmoid(l0.dot(weights_0_1))
        l2 = sigmoid(l1.dot(weights_1_2))

        error = (Y_hat - l2)**2
        global_error += error

        l2_delta = (l2 - Y_hat) * sigmoidDeriv(l2)

        l1_delta = (l2_delta.dot(weights_1_2.T)) * sigmoidDeriv(l1)
            
        weights_0_1_gradient = l0.T.dot(l1_delta)
        weights_1_2_gradient = l1.T.dot(l2_delta)

        weights_0_1 -= alpha * weights_0_1_gradient
        weights_1_2 -= alpha * weights_1_2_gradient

    global_error = global_error/indices.size
    
    if(i%10000 == 0):
        print("global error: " + str(global_error))

global_test_error = 0

for i in range(4):

    l0 = np.array(Test_Set[i], ndmin=2)
    Y_hat = np.array(Test_Set_Y[i], ndmin=2)

    l1 = sigmoid(l0.dot(weights_0_1))
    l2 = sigmoid(l1.dot(weights_1_2))

    error = (Y_hat - l2)**2
    global_test_error += error

    print("prediction: " + str(l2) + " actual: " + str(Y_hat) + " error: " + str(error))

global_test_error = global_test_error/Test_Set.size

print("global error on test set: " + str(global_test_error))
    