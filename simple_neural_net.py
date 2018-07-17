import numpy as np

# Definitining functions


def sigmoid(x):  # pass a normal function here to get its sigmoid transformation
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):  # pass a sigmoid function in here to get its derivative
    return sigmoid(x) * sigmoid(1 - x)


def run(x, O, W_1, W_2, b_1, b_2, learning_rate):
    # x = input; W_1 = matrix of weights; Y = output
    # forward prop
    A_0 = x.T
    Z_1 = np.dot(W_1, A_0) + b_1
    A_1 = sigmoid(Z_1)
    Z_2 = np.dot(W_2, A_1) + b_2
    Y = (sigmoid(Z_2)).T
    # backward prop
    # Following MSE as our loss function
    e = sigmoid(Z_2) - O.T  # overall error for the network
    # error signals (calculated according to MSE)
    d_2 = e
    d_1 = np.dot(W_2.T, d_2) * sigmoid_deriv(Z_1)
    # Update rule
    up_b1 = learning_rate * d_1
    up_b2 = learning_rate * d_2
    up_W_1 = learning_rate * np.dot(d_1, A_0.T)
    up_W_2 = learning_rate * np.dot(d_2, A_1.T)

    return Y, (up_W_1, up_W_2, up_b1, up_b2)


def train(x, O, W_1, W_2, b_1, b_2, n_iter, learning_rate):
    for i in range(n_iter):
        Y, upd = run(x, O, W_1, W_2, b_1, b_2, learning_rate)
        W_1 -= upd[0]
        W_2 -= upd[1]
        b_1 -= upd[2]
        b_2 -= upd[3]
        if i % 10 == 0:
            print(i)
            print((1 / np.size(O, 0)) * np.dot((Y - O).T, (Y - O)))

    return (W_1, W_2, b_1, b_2)


def predict(x, W_1, W_2, b_1, b_2):
    A_0 = x.T
    Z_1 = np.dot(W_1, A_0) + b_1
    A_1 = sigmoid(Z_1)
    Z_2 = np.dot(W_2, A_1) + b_2
    Y = (sigmoid(Z_2)).T
    return Y

# Setting up our data


inp = np.random.rand(100, 20)

out = np.random.rand(100, 1)


# Initializing parameters

n_nodes = 10
learning_rate = 0.5

w_1 = np.random.rand(n_nodes, np.size(inp, 1))
w_2 = np.random.rand(np.size(out, 1), n_nodes)
b1 = np.random.rand(n_nodes, np.size(inp, 0))
b2 = np.random.rand(np.size(out, 1), np.size(out, 0))

# Training
updated_params = train(inp, out, w_1, w_2, b1, b2, 1000, learning_rate)


# Testing
prediction = predict(inp, *updated_params)

print('Prediction: {0} \n Actual Output: {1} \n Error: {2}'.format(prediction.T, out.T, prediction.T - out.T))
# Success! It gave us the same output. (its a code wise success, may not be the best neural net though)

# You'll notice that the loss function doesn't go too low. Thats cause MSE isn't exactly the best loss function
# for binary prediction problems (apparantly)
