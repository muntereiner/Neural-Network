import numpy as np

#n queens solution finder
x = {}
n = 4

def place(k, i):
    if (i in x.values()):
        return False
    j = 1
    while(j < k):
        if abs(x[j]-i) == abs(j-k):
            return False
        j+=1
    return True

def clear_future_blocks(k):
    for i in range(k,n+1):
       x[i]=None

def NQueens(y, k):
    for i in range(1, n + 1):
        clear_future_blocks(k)
        if place(k, i):
            x[k] = i
            if (k==n):
                X = []
                for j in x:
                    X += [x[j]]
                y += [X]
            else:
                NQueens(y, k+1)

# board state generation
def step(l):
    for i in range(0, n):
        if (l[i] < n):
            l[i] += 1
            return l
        else:
            l[i] = 1

def generator():
    start = []
    end = []
    l = []
    for i in range(0, n):
        start += [1]
        end += [n]
    l = start
    result = []
    while (not (l == end)):
        result += [l]
        temp = step(l)
        l = []
        l += temp
        #l = step(l)
    return result
    
def toMatrix(X, y):
    sol = []
    NQueens(sol, 1)
    
    g = generator()
    for i in g:
        l = []
        for j in range(0, n):
            queenPos = i[j]
            for k in range(1, n + 1):
        		if (queenPos == k):
        		    l += [1]
        		else:
        		    l += [0]
        X += [l]
        
        validSol = False
        for j in sol:
            if (i == j):
                validSol = True
        if (validSol):
        	y += [1]
        else:
        	y += [0]

input = []
output = []
toMatrix(input, output)

# define the sigmoid function
def sigmoid(x, derivative=False):

    if (derivative == True):
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))

# choose a random seed for reproducible results
np.random.seed(1)

# learning rate
alpha = 0.1

# number of nodes in the hidden layer
num_hidden = pow(n, 2)

# inputs
X = np.array(input)

# outputs
# x.T is the transpose of x, making this a column vector
y = np.array([output]).T

# initialize weights randomly with mean 0 and range [-1, 1]
# the +1 in the 1st dimension of the weight matrices is for the bias weight
hidden_weights = 2*np.random.random((X.shape[1] + 1, num_hidden)) - 1
output_weights = 2*np.random.random((num_hidden + 1, y.shape[1])) - 1

# number of iterations of gradient descent
num_iterations = 10000

file = open("runtime.txt", "w")

# for each iteration of gradient descent
for i in range(num_iterations):

    # forward phase
    # np.hstack((np.ones(...), X) adds a fixed input of 1 for the bias weight
    input_layer_outputs = np.hstack((np.ones((X.shape[0], 1)), X))
    hidden_layer_outputs = np.hstack((np.ones((X.shape[0], 1)), sigmoid(np.dot(input_layer_outputs, hidden_weights))))
    output_layer_outputs = np.dot(hidden_layer_outputs, output_weights)

    # backward phase
    # output layer error term
    output_error = output_layer_outputs - y
    # hidden layer error term
    # [:, 1:] removes the bias term from the backpropagation
    hidden_error = hidden_layer_outputs[:, 1:] * (1 - hidden_layer_outputs[:, 1:]) * np.dot(output_error, output_weights.T[:, 1:])

    # partial derivatives
    hidden_pd = input_layer_outputs[:, :, np.newaxis] * hidden_error[: , np.newaxis, :]
    output_pd = hidden_layer_outputs[:, :, np.newaxis] * output_error[:, np.newaxis, :]

    # average for total gradients
    total_hidden_gradient = np.average(hidden_pd, axis=0)
    total_output_gradient = np.average(output_pd, axis=0)

    # update weights    
    hidden_weights += - alpha * total_hidden_gradient
    output_weights += - alpha * total_output_gradient
    
    for v in hidden_weights:
    	for w in v:
    		file.write(str(w) + "\n")

    for e in output_error:
    	file.write(str(e) + "\n")
    	
    for w in output_weights:
    	file.write(str(w) + "\n")

#sum = 0
#for e in output_error:
#	sum += pow(e, 2)
#print(sum)

file.close()

# print the final outputs of the neural network on the inputs X
#print("Output After Training: \n{}".format(output_layer_outputs))