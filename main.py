import numpy as np
import sklearn.datasets as datasets
import sys


#===============================
# Command Line Arguments
#===============================

# #1 Train Features
#----------------------
    # (str), the name of the training set feature file
    # The file should contain N lines (N is number of data points)
    # Each line should contain D space-delimited floating point values (where D is the feature dimension).
trainF = sys.argv[1]

trainF = np.loadtxt(trainF)
train_n = trainF.shape[0]

if len(trainF.shape) == 1:
    D = 1
elif len(trainF.shape) == 2:
    D = trainF.shape[1]
else:
    print("Train Features Dataset error")

# print("Train Features, ", trainF.shape)

# #2 Train Targets
#----------------------
trainT = sys.argv[2]

trainT = np.loadtxt(trainT)
# print("Train Targets,  ", trainT.shape)
# print(type(trainT[0]))

# #3 Dev Features
#----------------------
devF = sys.argv[3]

devF = np.loadtxt(devF)
dev_n = devF.shape[0]
# print(dev_n)
if len(devF.shape) == 1:
    D = 1
elif len(devF.shape) == 2:
    D = devF.shape[1]
else:
    print("Dev Features Dataset error")

full_n = max(dev_n, train_n)

# print("Dev Features,   ", devF.shape)

# #4 Dev Targets
#----------------------
devT = sys.argv[4]

devT = np.loadtxt(devT)
# print("Dev Targets,    ", devT.shape)

# #5 Num of Hidden Units
#----------------------
L = int(sys.argv[5])

# #6 Num of Hidden Layers
#----------------------
hiddenLayers = int(sys.argv[6])

# #7 Hidden Unit Activation
#----------------------
def activation(f):
    if sys.argv[7] == "sig":
        return sigmoid(f)
    elif sys.argv[7] == "tanh":
        return tanh(f)
    elif sys.argv[7] == "relu":
        return relu(f)
    else:
        print("No activation function specified")

def activationPrime(f):
    if sys.argv[7] == "sig":
        return sigmoidPrime(f)
    elif sys.argv[7] == "tanh":
        return tanhPrime(f)
    elif sys.argv[7] == "relu":
        return reluPrime(f)
    else:
        print("No activation function specified")

# #8 Problem Mode
#----------------------
mode = sys.argv[8]

# #9 Output dimension
#----------------------
C = int(sys.argv[9])


# #10 Total Updates
#----------------------
totalUpdates = int(sys.argv[10])

# #11 Learn Rate
#----------------------
learnRate = float(sys.argv[11])

# #12 Initialization Range
#----------------------
initialRange = float(sys.argv[12])

# #13 Minibatch Size
#----------------------
MB_size = int(sys.argv[13])

if MB_size == 0:
    MB_size = train_n

# #14 Report Frequency
#----------------------
reportFrequency = int(sys.argv[14])

# #15 Verbose Mode
#----------------------
if sys.argv[15] == "True":
    verboseMode = True
elif sys.argv[15] == "False":
    verboseMode = False
else:
    print("Verbose Argument not given")


#=============================
# Loss and Scoring Functions
#=============================

def MSE(y, y_pred, n):
    return (1/n)*((y-y_pred)**2).sum()

def BinaryCrossEntropy(y, y_pred, n):
    return (1/n) * -y*np.log(y_pred) - (1-y)*np.log(1-y_pred).sum()

def MultiCrossEntropy(y, y_pred):
    return -((y.T*np.log(y_pred)).sum())

def accuracy(y, y_pred, n):
    y_pred = y_pred.T
    y_pred = (y_pred == y_pred.max(axis=1, keepdims=True))
    y_pred = y_pred.T.astype(int)
    y_pred = np.where(y_pred, 1, 0)
    # print("accuracy shape: ", y_pred.shape)
    # print("y shape: ", y.T.shape)

    y = y.astype(int)

    y_pred_sum = y_pred & y.T
    # y_pred_sum = np.bitwise_and(y_pred, y)
    total_correct  = sum(sum(y_pred_sum))
    # print("total correct", total_correct, "\nn = ", n)
    # print("total_correct %", total_correct/n)

    return (total_correct/n).round(3)







#=============================
# Transformation Functions
#=============================

#--------------
# Reshuffle
#--------------

def reshuffle(matrix):
    back = 0
    front = MB_size
    MB_Array = []

    np.random.shuffle(matrix)
    # print(matrix)

    while front < (len(matrix)+1):
        # print(matrix[back:front, :])
        # print()
        MB_Array.append(matrix[back:front, :])

        back += MB_size
        front += MB_size
    
    return MB_Array

#--------------
# one hot encoding
#--------------
def oneHot(y):
    return np.eye(C)[y].T

#--------------
# transpose
#--------------

def transpose(x):
    if len(x.shape) == 1:
        return x.reshape(1, -1)
    else:
        return x.T

#----------------------
# Linear Regression
#----------------------

def zVector(X, W, B):
    return (W.T@X)+B

#----------------------
# Logistic Regression
#----------------------




#==============================
# Activation Functions
#==============================

#----------------------
# ReLU
#----------------------
# works with both ints and matricies
def relu(z):
    return np.maximum(0, z)

def reluPrime(z):
    return np.where(z > 0, 1, 0)    #return 1 when positive and 0 when negative

#----------------------
# sigmoid
#----------------------
def sigmoid(z):
    return 1 / (1 + np.e**-z)
    

def sigmoidPrime(z):
    return sigmoid(z) * (1 - sigmoid(z))

#----------------------
# tanh
#----------------------
def tanh(z):
#     return 2 * sigmoid(z) - 1
    return np.tanh(z)

def tanhPrime(z):
#     return 1 - (tanh(z))**2
    return 1 - np.tanh(z)**2

#----------------------
# SOFTMAX
#----------------------

def SOFTMAX(z):
    return (np.e**z) / sum(np.e**z)


#===============================
# Gradients
#===============================

# output dim: C x n
def deltaL(Y, Y_pred):
    return Y_pred - transpose(Y)

# output dim: Lk x n
# k = 1 example: f'(z1) o ( W2 * delta2)
def deltaK(z, W, delta): # W = Wk+1, delta = delta_k+1
    return activationPrime(z) * (W @ delta)

# output dim: Lk-1 x Lk
# shortcut: dim of prev. layer by dim of next layer
def W_grad(n, A, delta): #A = Ak-1  | Delta = deltak
    return (1/n) * (A @ transpose(delta))


# Bias calculations     [WORKS]
#------------------------------

# outputs b_grad vector (Lk, )
# delta = [Lk, n]
def b_grad(n, delta):
    grad_bk = delta.sum(axis=1, keepdims=True) / n    # (Lk, 1)
    grad_bk = grad_bk.reshape(-1)   #Reshapes from (Lk, 1) to (Lk, ) for broadcasting
    return grad_bk

# broadcasts b_grad into B_matrix [Lk, n]
def B_matrix(n, b_grad):
    zeros = np.zeros((b_grad.shape[0], n)).T
    B = (zeros + b_grad).T # [Lk, n]
    return B

# takes delta [Lk, n]
# returns B_grad [Lk, n]
def B(n, delta):
    return B_matrix(n, b_grad(n, delta))




#===============================
# Bias and Weight matricies
#===============================
# initializes a weight matrix
def createWeights(D, L):
    return np.random.uniform(-initialRange, initialRange, (D, L))

# initializes a bias vector
def createBias(L, n):
    return B_matrix(n, np.random.uniform(-initialRange, initialRange, (L)))


#===============================
# New Bias and Weight matricies
#===============================

def adjustWeights(n, A, delta, W):
    return W - (learnRate * W_grad(n, A, delta))

# inputs a b-vector
# returns b matrix
def adjustBiases(n, delta, b_vector):
    new_biases = b_vector - (learnRate * b_grad(n, delta))
    return B_matrix(n, new_biases)

#Replaces devBArray with updated biases from BArray to the size dev_n
# devBArray = [(L x dev_n), (L x dev_n), ..., (C x dev_n)]
def update_devBArray(BArray):
    for i in range(len(BArray)):
        devBArray[i] = B_matrix(dev_n, BArray[i].T[0])
    




#===============================
# Forwardpass and Backpass
#===============================

#returns nothing, updates WArray, BArray, Astack, and zStack
def forwardpass(X_data, n, Astack, zStack):

    if D == 1:
        A0 = X_data.reshape(1, -1)  #Turning (n,) -> (n, 1)
    else:
        A0 = X_data.T

    Astack.append(A0)

    for i in range(hiddenLayers+1):
        # print("LOOP: ", i)
        #Initializes WArray
        if WArray[i] is None and i != hiddenLayers:
            # print("test 1")
            WArray[i] = np.random.uniform(-initialRange, initialRange, (Astack[-1].shape[0], L))
        elif WArray[i] is None and i == hiddenLayers:
            # print("test 1")
            WArray[i] = np.random.uniform(-initialRange, initialRange, (Astack[-1].shape[0], C))
        #Initializes BArray 
        if BArray[i] is None and i != hiddenLayers:
            # print("test 2")
            BArray[i] = createBias(L, n)
        elif BArray[i] is None and i == hiddenLayers:
            # print("test 2")
            BArray[i] = createBias(C, n)

        z = zVector(Astack[-1], WArray[i], BArray[i][:, :n]) #X, W, B
        zStack.append(z)
        Astack.append(activation(z))

    return Astack, zStack

#returns nothing, updates WArray, BArray, and clears Astack & zStack
def backwardpass(y_data, y_pred, n, Astack, zStack):

    #updates last weight in weight matrix and pops from the Astack
    A_k_minus_one = Astack.pop()
    lastCalculatedDelta = deltaL(y_data, y_pred)  #saves the last calculated delta
    # print("y_data", y_data)
    # print("delta", lastCalculatedDelta)
    WArray[-1] = adjustWeights(n, A_k_minus_one, lastCalculatedDelta, WArray[-1])
    BArray[-1] = adjustBiases(n, lastCalculatedDelta, BArray[-1][:, 0])

    for j in range(hiddenLayers):
        zCurrent = zStack.pop()
        A_k_minus_one = Astack.pop()
        lastCalculatedDelta = deltaK(zCurrent, WArray[hiddenLayers-j], lastCalculatedDelta)

        WArray[hiddenLayers-j-1] = adjustWeights(n, A_k_minus_one, lastCalculatedDelta, WArray[hiddenLayers-j-1])
        BArray[hiddenLayers-j-1] = adjustBiases(n, lastCalculatedDelta, BArray[hiddenLayers-j-1][:, 0])

# returns loss
def lossCalculations(y_data, y_pred, n):
    if mode == "R":
        return MSE(y_data.T, y_pred, n)
    elif (mode == "C") and (C == 2):
        return BinaryCrossEntropy(y_data, y_pred, n)
    elif (mode == "C") and (C > 2):
        return accuracy(y_data, y_pred, n)
    else:
        print("lossCalculation Error")














#===============================
# Neural Network
#===============================

WArray = np.empty(hiddenLayers+1, dtype = object)
BArray = np.empty(hiddenLayers+1, dtype = object)
devBArray = np.empty(hiddenLayers+1, dtype = object)



#data_F = [n x D],     
# in R data_T = [n x C], 
# in C data_T = (n, )
def neuralNetwork(data_F, data_T, n):
    # print("n", n)
    if mode == "C":
        data_T = data_T.astype(int)
        yHot = oneHot(data_T).T
        # print(yHot.shape)

    Astack = []
    zStack = []
    

    for u in range(1): #Left in here for testing
        Astack, zStack = forwardpass(data_F, n, Astack, zStack)
        
        y_pred = Astack.pop()

        if mode == "C":
            if C == 2:
                y_pred = sigmoid(y_pred)
            elif C > 2:
                y_pred = SOFTMAX(y_pred)
                data_T = yHot

        loss = lossCalculations(data_T, y_pred, n)
        # print("Loss: ", loss)

        zStack.pop()

        backwardpass(data_T, y_pred, n, Astack, zStack)
    return loss

# neuralNetwork(trainF, trainT, train_n)





def devAssesment(X_dev, Y_dev, n):

    # print("devin", n)

    temp_Astack = []
    temp_zStack = []

    if mode == "C":
        Y_dev = Y_dev.astype(int)
        yHot = oneHot(Y_dev).T

    if D == 1:
        A0 = X_dev.reshape(1, -1)  #Turning (n,) -> (n, 1)
    else:
        A0 = X_dev.T

    temp_Astack.append(A0)

    # temp_Astack, temp_zStack = forwardpass(X_dev, n, temp_Astack, temp_zStack)
    for i in range(hiddenLayers+1):
        z = zVector(temp_Astack[-1], WArray[i], devBArray[i][:, :n]) #X, W, B
        temp_zStack.append(z)
        temp_Astack.append(activation(z))
    
    y_pred = temp_Astack[-1]
    
    if mode == "C":
        if C == 2:
            y_pred = sigmoid(y_pred)
        elif C > 2:
            y_pred = SOFTMAX(y_pred)
            Y_dev = yHot

    loss = lossCalculations(Y_dev, y_pred, n)
    return loss


   

    


   






#===============================
# Mini Batch
#===============================
X_data = trainF
Y_data = trainT
if len(Y_data.shape) == 1:
    Y_data = Y_data.reshape(len(Y_data), 1)
if len(X_data.shape) == 1:
    X_data = X_data.reshape(len(X_data), 1)

Y_data_width = Y_data.shape[1]

fullds = np.concatenate((X_data, Y_data), axis = 1)
# print("fullds: ", fullds.shape)

# print(fullds)

MB_Array = reshuffle(fullds)

# print(MB_Array)

Update_counts = 0
Epoch_count = 0

# print(len(MB_Array))

while Update_counts < totalUpdates:
    MB_Array = reshuffle(fullds)
    for m in range(len(MB_Array)):
        if Update_counts == totalUpdates:
            Epoch_count -=1
            break

        brokenX = MB_Array[0][:, :-Y_data_width]
        brokenY = MB_Array[0][:, -Y_data_width:].squeeze()
        loss = neuralNetwork(brokenX, brokenY, len(brokenX))

        # print("Barray after NN", BArray)
        
        #for the initial (SORRY)
        if Update_counts == 0:
            print("Epoch ", str(Epoch_count).zfill(4), "  UPDATE ", str(Update_counts).zfill(6), ":", sep="", end="")
            if verboseMode == True:
                print("  minibatch = ", loss.round(3), end="")

            update_devBArray(BArray)
            devLoss = devAssesment(devF, devT, dev_n)
            print("\tdev = ", devLoss.round(3))


        #For every other one
        Update_counts += 1
        if Update_counts % reportFrequency == 0:
            print("Epoch ", str(Epoch_count).zfill(4), "  UPDATE ", str(Update_counts).zfill(6), ":",  sep="", end="")
            if verboseMode == True:
                print("  minibatch = ", loss.round(3), end="")
            update_devBArray(BArray)
            devLoss = devAssesment(devF, devT, dev_n)
            print("\tdev = ", devLoss.round(3))

        
    Epoch_count +=1
    
    




#===============================
# MAIN
#===============================
#for test Lk= = 6, n = 10
# Lk = 6
# n = 10

# Delta = np.random.randint(1, 10, (6, 10))
# bias = np.random.randint(1, 10, 6)

# print(bias)
# print(adjustBiases(train_n, Delta, bias))

# print("b", b_grad(n, Delta))

# print(B(n, Delta))