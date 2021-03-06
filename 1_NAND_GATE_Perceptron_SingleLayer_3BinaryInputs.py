import numpy as np

input = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
labels = np.array([1,1,1,1,1,1,1,0])
labels = labels.reshape(8,1)

# Define weights, bias and learning rate
np.random.seed(0)
weights = np.random.rand(3,1)
bias = np.random.rand(1)
lr = 0.05

def sigmoid(x):
  return 1/(1+np.exp(-x))
# Derivative of sigmoid function:
def sigmoid_der(x):
  return sigmoid(x)*(1-sigmoid(x))

def threshold(y):
  if(y>=0.5):
    return 1
  else:
    return 0

for epoch in range(10000):
  inputs = input

  #Feedforward input:
  XWB = np.dot(inputs, weights) + bias 
  #Feedforward output:
  z = sigmoid(XWB) 
  
  #Backpropogation
  error = z - labels
  sum_error = error.sum()
    
  # backpropagation step 2
  dcost_dpred = error
  dpred_dz = sigmoid_der(z)
  dz = dcost_dpred * dpred_dz

  inputs = input.T
  weights -= lr * np.dot(inputs, dz)

  for num in dz:
    bias -= lr * num

print(sum_error)

single_point = np.array([0,0,1]) #1st step:
result1 = np.dot(single_point, weights) + bias #2nd step:
result2 = sigmoid(result1) #Print final result
print(result2,threshold(result2))

#Taking inputs:
single_point = np.array([0,1,1]) #1st step:
result1 = np.dot(single_point, weights) + bias #2nd step:
result2 = sigmoid(result1) #Print final result
print(result2,threshold(result2))

#Taking inputs:
single_point = np.array([1,0,1]) #1st step:
result1 = np.dot(single_point, weights) + bias #2nd step:
result2 = sigmoid(result1) #Print final result
print(result2,threshold(result2))

#Taking inputs:
single_point = np.array([1,1,1]) #1st step:
result1 = np.dot(single_point, weights) + bias #2nd step:
result2 = sigmoid(result1) #Print final result
print(result2,threshold(result2))
