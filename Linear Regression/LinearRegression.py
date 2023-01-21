import numpy as np

class LinearRegression:
  
  def __init__(self, lr, in_iters):
    self.lr = lr
    self.n_iters = n_iters
    self.bias = None
    self.weights = None
    
    
  def fit(self, X, y):
    #add bias in the data
    self.bias = 0 #bias will be zero
    n_sample, n_features = X.shape
    # initialize the weights
    self.weights = np.zeors(n_features)
    
    for _ in range(self.n_iters):
      # calculating predicted value
      '''
      y_pred = [w1x1 + w2x2 + w3x3 + .......... + wnxn] + b0
      w1, w2, w3... wn are weights and b0 is bias
      '''
      y_pred = np.dot(X, self.weights) + self.bias
      #residuals
      residuals = y_pred - np.array(y).reshape(-1,)
      #applying gradient descent for learning
      '''
      formula:
      updated_weights, w <-- w - lr * dw
      updated bias, b0 <-- b0 - lr * db
      '''
      dw = (1/n_samples) * np.dot(X.T, residuals)
      db = (1/n_samples) * np.sum(residuals)
      
      # updating weights
      self.weights -= (self.lr * dw)
      # updating bias
      self.bias -= (self.lr * db)
      
   return self.bias, self.weights   
      
      
     
  
