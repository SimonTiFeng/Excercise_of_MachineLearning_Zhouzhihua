import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import scipy.optimize as opt

# Load data
data = pd.read_csv('3.0a.csv').to_numpy()

# Data preprocessing
y = data[:, -1]
X = data[:, 1:3]
X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias term

# Initialize beta
beta = np.zeros(X.shape[1])

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient function
def gradient(beta, X, y):
    y_forecast = sigmoid(np.dot(X, beta))
    error = y_forecast - y
    gradient = np.dot(X.T, error)
    return gradient

# Log-likelihood function 
def log_likelihood(beta, X, y):
    y_forecast = sigmoid(np.dot(X, beta))
    epsilon = 1e-9  # To avoid log(0)
    log_likelihood = -np.sum(y * np.log(y_forecast + epsilon) + (1 - y) * np.log(1 - y_forecast + epsilon))
    return log_likelihood
result = opt.minimize(fun=log_likelihood, x0=beta, args=(X, y), method='BFGS', jac=gradient)

# Get optimized beta
beta_optimized = result.x
print("Optimized beta coefficients:", beta_optimized)

# Visualizing the decision boundary
def plot_decision_boundary(X, y, beta):
    plt.figure(figsize=(10, 6))
    
    plt.scatter(X[y == 0][:, 1], X[y == 0][:, 2], label='Class 0', color='red', edgecolor='k')
    plt.scatter(X[y == 1][:, 1], X[y == 1][:, 2], label='Class 1', color='blue', edgecolor='k')
    x_values = [np.min(X[:, 1]), np.max(X[:, 1])]
    y_values = -(beta_optimized[0] + np.dot(beta_optimized[1], x_values)) / beta_optimized[2]
    
    plt.plot(x_values, y_values, label='Decision Boundary', color='green')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

# Plot the decision boundary
plot_decision_boundary(X, y, beta_optimized)
