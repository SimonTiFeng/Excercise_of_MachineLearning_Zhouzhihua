import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

# Load data
data = pd.read_csv('Excercise_of_MachineLearning_Zhouzhihua/3.0a.csv')
data = data.values

# Data preprocessing
X = data[:, 1:3]  # Features (17 samples, 2 features)
y = data[:, 3]    # Target (17 samples)

# Initialize parameters
miu = np.zeros([X.shape[1], X.shape[1]])  # miu is (2, 2)
gama = np.zeros([1, X.shape[1]])  # gama is (1, 2)
omega = np.zeros([X.shape[1], 1])  # omega is (2, 1)
theta = 0  # Scalar bias for output

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Flatten parameters into a single vector for optimization
def flatten_combination(miu, gama, omega, theta):
    combined = np.concatenate((miu.flatten(), gama.flatten(), omega.flatten(), np.array([theta])))
    return combined

# Restore parameters from the flattened vector
def restore_combination(flattened, shapes):
    miu_shape, gama_shape, omega_shape, theta_shape = shapes

    miu_size = np.prod(miu_shape)
    gama_size = np.prod(gama_shape)
    omega_size = np.prod(omega_shape)
    theta_size = np.prod(theta_shape)

    miu_flat = flattened[:miu_size]
    gama_flat = flattened[miu_size:miu_size + gama_size]
    omega_flat = flattened[miu_size + gama_size:miu_size + gama_size + omega_size]
    theta_flat = flattened[-theta_size:]

    miu = miu_flat.reshape(miu_shape)
    gama = gama_flat.reshape(gama_shape)
    omega = omega_flat.reshape(omega_shape)
    theta = theta_flat[0]

    return miu, gama, omega, theta

# Define the cost function
def cost(parameters, X, y, shapes, Lambda):
    miu, gama, omega, theta = restore_combination(parameters, shapes)
    
    # Forward propagation
    alpha = np.dot(X, miu) 
    B = sigmoid(alpha - gama) 
    beta = np.dot(B, omega)  
    error = y.reshape(-1, 1) - sigmoid(beta - theta) 
    J = np.sum(error ** 2) / (2 * X.shape[0]) + Lambda * (np.sum(omega**2) + theta**2)
    return J

# Define the gradient function
def gradient(parameters, X, y, shapes, Lambda):
    miu, gama, omega, theta = restore_combination(parameters, shapes)
    
    alpha = np.dot(X, miu)  # (17, 2)
    B = sigmoid(alpha - gama)  # (17, 2)
    beta = np.dot(B, omega)  # (17, 1)
    y_pred = sigmoid(beta - theta)  # (17, 1)
    
    error = y_pred - y.reshape(-1, 1)  # Reshape y to (17, 1)
    
    # Compute gradients
    delta_omega = np.dot(B.T, error * y_pred * (1 - y_pred)) / X.shape[0] + 2 * Lambda * omega
    delta_theta = -np.sum(error * y_pred * (1 - y_pred)) / X.shape[0] + 2 * Lambda * theta
    delta_gama = -np.sum(error * y_pred * (1 - y_pred) * omega.T * B * (1 - B), axis=0) / X.shape[0]
    delta_miu = np.dot(X.T, (error * y_pred * (1 - y_pred) * omega.T * B * (1 - B))) / X.shape[0]
    
    # Flatten the gradients
    grad = flatten_combination(delta_miu, delta_gama, delta_omega, delta_theta)
    return grad

# Predict function
def predict(parameters, X, shapes):
    miu, gama, omega, theta = restore_combination(parameters, shapes)
    
    alpha = np.dot(X, miu)
    B = sigmoid(alpha - gama)
    beta = np.dot(B, omega)
    
    y_pred = sigmoid(beta - theta)
    
    return y_pred >= 0.5  # Classification threshold at 0.5

# Training the model
Lambda = 0.01  # Regularization parameter
initial_parameters = flatten_combination(miu, gama, omega, theta)
shapes = (miu.shape, gama.shape, omega.shape, (1,))  # Parameter shapes

# Optimize using scipy minimize
result = opt.minimize(fun=cost, 
                      x0=initial_parameters, 
                      args=(X, y, shapes, Lambda), 
                      jac=gradient, 
                      method='TNC')

trained_parameters = result.x
predictions = predict(trained_parameters, X, shapes)
accuracy = np.mean(predictions == y.reshape(-1, 1)) * 100
print(f"Training accuracy: {accuracy:.2f}%")

# Visualize decision boundary
def plot_decision_boundary(parameters, X, y, shapes):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    Z = predict(parameters, np.c_[xx.ravel(), yy.ravel()], shapes)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap='Paired', alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap='Paired')
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Plot decision boundary
plot_decision_boundary(trained_parameters, X, y, shapes)

