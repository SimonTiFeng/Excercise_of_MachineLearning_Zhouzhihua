import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('3.0a.csv').to_numpy()
# Data preprocessing
y = data[:, -1]
X = data[:, 1:3]

# Centering the data (optional, depending on the dataset)
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

X_Positives = X_centered[y == 1]
X_Negatives = X_centered[y == 0]
Sw = np.cov(X_Positives.T) + np.cov(X_Negatives.T)

mean_diff = np.mean(X_Positives, axis=0) - np.mean(X_Negatives, axis=0)
Sw_inv = np.linalg.inv(Sw)
Omega = np.dot(Sw_inv, mean_diff)

# Visualizing the decision boundary
def plot_decision_boundary(X, y, Omega):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Class 0', color='red', edgecolor='k')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Class 1', color='blue', edgecolor='k')
    x_values = np.array([np.min(X[:, 0]), np.max(X[:, 0])])
    y_values = - (Omega[0] * x_values) / Omega[1]
    plt.plot(x_values, y_values, label='Decision Boundary', color='green')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('LDA Decision Boundary')
    plt.show()
plot_decision_boundary(X_centered, y, Omega)