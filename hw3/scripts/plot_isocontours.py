import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

##############
# Question 6.1
mu1 = np.array([1, 1])
sigma1 = np.array([[1, 0], [0, 2]])

# Grid of x, y points
x = np.linspace(-3, 4, 500)
y = np.linspace(-2, 4, 500)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

rv1 = multivariate_normal(mu1, sigma1)

# Plot
plt.figure(figsize=(8, 6))
plt.contour(x, y, rv1.pdf(pos), cmap='Reds')
plt.colorbar(label='Density')
plt.title('Isocontours of f(µ, Σ) for 6.1')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

##############
# Question 6.2
mu2 = np.array([-1, 2])
sigma2 = np.array([[2, 1], [1, 4]])

# Grid of x, y points
x = np.linspace(-4, 4, 500)
y = np.linspace(-2, 6, 500)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

rv2 = multivariate_normal(mu2, sigma2)

# Plot
plt.figure(figsize=(8, 6))
plt.contour(x, y, rv2.pdf(pos), cmap='Reds')
plt.colorbar(label='Density')
plt.title('Isocontours of f(µ, Σ) for 6.2')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

##############
# Question 6.3
mu1 = np.array([0, 2])
mu2 = np.array([2, 0])
sigma1 = np.array([[2, 1], [1, 1]])

# Grid of x, y points
x = np.linspace(-3, 5, 500)
y = np.linspace(-2, 4, 500)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

rv1 = multivariate_normal(mu1, sigma1)
rv2 = multivariate_normal(mu2, sigma1)
z = rv1.pdf(pos) - rv2.pdf(pos)

# Plot
plt.figure(figsize=(8, 6))
plt.contour(x, y, z, cmap='Reds')
plt.colorbar(label='Density')
plt.title('Isocontours of f(µ, Σ) for 6.3')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

##############
# Question 6.4
mu1 = np.array([0, 2])
mu2 = np.array([2, 0])
sigma1 = np.array([[2, 1], [1, 1]])
sigma2 = np.array([[2, 1], [1, 4]])

# Grid of x, y points
x = np.linspace(-4, 4, 500)
y = np.linspace(-3, 5, 500)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

rv1 = multivariate_normal(mu1, sigma1)
rv2 = multivariate_normal(mu2, sigma2)
z = rv1.pdf(pos) - rv2.pdf(pos)

# Plot
plt.figure(figsize=(8, 6))
plt.contour(x, y, z, cmap='Reds')
plt.colorbar(label='Density')
plt.title('Isocontours of f(µ, Σ) for 6.4')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

##############
# Question 6.5
mu1 = np.array([1, 1])
mu2 = np.array([-1, -1])
sigma1 = np.array([[2, 0], [0, 1]])
sigma2 = np.array([[2, 1], [1, 2]])

# Grid of x, y points
x = np.linspace(-4, 4, 500)
y = np.linspace(-5, 3, 500)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

rv1 = multivariate_normal(mu1, sigma1)
rv2 = multivariate_normal(mu2, sigma2)
z = rv1.pdf(pos) - rv2.pdf(pos)

# Plot
plt.figure(figsize=(8, 6))
plt.contour(x, y, z, cmap='Reds')
plt.colorbar(label='Density')
plt.title('Isocontours of f(µ, Σ) for 6.5')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()