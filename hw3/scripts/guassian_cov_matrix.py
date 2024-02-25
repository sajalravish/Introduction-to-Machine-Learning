import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

##############
# Question 7.1

X1 = np.random.normal(loc=3, scale=3, size=100) # X1 ∼ N(3, 9)
X2 = 0.5*X1 + np.random.normal(loc=4, scale=2, size=100) # X2 ∼ 0.5*X1 + N(4, 4)
sample_points = np.column_stack((X1, X2))

# Compute the mean
sample_mean = np.mean(sample_points, axis=0)
print("Mean of the sample:")
print(sample_mean)
# Results: Mean of the sample = [2.68846045, 5.3888394]

##############
# Question 7.2

sample_covariance_matrix = np.cov(sample_points, rowvar=False)
print("\nCovariance matrix of the sample:")
print(sample_covariance_matrix)
# Results: Covariance matrix of the sample = [[7.42292904 3.00253936]
#                                             [3.00253936 4.78474509]]

##############
# Question 7.3
eigenvalues, eigenvectors = np.linalg.eig(sample_covariance_matrix)
print("\nEigenvalues of the sample:")
print(eigenvalues)
print("\nEigenvectors of the sample:")
print(eigenvectors)
# Results: Eigenvalues of the sample = [9.38335628 2.82431785]
#          Eigenvectors of the sample = [[ 0.83732346 -0.54670781]
#                                        [ 0.54670781  0.83732346]]

##############
# Question 7.4

# Part (i): plot all n=100 data points
plt.figure(figsize=(8, 8))  # Set figure size to make it square
plt.scatter(sample_points[:, 0], sample_points[:, 1], color='purple', label='Data Points')

# Part (ii): plot the eigenvectors as arrows
for i in range(len(eigenvalues)):
    eigenvector = eigenvectors[:, i] # access the ith column (or eigenvector) of the eigenvectors array
    eigenvalue = eigenvalues[i]
    scaled_eigenvector = eigenvector * eigenvalue
    plt.arrow(sample_mean[0], sample_mean[1], scaled_eigenvector[0], scaled_eigenvector[1], 
              head_width=1, head_length=1, fc='red', ec='red', label=f'Eigenvector')

# Plot
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Data Points and Eigenvectors')
plt.legend()
plt.grid(True)
plt.show()

##############
# Question 7.5

# Sort eigenvectors by eigenvalues (descending order)
sort_indices = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, sort_indices]

rotation_matrix = eigenvectors  # rotation matrix U^T
centered_points = sample_points - sample_mean  # center the sample points
rotated_points = np.dot(centered_points, rotation_matrix)  # rotate each point by U^T

# Plot
plt.figure(figsize=(8, 8))
plt.scatter(rotated_points[:, 0], rotated_points[:, 1], color='purple', label='Rotated Points')
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Rotated Sample Points')
plt.grid(True)
plt.show()