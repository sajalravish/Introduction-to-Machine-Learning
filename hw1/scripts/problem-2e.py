import numpy as np
import matplotlib.pyplot as plt

# Load toy data
toy_data = np.load(f"../data/toy-data.npz")
data, labels = toy_data["training_data"], toy_data["training_labels"]

# Define our variables (given in problem statement)
w1, w2 = -0.4528, -0.5190
w = np.array([w1, w2])
b = 0.1471

plt.scatter(data[:, 0], data[:, 1], c=labels)
# Plot the decision boundary
x = np.linspace(-5, 5, 100)
y = -(w[0] * x + b) / w[1]
plt.plot(x, y, "k")

# Plot the margins
margin1 = (1 - w[0] * x - b) / w[1]
margin2 = (-1 - w[0] * x - b) / w[1]

plt.plot(x, margin1, "k--", label="Pos. Margin")
plt.plot(x, margin2, "k--", label="Neg. Margin")

plt.title('SVM Decision Boundary with Margins')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
