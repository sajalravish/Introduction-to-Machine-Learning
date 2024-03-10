import numpy as np
import matplotlib.pyplot as plt

w1_range = np.linspace(-2, 2, 500)
w2_range = np.linspace(-2, 2, 500)
W1, W2 = np.meshgrid(w1_range, w2_range)
W = np.dstack((W1, W2))

# Formula to compute the l-norm
def lnorm(w, p):
    return pow(np.sum(np.abs(w)**p, axis=-1), 1/p)

norm_p05 = lnorm(W, 0.5)
norm_p1 = lnorm(W, 1)
norm_p2 = lnorm(W, 2)

plt.figure(figsize=(12, 4))

########################
# Question 5.1: part (a)
plt.subplot(131)
plt.contour(W1, W2, norm_p05, levels=20)
plt.xlabel('w1')
plt.ylabel('w2')
plt.title('Isocontours for ℓ0.5-norm')

########################
# Question 5.1: part (b)
plt.subplot(132)
plt.contour(W1, W2, norm_p1, levels=20)
plt.xlabel('w1')
plt.ylabel('w2')
plt.title('Isocontours for ℓ1-norm')

########################
# Question 5.1: part (c)
plt.subplot(133)
plt.contour(W1, W2, norm_p2, levels=20)
plt.xlabel('w1')
plt.ylabel('w2')
plt.title('Isocontours for ℓ2-norm')

plt.tight_layout()
plt.show()