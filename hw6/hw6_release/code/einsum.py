import numpy as np

# Problem 6.1
# Part 1: For a random matrix A ∈ (5×5), find its trace.
A = np.random.randn(5, 5)
trace_np = np.trace(A)
trace_einsum = np.einsum('i...i', A)
print("Norms of the differences:")
print("1.", np.linalg.norm(trace_np - trace_einsum))

# Part 2: For random matrices A, B ∈ (5×5), compute their matrix product.
B = np.random.randn(5, 5)
product_np = np.matmul(A, B)
product_einsum = np.einsum('ij,jk->ik', A, B)
print("2.", np.linalg.norm(product_np - product_einsum))

# Part 3: For a batch of random matrices of shapes (3, 4, 5) and (3, 5, 6) (the batch size is 3 here), 
#         compute their batchwise matrix product (the resulting batch will have shape (3, 4, 6)).
A_batch = np.random.randn(3, 4, 5)
B_batch = np.random.randn(3, 5, 6)
product_batch_np = np.matmul(A_batch, B_batch)
product_batch_einsum = np.einsum('ijk,ikl->ijl', A_batch, B_batch)
print("3.", np.linalg.norm(product_batch_np - product_batch_einsum))
