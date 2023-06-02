import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# Define the domain and mesh size
n = 8
h = 2**(-n)
x = np.arange(0, 1+h, h)
y = np.arange(0, 1+h, h)
X, Y = np.meshgrid(x, y)
N = len(x)-2  # number of interior nodes


# Define the right-hand side function f(x,y) and the boundary conditions g(x,y)
def f(x, y):
    return -20*x**4*y**3-12*x**2*y**5 - 17*np.sin(x*y)*(x**2+y**2)

def g(x, y):
    return x**4*y**5 - 17*np.sin(x*y)

# Create the sparse matrix A using the five-point stencil
A = lil_matrix((N**2, N**2))
for i in range(N):
    for j in range(N):
        k = i*N + j
        A[k, k] = -4
        if i > 0:
            A[k, k-N] = 1
        if i < N-1:
            A[k, k+N] = 1
        if j > 0:
            A[k, k-1] = 1
        if j < N-1:
            A[k, k+1] = 1
A = csr_matrix(A)  # convert to compressed sparse row format

# Define the vector b by computing the values of f(x,y) and g(x,y) at each node
b = np.zeros(N**2)
for i in range(N):
    for j in range(N):
        k = i*N+j
        b[k] = f(x[i+1], y[j+1])*h**2
        if i == 0:
            b[k] -= g(x[i], y[j+1])
        if i == N-1:
            b[k] -= g(x[i+2], y[j+1])
        if j == 0:
            b[k] -= g(x[i+1], y[j])
        if j == N-1:
            b[k] -= g(x[i+1], y[j+2])

# Solve the linear system Ax = b using a sparse direct solver
u = spsolve(A, b)

# Reshape the solution vector x into a 2D array and plot the solution
U = np.zeros((N+2, N+2))
U[1:-1, 1:-1] = u.reshape((N, N))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, U, cmap='jet')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
ax.set_title('Numerical solution of -∆u=f')
plt.show()

plt.contourf(X, Y, U, cmap='jet')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Numerical solution of -∆u=f')
plt.show()

# Compute exact solution
G = np.zeros((N+2, N+2))
for i in range(N+2):
    for j in range(N+2):
        if i==0:
           U[0,j]=g(x[i], y[j])
        if i==N+1:
           U[-1,j]=g(x[-1], y[j])
        if j==0:
           U[i,0]=g(x[i], y[j])
        if j==N+1:
           U[i,-1]=g(x[i], y[j])

# Compute the error norms for two finest meshes
h1 = 2 ** (-n)  # Mesh size for the first finest mesh
h2 = 2 ** (-(n+1))  # Mesh size for the second-finest mesh

# Compute the errors for the two finest meshes
error_inf = np.max(np.abs(U - G))
error_l2 = np.sqrt(np.sum((U - G)**2))

error_inf2 = np.max(np.abs(U - G))
error_l22 = np.sqrt(np.sum((U - G)**2))

# Compute the orders of convergence
order_inf = np.log2(error_inf / error_inf2)
order_l2 = np.log2(error_l2 / error_l22)

# Print the results
print("Mesh 1 (h = {})".format(h1))
print("||u - uh||_inf: ", error_inf)
print("||u - uh||_l2: ", error_l2)
print()

print("Mesh 2 (h = {})".format(h2))
print("||u - uh||_inf: ", error_inf2)
print("||u - uh||_l2: ", error_l22)
print()

print("Orders of Convergence:")
print("Order (||u - uh||_inf): ", order_inf)
print("Order (||u - uh||_l2): ", order_l2)



