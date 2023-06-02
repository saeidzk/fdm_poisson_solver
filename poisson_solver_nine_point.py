import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# Define the domain and mesh size
def nine_point_stencil(n):
  h = 2**(-n)
  x = np.arange(0, 1+h, h)
  y = np.arange(0, 1+h, h)
  X, Y = np.meshgrid(x, y)
  N = len(x)-2  # number of interior nodes


  # Define the right-hand side function f(x,y) and the boundary conditions g(x,y)
  def f(x, y):
      return -20 * x**4 * y**3 - 12 * x**2 * y**5 - 17 *np.sin(x*y) * (x**2 + y**2)

  def g(x, y):
      return x**4 * y**5 - 17 * np.sin(x*y)

  # Create the sparse matrix A using the five-point stencil
  gamma = -20                # Main point coefficient
  alfa = 4                  # Right, Left, Up, and Down neighbors coefficient
  betta = 1                 # Corner neighbors coefficient
  A = lil_matrix((N **2, N ** 2))
  for i in range(N):
      for j in range(N):
          k = i * N + j
          #The main point
          A[k, k] = gamma
        
          if j< N - 1:
              # Right neighbor
              A[k, k + 1] = alfa
              # Left neighbor
              A[k + 1, k ] = alfa
            
              if i < N - 1:
                # Up_Left neighbor
                 A[k + 1, k + N] = betta
                 # Down_Right neighbor
                 A[k + N, k + 1] = betta
              
          if i < N - 1:
            
              # Up neighbor
              A[k, k + N] = alfa
              # Down neighbor
              A[k + N, k] = alfa
            
              if j< N - 1:
                # Up_Right neighbor
                  A[k, k + N+1] = betta
                # Down_Left neighbor
                  A[k + N+1, k] = betta
  
  A = csr_matrix(A)  # convert to compressed sparse row format
 

  # Define the vector b by computing the values of f(x,y) and g(x,y) at each node
  
  b = np.zeros(N**2)
  for i in range(N):
      for j in range(N):
          k = i*N+j
          b[k] = - f(x[i+1], y[j+1])*6*h**2
          
          if i == 0:
             if j == 0: 
                b[k] -= alfa * g(x[0], y[1]) + alfa * g(x[1], y[0]) + betta * g(x[0], y[0]) + betta * g(x[0], y[2]) + betta * g(x[2], y[0])
            
             if j == N-1: 
                b[k] -= alfa * g(x[0], y[N]) + alfa * g(x[1], y[N+1]) + betta * g(x[0], y[N+1]) + betta * g(x[0], y[N-1]) + betta * g(x[2], y[N+1])
              
             if j > 0 and j < N-1:
                b[k] -= betta * g(x[0], y[j]) + alfa * g(x[0], y[j+1]) + betta * g(x[0], y[j+2])
               
          
          
          if i == N-1:
              if j == 0: 
                b[k] -= alfa * g(x[N+1], y[1]) + alfa * g(x[N], y[0]) + betta * g(x[N-1], y[0]) + betta * g(x[N+1], y[0]) + betta * g(x[N+1], y[2])
                
              if j == N-1: 
                b[k] -= alfa * g(x[N+1], y[N]) + alfa * g(x[N], y[N+1]) + betta * g(x[N+1], y[N+1]) + betta * g(x[N-1], y[N+1]) + betta * g(x[N+1], y[N-1])
                
              if j > 0 and j < N-1:
                b[k] -= betta * g(x[N+1], y[j]) + alfa * g(x[N+1], y[j+1]) + betta * g(x[N+1], y[j+2])   
                
          
          if i > 0 and i < N-1:
             if j == 0:
                b[k] -= betta * g(x[i], y[0]) + alfa * g(x[i+1], y[0]) + betta * g(x[i+2], y[0])
                
             if j == N-1:
                b[k] -= betta * g(x[i], y[N+1]) + alfa * g(x[i+1], y[N+1]) + betta * g(x[i+2], y[N+1])
           

  # Solve the linear system Ax = b using a sparse direct solver
  u = spsolve(A, b)

  # Reshape the solution vector x into a 2D array and plot the solution
  U = np.zeros((N+2, N+2))
  U[1:-1, 1:-1] = u.reshape((N, N))


  # Set boundary conditions
  for i in range(N+2):
      for j in range(N+2):
        
          if i == 0:
             U[0, j]=g(x[i], y[j])
             
          if i == N+1:
             U[N+1, j]=g(x[i], y[j])
             
          if j == 0:
             U[i, 0]=g(x[i], y[j])
             
          if j == N+1:
             U[i, N+1]=g(x[i], y[j])

  u_exact = np.zeros((N+2, N+2))
  for i in range(N+2):
      for j in range(N+2):
        u_exact [i,j]=  g(x[i], y[j]) 
  

  error_inf_matrix = np.abs(U-u_exact)
  error_inf = np.max(error_inf_matrix)
  error_l2 = np.sqrt(np.sum((U-u_exact)**2))
  
  
  fig = plt.figure(1)
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
  
  plt.contourf(X, Y, error_inf_matrix, cmap='jet')
  plt.colorbar()
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title('Error_inf')
  plt.show()
  

  return error_inf, error_l2
errors_inf = []
errors_l2 = []
i= 0
for n in [7,8]:
  i += i
  result = nine_point_stencil(n)
  errors_inf.append (result [0])
  errors_l2 .append (result [1])
  
odredr_inf = np.log2 (errors_inf[0]/errors_inf[1])
print (odredr_inf)
print (errors_inf)

odredr_l2 = np.log2 (errors_l2[0]/errors_l2[1])
print (odredr_l2)
print (errors_l2)


