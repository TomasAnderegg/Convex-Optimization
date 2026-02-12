import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# load data
A = np.load('data/A.npy')
B = np.load('data/B.npy')
d = np.load('data/d.npy')
R = np.load('data/R.npy')
D = np.load('data/D-mat.npy')
k = np.load('data/k.npy')
N = np.load('data/N.npy').item()
Loc = np.load('data/Loc.npy')

# Decision variables
x = cp.Variable(N+1)
u = cp.Variable(N**2)
V = cp.Variable((N**2, N))
Pi = cp.Variable((N+1, N+1))
Lambda = cp.Variable((N**2, N+1))

# Objective function
objective = cp.Minimize(x[-1])

# Constraints
constraints = []
constraints += [A @ x + B @ u - Pi @ d >= 0]
constraints += [B @ V >= R - Pi @ D]
constraints += [u - Lambda @ d >= 0]
constraints += [Lambda @ D + V >= 0]
constraints += [Pi >= 0, Lambda >= 0]
constraints += [0 <= x[:-1], x[:-1] <= 20]

# Specify solver settings and run solver
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.GUROBI, verbose=True)

# Retrieve optimal objective value and decisions
opt_obj = prob.value
x_val = x.value
u_val = u.value
V_val = V.value

# Save optimal decisions to mat-file
np.save('xval.npy', x_val)
np.save('uval.npy', u_val)
np.save('Vval.npy', V_val)


# Visualization

# Visualize the locations of the stores and the stock allocations to each store
plt.figure(figsize=(12, 10))
plt.scatter(Loc[0,:], Loc[1,:], s=10*x_val[0:N]+0.01, color='b')  # the +0.01 is to avoid errors in the scatter plot if some x(i)=0
plt.scatter(Loc[0,:], Loc[1,:], facecolors='none', edgecolors='b')
plt.title('Stock Allocations from Affine Policies', fontsize=18)
plt.show()
