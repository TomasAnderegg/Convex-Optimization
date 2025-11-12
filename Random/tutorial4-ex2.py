'''
EPFL | MGT-418: Convex Optimization | Tutorial 4, Exercise 2
'''
import cvxpy as cp
import numpy as np
# define problem parameters
A = np.array([[-1, 0.4, 0.8], [1, 0, 0], [0, 1, 0]])
b = np.array([1, 0, 0.5])
x_des = np.array([7, 2, -6])
N = 10
n = A.shape[0]
# define decision variables (note: time index runs through columns)
x = cp.Variable((n, N+1))
u = cp.Variable(N)
z = cp.Variable(N) 
# initialize constraints
constraints = []
# add initial state constraints
constraints.append(x[:, 0] == np.zeros(n))
# add desired final state constraints
constraints.append(x[:, N] == x_des)
# add system dynamics constraints
for t in range(N):
    constraints.append(x[:, t+1] == A @ x[:,t] + b*u[t])
# add epigraphical variable constraints
for t in range(N):
    constraints.append(-z[t] <= u[t])
    constraints.append(u[t] <= z[t])
    constraints.append(cp.square(u[t]) <= z[t]) ## We need to write the constraint in this form otherwise CVXPY does not recognize the quadratic expression
# define objective function
objective = cp.Minimize(cp.sum(z))
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.MOSEK)

# retrieve and display optimal objective value
print('optimal objective value:');
opt_objective = objective.value;
print(opt_objective);

# retrieve and display optimal solution values
print('optimal solution values:');
opt_x = x.value
print(opt_x);
opt_u = u.value
print(opt_u);
opt_z = z.value
print(opt_z)