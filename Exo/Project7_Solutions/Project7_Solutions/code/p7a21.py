import cvxpy as cp
import numpy as np

# Load data
A = np.load('data/A.npy')
B = np.load('data/B.npy')
d = np.load('data/d.npy')
R = np.load('data/R.npy')
D = np.load('data/D-mat.npy')
k = np.load('data/k.npy')
N = np.load('data/N.npy').item()
Loc = np.load('data/Loc.npy')
x = np.load('xval.npy')
u = np.load('uval.npy')
V = np.load('Vval.npy')

# Find the subset of critical scenarios
CS = []

# Critical scenarios for the first constraint
for i in range(N+1):
    z = cp.Variable(N)
    objective = (R[i,:] - B[i,:] @ V) @ z - B[i,:] @ u - A[i,:] @ x
    constraints = [D @ z <= d, z >= 0]
    prob = cp.Problem(cp.Minimize(-objective), constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)
    if objective.value >= -1e-8:
        CS.append(z.value)

# Critical scenarios for the non-negativity constraints
for i in range(N**2):
    z = cp.Variable(N)
    objective = -u[i] - V[i,:] @ z
    constraints = [D @ z <= d, z >= 0]
    prob = cp.Problem(cp.Minimize(-objective), constraints)
    prob.solve(solver=cp.MOSEK)
    if objective.value >= -1e-8:
        CS.append(z.value)

# Retrieve critical scenarios and keep identical scenarios only once
CS = np.array(CS)
CS = np.unique(CS, axis=0)

# Solve problem (5) with the critical scenarios obtained
x_r = cp.Variable(N+1)
y_r = cp.Variable((N**2, CS.shape[0]))

# Objective function
objective = cp.Minimize(x_r[-1])

# Constraints
constraints = []
for i in range(CS.shape[0]):
    constraints += [A @ x_r + B @ y_r[:,i] >= R @ CS[i,:]]
    constraints += [y_r[:,i] >= 0]
constraints += [0 <= x_r[:N], x_r[:N] <= k]

# Run solver
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.GUROBI, verbose=False)

# Retrieve optimal objective value and decisions
opt_obj = prob.value
x_r_val = x_r.value
y_r_val = y_r.value

# Suboptimality percentage
e_opt = (x[-1] - x_r_val[-1]) / x[-1]
print(e_opt)
