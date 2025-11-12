'''
EPFL | MGT-418: Convex Optimization | Tutorial 4, Exercise 1 (Python)
'''
import cvxpy as cp
import numpy as np
# define problem parameters
d = [3475, 1223, 2260, 2700, 2950]
m = [0, 0.47, 0.5, 0.4, 0.3]
s = [2000, 1600, 1000, 990, 2800]
t = [0.85, 0.9, 0.85, 0.85, 0.9]
# define decision variables
x = cp.Variable(5)
y = cp.Variable((5,5))
# define objective function
objective = cp.Minimize(cp.sum(x))

# initialize constraints
constraints = []

# add nonnegativity constraints
constraints.extend([x >= 0, y >=0])

# add demand meeting constraints
for j in range(5):
    constraints.append(x[j] + t[j] * cp.sum(y[:, j]) >= d[j])
# add minimum freshwood amount constraints
for j in range(5):
    constraints.append((1-m[j])*x[j] - m[j]*t[j]*sum(y[:,j]) >= 0)
# add old paper availability constraints
for i in range(5):
    constraints.append(sum(y[i,:]) <= s[i])
# add recycling compatibility constraints
constraints.extend([y[0,2]==0, y[0,3]==0, y[0,4]==0,
                            y[1,4]==0,
                            y[2,0]==0,
                            y[3,0]==0, y[3,3]==0,
                            y[4,0]==0, y[4,1]==0, y[4,2]==0, y[4,3]==0])
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.MOSEK, verbose=0)
#retrieve and display optimal objective value
print('optimal objective value:')
opt_objective = objective.value
print(opt_objective)
#retrieve and display optimal solution values
print('optimal solution values:')
opt_x = x.value
print(opt_x)
opt_y = y.value
print(opt_y)