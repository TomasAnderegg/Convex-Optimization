'''
EPFL | MGT-418: Convex Optimization | Tutorial 4, Exercise 3 (Python)
''' 
import cvxpy as cp
import numpy as np

# define problem parameters
v_1 = 1
eta_2 = 1.5
eta_3 = 1.3

# define decision variables
ell = cp.Variable(3)
p = cp.Variable(2)

# define objective function
objective = cp.Minimize(np.array([1/v_1, eta_2/v_1, eta_3/v_1]).T @ ell)


# define all constraints in one shot
constraints = [ell[0] >= cp.norm(cp.vstack([p[0], 1])),
               ell[1] >= cp.norm(cp.vstack([p[1]-p[0], 1])),
               ell[2] >= cp.norm(cp.vstack([4-p[1], 0.5]))]

prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.MOSEK, verbose=0)
#retrieve and display optimal objective value
print('optimal objective value:')
opt_objective = objective.value
print(opt_objective)
#retrieve and display optimal solution values
print('optimal solution values:');
opt_ell = ell.value
print(opt_ell)
opt_p = p.value
print(opt_p)