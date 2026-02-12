import cvxpy as cp
import numpy as np
import scipy.io
import time

# Load data
system = 9
c = np.load(f'IEEE{system}/c.npy')
p_min = np.load(f'IEEE{system}/pmin.npy')
p_max = np.load(f'IEEE{system}/pmax.npy')
q_min = np.load(f'IEEE{system}/qmin.npy')
q_max = np.load(f'IEEE{system}/qmax.npy')
p_d = np.load(f'IEEE{system}/pd.npy')
q_d = np.load(f'IEEE{system}/qd.npy')
v_min = np.load(f'IEEE{system}/vmin.npy')
v_max = np.load(f'IEEE{system}/vmax.npy')
G = np.load(f'IEEE{system}/G.npy')
B = np.load(f'IEEE{system}/B.npy')
N = len(p_min)

# Define decision variables
alpha = ...
beta = ...
gammal = ...
gammau = ...
deltal = ...
deltau = ...
phil = ...
phiu = ...

# Define objective
obj = ...

# Define constraints
con = ...

# Set and run solver
start_time = time.time()
prob = cp.Problem(obj, con)
prob.solve(verbose=True)
end_time = time.time()

# Get results
obj_val = prob.value
print('Optimal Value:')
print(obj_val)
print('Overall Time')
print(end_time-start_time)
print('Solvetime')
print(prob.solver_stats.solve_time)
