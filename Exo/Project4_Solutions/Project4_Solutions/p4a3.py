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
alpha = cp.Variable(N)
beta = cp.Variable(N)
gammal = cp.Variable(N)
gammau = cp.Variable(N)
deltal = cp.Variable(N)
deltau = cp.Variable(N)
phil = cp.Variable(N)
phiu = cp.Variable(N)

# Define objective
obj = cp.Maximize(p_d @ alpha + q_d @ beta + p_min @ gammal 
                        - p_max @ gammau + q_min @ deltal 
                        - q_max @ deltau + v_min**2 @ phil 
                        - v_max**2 @ phiu)

# Define constraints
con = []

# Equality constraints
con.append(c - alpha - gammal + gammau == 0)
con.append(beta + deltal - deltau == 0)

# Positive semi-definiteness constraint
X = cp.bmat([
    [cp.diag(alpha) @ G + G.T @ cp.diag(alpha), -cp.diag(alpha) @ B + B.T @ cp.diag(alpha)],
    [cp.diag(alpha) @ B - B.T @ cp.diag(alpha), cp.diag(alpha) @ G + G.T @ cp.diag(alpha)]
]) + cp.bmat([
    [-cp.diag(beta) @ B - B.T @ cp.diag(beta), -cp.diag(beta) @ G + G.T @ cp.diag(beta)],
    [cp.diag(beta) @ G - G.T @ cp.diag(beta), -cp.diag(beta) @ B - B.T @ cp.diag(beta)]
]) + 2* cp.bmat([
    [cp.diag(phiu - phil), np.zeros((N,N))],
    [np.zeros((N,N)), cp.diag(phiu - phil)]
])
con.append(X >> 0)

# Non-negativity constraints
con.append(gammal >= 0)
con.append(gammau >= 0)
con.append(deltal >= 0)
con.append(deltau >= 0)
con.append(phil >= 0)
con.append(phiu >= 0)

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
