import numpy as np
import cvxpy as cp
import scipy.io as sio

import time
# Load data
system = 118
# system = 9
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

# Declare decision variables
pg = cp.Variable(N)          # active power generation
qg = cp.Variable(N)          # reactive power generation   
E = cp.Variable((N, N), symmetric=True)  # matrix of products between real parts of nodal voltage phasors, needs to be symmetric
F = cp.Variable((N, N), symmetric=True)  # matrix of products between imaginary parts of nodal voltage phasors, needs to be symmetric
H = cp.Variable((N, N))     # matrix of products between real and imaginary parts of nodal voltage phasors, does not need to be symmetric

# Define objective
obj = cp.Minimize(c.T @ pg)

# Define constraints
con = []
for n in range(N):
    # active power balance
    con.append(cp.sum(cp.multiply(G[n], E[n] + F[n]) - cp.multiply(B[n], H[n] - H.T[n])) == pg[n] - p_d[n])
    # reactive power balance
    con.append(-cp.sum(cp.multiply(B[n], E[n] + F[n]) + cp.multiply(G[n], H[n] - H.T[n])) == qg[n] - q_d[n])

# generation limits
con.append(p_min <= pg)
con.append(pg <= p_max)

con.append(q_min <= qg)
con.append(qg <= q_max)

# voltage limits
con.append(v_min**2 <= cp.diag(E) + cp.diag(F))
con.append(cp.diag(E) + cp.diag(F) <= v_max**2)

# positive semi-definite
con.append(cp.bmat([[E, H], [H.T, F]]) >> 0) #bmat = block matrix

# Set and run solver
start_time = time.time()
prob = cp.Problem(obj, con)
prob.solve(verbose=True)
end_time = time.time()

# Retrieve results
obj_val = obj.value

print('Optimal Value:')
print(obj_val)
print('Overall Time')
print(end_time-start_time)
print('Solvetime')
print(prob.solver_stats.solve_time)
