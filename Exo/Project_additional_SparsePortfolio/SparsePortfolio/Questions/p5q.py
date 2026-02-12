"""
Convex Optimization Project 5
Mean-Variance Sparse Portfolio Selection
"""


import numpy as np
import cvxpy as cp
import time
from miqp_solver import miqp_solver


# Define the reduced size problem
def solve_reduced_size(Sigma, r, mu, supp):
    """
    Solves the reduced-size problem using the closed-form solution:
    Do NOT solve this by Mosek. Use the closed-form solution you found in Q1-a.
    
    Sigma: Covariance matrix
    r: expected returns
    mu: target return
    supp: a feasible support
    """
    
    
    # +-----------------+
    # | Your Code Here! |
    # +-----------------+
    #FILL
    
    return x_cont, obj_val
        
# Define the SOCP problem 
def solve_socp(Sigma, D, r, mu, k):
    """
    Solves the SOCP problem.
    
    Sigma: Covariance matrix
    D: Diagonal matrix that you need to initialize
    r: expected returns
    mu: target return
    k: cardinality
    """
    
    # +-----------------+
    # | Your Code Here! |
    # +-----------------+
    #FILL
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)
    
    # You can directly get the runtime from prob.solver_stats.solve_time
    return x_socp, prob.value, prob.solver_stats.solve_time

# Define the SDP problem
def solve_sdp(Sigma, r, mu, k):
    """
    Solves the SDP problem.
    
    Sigma: Covariance matrix
    r: expected returns
    mu: target return
    k: cardinality
    """
    
    # +-----------------+
    # | Your Code Here! |
    # +-----------------+
    #FILL
    
    return x_sdp, prob.value, prob.solver_stats.solve_time

    

#%% Data initialization

X = np.load('sp500_1.npy') 
row, col = X.shape
r = np.zeros((col,1))
desired_cardinality = 5

for i in range(col):
    r[i] = sum(X[:,i])/row
mu = np.median(r) 
# Note that Sigma = (X - r)^T (X - r)  is the theoretical formula for covariance,
# in practice numerical rounding can break positive-definiteness.
# Using X^T X / n ensures improves numerical stability. You could also add a small identity instead.
Sigma = X.T @ X / row  

#%% Solve the SDP and SOCP

suitable_D = #FILL
x_socp, val_socp, time_socp = solve_socp(Sigma, suitable_D, r, mu, desired_cardinality) 
support_indices_socp = np.where(np.abs(x_socp) > 1e-8)[0]
print(f"\nLower bound from SOCP: {val_socp:.8f}")
print(f"Time SOCP  {time_socp:.4f}")

x_sdp, val_sdp, time_sdp = solve_sdp(Sigma, r, mu, desired_cardinality)
support_indices_sdp = np.where(np.abs(x_sdp) > 1e-8)[0]
print(f"Lower bound from SDP:  {val_sdp:.8f}")
print(f"Time SDP {time_sdp:.4f}")

start1 = time.time()
x_cont, val_cont = solve_reduced_size(Sigma, r, mu, support_indices_socp)
end1 = time.time()
print(f"\nUpper bound from SOCP support:  {val_cont[0][0]:.8f}")
print(f"Time reduced size  {(end1 - start1):.4f}")
gap_socp = #FILL
print(f"Gap from SOCP {gap_socp:.4f}")

start2 = time.time()
x_cont2, val_cont2 = solve_reduced_size(Sigma, r, mu, support_indices_sdp)
end2 = time.time()
print(f"\nUpper bound from SDP support:  {val_cont2[0][0]:.8f}")
print(f"Time reduced size 2 {(end2 - start2):.4f}")

gap_sdp = #FILL
print(f"Gap from SDP {gap_sdp:.4f}")

#%% MIQP

# x_val, obj_val, runtime = miqp_solver(Sigma, r, mu, desired_cardinality)
# support_indices = np.where(np.abs(x_val) > 1e-8)

# print("Optimal support MIQP ", support_indices[0] )
# print(f"Optimal value from MIQP:  {obj_val:.8f}")
# print(f"Time MIQP { runtime:.4f}")

