"""
Convex Optimization Project 5
Mean-Variance Sparse Portfolio Selection
"""


import numpy as np
import cvxpy as cp
import scipy
import time
from miqp_solver import miqp_solver


# Define the reduced size problem
def solve_reduced_size(Sigma, r, mu, supp):
    """
    Solves the reduced size problem using the closed-form solution.
        
    Sigma: Covariance matrix
    r: expected returns
    mu: target return
    supp: a feasible support
    """
    n = Sigma.shape[0]

    # Extract the submatrix Sigma_sub and subvector r_sub corresponding to the free variables
    Sigma_sub = Sigma[np.ix_(supp, supp)]
    r_sub = r[supp]
    L_sub = np.linalg.cholesky(Sigma_sub)
    L_sub_inv = np.linalg.inv(L_sub)
    Sigma_sub_inv = L_sub_inv.T @ L_sub_inv #can be computationally faster for large instances
    
    onew = np.ones((len(supp),1))
    
    a = onew.T @ (Sigma_sub_inv @ onew)
    b = onew.T @ (Sigma_sub_inv @ r_sub)
    c = r_sub.T @ (Sigma_sub_inv @ r_sub)
    
    d1 = (c - b * mu)/(a * c - b**2)
    d2 = -(a * mu - b)/( a * c - b**2)
    x_sub = d1 * Sigma_sub_inv @ onew - d2 * Sigma_sub_inv @ r_sub
    
    x_full = np.zeros((n,1))
    x_full[supp] = x_sub
    obj_val = 0.5 * x_sub.T @ Sigma_sub @ x_sub
    return x_full, obj_val
        

# Define the SDP problem 
def solve_sdp(Sigma, r, mu, k):
    """
    Solves the SDP problem.
    
    Sigma: Covariance matrix
    r: expected returns
    mu: target return
    k: cardinality
    """
    
    n = Sigma.shape[0]
    k = int(k)
    
    
    # Define variables
    y = cp.Variable((1,1))
    d = cp.Variable((n,1))
    z = cp.Variable((n,1))
    c = cp.Variable((1,1))
    lambda_var = cp.Variable((1,1))
    nu = cp.Variable((1,1))
    ddiag = cp.Variable(n)
    D = cp.diag(ddiag)
    sigma = nu * np.ones((col,1)) + lambda_var * r
    theta = -lambda_var * mu - nu
    
    objective = cp.Maximize(k * y + cp.sum(z) + c)
    
    first_row = cp.hstack([Sigma - D, sigma - d])
    second_row = cp.vstack([sigma - d, 2 * (theta - c)])  
    
    constraints = [D >> 1e-8 * np.eye(n), 
                   cp.vstack([first_row, second_row.T]) >> 0,
                   z <= 0]

    for i in range(n):
        constraints.append( cp.bmat([[ -y - z[i],                 cp.reshape(d[i], (1, 1)) ],
                                     [ cp.reshape(d[i], (1, 1)),      cp.reshape(2 * cp.diag(D)[i], (1, 1)) ]] ) >> 0)
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)
    

   
    d_values = d.value 
    D_diag = np.diag(D.value).reshape((n,1))
    # Compute the set { -d_i^2 / D_i } for all i
    d_squared_over_H = ( -np.power(d_values, 2) / D_diag ).flatten()
    
    # Find the indices of the first k smallest elements
    F_k = np.argsort(d_squared_over_H)[:k]  
    x = np.zeros_like(d_values)
    
    # Set x_i = -d_i / D_i for i in F_k, 0 otherwise
    for i in F_k:
        x[i] = -d_values[i] / D_diag[i]
    
    return x, D.value, prob.value, prob.solver_stats.solve_time
    

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
    
    n = Sigma.shape[0]
    k = int(k)
    
    
    # Define variables
    y = cp.Variable((1,1))
    d = cp.Variable((n,1))
    z = cp.Variable((n,1))
    c = cp.Variable((1,1))
    lambda_var = cp.Variable((1,1))
    nu = cp.Variable((1,1)) 
    sigma = nu * np.ones((col,1)) + lambda_var * r
    theta = -lambda_var * mu - nu
    
    invsqrt_Sigma_D = np.linalg.inv(scipy.linalg.sqrtm(Sigma - D))
    objective = cp.Maximize(k * y + cp.sum(z) + c)
    

    constraints = [z <= 0]
    constraints.append(cp.norm(cp.vstack([2 * invsqrt_Sigma_D @ (sigma - d), 1 - 2 * (theta - c)])) <= 1 + 2 * (theta - c))
    
    
    
    for i in range(n):
        constraints.append(cp.norm(cp.vstack( [ cp.reshape(2 * np.sqrt(1 / np.diag(D)[i]) * d[i], (1,1)) , 1 + 2 * (y + z[i]) ] )) <= 1 - 2 * (y +  z[i]))
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)

    prob_val = prob.value

    d_values = d.value 
    D_diag = np.diag(D).reshape((n,1))
    # Compute the set { -d_i^2 / D_i } for all i
    d_squared_over_H = ( -np.power(d_values, 2) / D_diag ).flatten()
    # Find the indices of the first k smallest elements
    F_k = np.argsort(d_squared_over_H)[:k]  
    x = np.zeros_like(d_values)
    
    # Set x_i = -d_i / D_i for i in F_k, 0 otherwise
    for i in F_k:
        x[i] = -d_values[i] / D_diag[i]
    
    
    return x, prob_val, prob.solver_stats.solve_time


#%% Data initialization

X = np.load('sp500_1.npy') #595 days
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


#%%

# Solve the SDP and obtain a lower bound (val_sdp is a lower bound to the problem (1))
x_sdp, D_sdp, val_sdp, time_sdp = solve_sdp(Sigma, r, mu, desired_cardinality)
support_indices_sdp = np.where(np.abs(x_sdp) > 1e-8)[0]
print(f"Optimal value from SDP:  {val_sdp:.8f}")
print(f"Time SDP {time_sdp:.4f}")

# Solve the SOCP and obtain a lower bound (val_socp is a lower bound to the problem (1))
# x_socp, val_socp, time_socp = solve_socp(Sigma, D_sdp, r, mu, desired_cardinality)
x_socp, val_socp, time_socp = solve_socp(Sigma, np.eye(col) * np.min(np.linalg.eigvals(Sigma))/2, r, mu, desired_cardinality)
support_indices_socp = np.where(np.abs(x_socp) > 1e-8)[0]
print(f"\nOptimal value from SOCP: {val_socp:.8f}")
print(f"Time SOCP  {time_socp:.4f}")

# With the feasible supports,
start1 = time.time()
x_cont, val_cont = solve_reduced_size(Sigma, r, mu, support_indices_socp)
end1 = time.time()
print(f"\nUpper bound from SOCP support:  {val_cont[0][0]:.8f}")
print(f"Time reduced size  {(end1 - start1):.4f}")
gap_socp = (val_cont[0][0] - val_socp)/val_cont[0][0]*100
print(f"Gap from SOCP {gap_socp:.4f}")

start2 = time.time()
x_cont2, val_cont2 = solve_reduced_size(Sigma, r, mu, support_indices_sdp)
end2 = time.time()
print(f"\nUpper bound from SDP support:  {val_cont2[0][0]:.8f}")
print(f"Time reduced size2 {(end2 - start2):.4f}")
gap_sdp = (val_cont2[0][0] - val_sdp)/val_cont2[0][0]*100
print(f"Gap from SDP {gap_sdp:.4f}")






#%%

# # """ MIQP """
# x_val, obj_val, runtime = miqp_solver(Sigma, r, mu, desired_cardinality)
# support_indices = np.where(np.abs(x_val) > 1e-8)

# print("Optimal support MIQP ", support_indices[0] )
# print(f"Optimal value from MIQP:  {obj_val:.8f}")
# print(f"Time MIQP { runtime:.4f}")

