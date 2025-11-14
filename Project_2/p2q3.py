# %%%%%%%%%%%%%%%%%%%%%% MGT-418 Convex Optimization %%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.io import loadmat

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.unicode_minus": False 
    })

# ------------------- Load data -------------------
mat = loadmat("Project_2/p2data1.mat")
x = np.asarray(mat['x'], dtype=float)
y = np.asarray(mat['y'], dtype=float).reshape(-1)
m, d = x.shape

# ------------------- Parameters -------------------
rho = 1e-4   # regularization parameter
sigma = 3.0  # bandwidth of Gaussian kernel

# Dual problem with Gaussian kernel
# Solve the dual problem (4) with the Gaussian kernel 

# Gaussian kernel matrix K_ij = exp(-||xi - xj||^2 / (2σ^2))
K = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        diff = x[i] - x[j]
        K[i, j] = np.exp(-np.dot(diff, diff) / (2 * sigma**2))

# CVXPY variable: dual variables λ
lam = cp.Variable(m)

# Dual objective:
# sum_i (λi - (m/2) λi^2) - (1/(2ρ)) sum_{i,j} λi λj yi yj K_ij
quad_term = cp.sum(cp.multiply(y, lam))  # ∑ λi yi but not enough
obj = cp.sum(lam - (m/2) * cp.square(lam)) \
      - (1/(2*rho)) * cp.quad_form(cp.multiply(lam, y), cp.psd_wrap(K))

# Constraints: ∑ λi yi = 0, 0 ≤ λi ≤ 1/m
constraints = [
    lam @ y == 0,
    lam >= 0,
    lam <= 1/m
]

dual_problem = cp.Problem(cp.Maximize(obj), constraints)
dual_problem.solve(solver=cp.SCS, verbose=False)

lambda_opt = lam.value
# ----------------------------------------------------

# Compute optimal b (denote by b_opt) using the optimal dual solution

# Choose a k with λ_k in (0, 1/m) → support vector
epsilon = 1e-6
sv_indices = np.where((lambda_opt > epsilon) & (lambda_opt < 1/m - epsilon))[0]
k = sv_indices[0]  # pick first valid SV

# Compute term: ∑ λ_i y_i K(x_i, x_k)
sum_term = 0
for i in range(m):
    sum_term += lambda_opt[i] * y[i] * K[i, k]

# b* = y_k (m λ_k – 1) + (1/ρ) ∑ λ_i y_i K(x_i, x_k)
b_opt = y[k] * (m * lambda_opt[k] - 1) + (1/rho) * sum_term

# ------------------- Discretization & labels -------------------

# Build a 100x100 grid in 2D (since d=2)
grid_points = 100
x1_vals = np.linspace(np.min(x[:,0]), np.max(x[:,0]), grid_points)
x2_vals = np.linspace(np.min(x[:,1]), np.max(x[:,1]), grid_points)

feature = np.array([[a, b] for a in x1_vals for b in x2_vals])

# Compute decision value f(z) = (1/ρ) ∑ λ_i y_i K(x_i, z) - b
label = np.zeros(feature.shape[0])

for idx, z in enumerate(feature):
    val = 0
    for i in range(m):
        diff = x[i] - z
        Kiz = np.exp(-np.dot(diff, diff) / (2 * sigma**2))
        val += lambda_opt[i] * y[i] * Kiz
    label[idx] = (1/rho) * val - b_opt

label = label.reshape(-1)

# ------------------- Visualization -------------------
# feel free to comment out and construct your own plots.
m_light_red  = label >= 1
m_dark_red   = (label >= 0) & (label < 1)
m_dark_blue  = (label < 0)  & (label > -1)
m_light_blue = label <= -1
plt.figure(figsize=(7, 6))
ax = plt.gca()
ax.set_facecolor("white")  # improve contrast
 
# plot light regions first (more transparent)
plt.scatter(feature[m_light_blue, 0], feature[m_light_blue, 1],
            s=23, c=[[0, 0.3, 1, 0.25]], marker='.', edgecolors='none',
            zorder=1, rasterized=True)
plt.scatter(feature[m_light_red, 0], feature[m_light_red, 1],
            s=23, c=[[1, 0.3, 0, 0.25]], marker='.', edgecolors='none',
            zorder=2, rasterized=True)
 
# plot dark regions on top (less transparent)
plt.scatter(feature[m_dark_blue, 0], feature[m_dark_blue, 1],
            s=23, c=[[0, 0, 1, 0.6]], marker='.', edgecolors='none',
            zorder=3, rasterized=True)
plt.scatter(feature[m_dark_red, 0], feature[m_dark_red, 1],
            s=23, c=[[1, 0, 0, 0.6]], marker='.', edgecolors='none',
            zorder=4, rasterized=True)
 
# training points on top
train_red  = (y >= 1)
train_blue = ~train_red
plt.scatter(x[train_blue, 0], x[train_blue, 1],
            s=35, c=[[0, 0, 1, 1.0]], marker='o', edgecolors='none',
            zorder=5, label='Class +1')
plt.scatter(x[train_red, 0], x[train_red, 1],
            s=35, c=[[1, 0, 0, 1.0]], marker='o', edgecolors='none',
            zorder=6, label='Class -1')
 
plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel(r'$x_2$', fontsize=14)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Project_2/svm_gaussian_kernel_p2data1.png', dpi=300)
plt.show()