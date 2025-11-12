# %%%%%%%%%%%%%%%%%%%%%% MGT-418 Convex Optimization %%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ------------------- Load data -------------------
mat = loadmat(r'C:\Users\tjga9\Documents\Tomas\EPFL\MA3\Convex\Project 2\code\p2data1.mat')
# mat = loadmat('p2data2.mat') #uncomment to solve for the second data set

x = np.asarray(mat['x'], dtype=float)
y = np.asarray(mat['y'], dtype=float).reshape(-1)
m, d = x.shape

# ------------------- Parameters -------------------
rho = 1e-4  # regularization parameter

# Solve SVM problem with smooth Hinge loss to compute the SVM coefficients 
# w and b (denote them by w and b and describe w as a column vector)
# +-----------------+
s = cp.Variable(m)
w = cp.Variable(d)
t = cp.Variable(m)
b = cp.Variable()

# define objective function
objective = cp.Minimize(1/m * cp.sum(s) + rho/2 * cp.norm2(w)**2)

# initialize constraints
constraints = []

# add constraints
# constraints.extend([1/2 * t**2 <= s, 1/2 * t**2 + 1 - y*(cp.transpose(w)*x - b) + t <= s])
# add demand meeting constraints
for j in range(m):
    constraints.append(1/2 * t[j]**2 <= s[j])
    constraints.append(1/2 * t[j]**2 + 1 - y[j]*(cp.matmul(cp.transpose(w), x[j,:]) - b) <= s[j])

prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.MOSEK, verbose=0)

# #retrieve and display optimal objective value
# print('optimal objective value:')
# opt_objective = objective.value
w_val = w.value
b_val = b.value

# +-----------------+

print(f"Objective*: {prob.value:.6f}  ||w||={np.linalg.norm(w_val):.4f}  b={b_val:.4f}")

# ------------------- Discretization & labels (100 points per feature) -------------------
# Discretize each feature range to 100 discretization points to get 100^d
# total number of discretization points in the feature space
# Construct a feature matrix (denote by feature) of discretization points
# Specifically, feature will be a matrix in R^((100^d) x d), where each row
# represents a distinct feature vector
# Compute the label of each discrete point by using optimal w and b
# Construct a label vector (denoted by label) containing the respective labels
# Specifically, label will be a vector in R^((100^d) x 1)

# +-----------------+
'''
    Donc si mon feature x_1 varie entre [-1,1] alors on voudra diviser cet intervalle en 100 points
    et faire pareil pour x_2 ... x_d. Donc si on est en d = 2 on aura une grille de 100x100 points.
'''
x_min, x_max = x.min(axis=0), x.max(axis=0)

grids_1d = [np.linspace(x_min[j], x_max[j], 100) for j in range(d)]

meshes = np.meshgrid(*grids_1d, indexing='ij')
feature = np.column_stack([m.ravel() for m in meshes])  # shape = (100**d, d)

w_val_flat = w_val.reshape(-1)    # s'assurer que w_val est un vecteur 1D
b_val_scalar = float(b_val)       # convertir b_val en scalaire

label = feature @ w_val_flat - b_val_scalar

# +-----------------+

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
            zorder=5)
plt.scatter(x[train_red, 0], x[train_red, 1],
            s=35, c=[[1, 0, 0, 1.0]], marker='o', edgecolors='none',
            zorder=6)
 
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()