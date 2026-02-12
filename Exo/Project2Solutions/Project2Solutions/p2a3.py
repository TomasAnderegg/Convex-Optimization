# %%%%%%%%%%%%%%%%%%%%%% MGT-418 Convex Optimization %%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ------------------- Load data -------------------
mat = loadmat("p2data2.mat")
x = np.asarray(mat['x'], dtype=float)
y = np.asarray(mat['y'], dtype=float).reshape(-1)
m, d = x.shape

# ------------------- Parameters -------------------
rho = 1e-4   # regularization parameter
sigma = 3.0  # bandwidth of Gaussian kernel

# Decision variable
lam = cp.Variable(m)

# Build kernel matrix
K = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        diff = x[i, :] - x[j, :]
        K[i, j] = np.exp(-np.dot(diff, diff) / (2.0 * sigma ** 2))

z = cp.multiply(lam, y)

# Objective:
objective = (
    cp.sum(lam)
    - (m / 2.0) * cp.sum_squares(lam)
    - (1.0 / (2.0 * rho)) * cp.quad_form(z, cp.psd_wrap(K))
)

# Constraints:
constraints = [
    y @ lam == 0,
    lam >= 0,
    lam <= 1.0 / m
]

# Solve the dual (maximize objective)
prob = cp.Problem(cp.Maximize(objective), constraints)
prob.solve(solver=cp.SCS, verbose=False)

if lam.value is None:
    raise RuntimeError("Solver failed to find a solution.")

lambda_opt = lam.value  # numpy array of shape (m,)

b_opt = None
for k in range(m):
    if 0 < lambda_opt[k] < 1.0 / m:
        # z = lambda .* y
        z_vec = lambda_opt * y
        b_opt = (m * lambda_opt[k] - 1.0) * y[k] + (1.0 / rho) * np.dot(
            z_vec, K[:, k]
        )
        break

if b_opt is None:
    raise RuntimeError("No optimal b found. Daniel said this should never happen :)")

# ------------------- Discretization & labels (100 points per feature) -------------------
# Discretize each feature range to 100 discretization points to get 100^d
# total number of discretization points in the feature space
# Construct a feature matrix (denote by feature) of discretization points
# Specifically, feature will be a matrix in R^((100^d) x d), where each row
# represents a distinct feature vector
# Compute the label of each discrete point by using optimal w and b
# Construct a label vector (denoted by label) containing the respective labels
# Specifically, label will be a vector in R^((100^d) x 1)
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()

# 100 points in each direction
grid_x1 = np.linspace(x1_min, x1_max, 100)
grid_x2 = np.linspace(x2_min, x2_max, 100)

feature_list = []
label_list = []

for v1 in grid_x1:
    for v2 in grid_x2:
        feat = np.array([v1, v2])
        # Compute kernel(x_i, feat) for all i
        diffs = x - feat
        sq_norms = np.sum(diffs ** 2, axis=1)
        k_vec = np.exp(-sq_norms / (2.0 * sigma ** 2))

        # w_temp(i) = lambda_i * y_i * K(x_i, feat)
        w_temp = lambda_opt * y * k_vec
        lbl = (1.0 / rho) * np.sum(w_temp) - b_opt

        feature_list.append(feat)
        label_list.append(lbl)

feature = np.array(feature_list)  # shape (m_grid, 2)
label = np.array(label_list)     # shape (m_grid,)

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