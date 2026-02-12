# %%%%%%%%%%%%%%%%%%%%%% MGT-418 Convex Optimization %%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ------------------- Load data -------------------
# mat = loadmat('p2data1.mat')
mat = loadmat('p2data1.mat') #uncomment to solve for the second data set

x = np.asarray(mat['x'], dtype=float)
y = np.asarray(mat['y'], dtype=float).reshape(-1)
m, d = x.shape

# ------------------- Parameters -------------------
rho = 1e-4  # regularization parameter

# ------------------- Decision variables -------------------
w = cp.Variable((d, 1))
b = cp.Variable()
t = cp.Variable(m)
s = cp.Variable(m)

# ------------------- Objective -------------------
# objective = (1/m) * sum(s) + (rho/2) * w'w
objective = (1.0 / m) * cp.sum(s) + (rho / 2.0) * cp.sum_squares(w)

# ------------------- Constraints -------------------
# For each i:
#   (1/2)*t_i^2 <= s_i
#   (1/2)*t_i^2 + 1 - y_i * (w' x_i - b) - t_i <= s_i
constraints = []
constraints += [0.5 * cp.square(t) <= s]

scores = (x @ w)[:, 0] - b     # shape (m,)
constraints += [0.5 * cp.square(t) + 1 - cp.multiply(y, scores) - t <= s]

# ------------------- Solve -------------------
prob = cp.Problem(cp.Minimize(objective), constraints)
prob.solve(solver=cp.MOSEK, verbose=False)

w_val = w.value.reshape(-1, 1)  # column vector
b_val = float(b.value)
print(f"Objective*: {prob.value:.6f}  ||w||={np.linalg.norm(w_val):.4f}  b={b_val:.4f}")

# ------------------- Discretization & labels (100 points per feature) -------------------
dx1 = (x[:, 0].max() - x[:, 0].min()) / 99
dx2 = (x[:, 1].max() - x[:, 1].min()) / 99

xs1 = np.arange(x[:, 0].min(), x[:, 0].max() + 0.5 * dx1, dx1)
xs2 = np.arange(x[:, 1].min(), x[:, 1].max() + 0.5 * dx2, dx2)

feature = []
label = []
for inp1 in xs1:
    for inp2 in xs2:
        feature.append([inp1, inp2])  
        label.append(float(np.dot(w_val[:, 0], np.array([inp1, inp2])) - b_val))

feature = np.array(feature)          # shape (100^2, 2)
label = np.array(label).reshape(-1)  # shape (100^2,)



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