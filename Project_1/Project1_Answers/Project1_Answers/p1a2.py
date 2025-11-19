import numpy as np
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
from cvxpy import norm, log_det

# load the document matrix
docs = np.load('docs.npy')

# load the words
words = []
with open('words.txt', 'r') as file:
    for line in file:
        words.append(line[:-1])


n, m = docs.shape

# Moments information
mu = np.mean(docs, axis=1)          # mean vector
M = 1 / m * (docs @ docs.T)           # second-order moment

# Regularization parameter
rho = 0.005


# Auxiliary functions
def remove_diag(x):
    return x - cp.diag(cp.diag(x))   # set matrix diagonal to zero


# Function to construct the R(\theta) matrix
def R(x):
    return cp.bmat([[np.array([[0]]), cp.reshape(cp.diag(x), (1, cp.diag(x).shape[0]))], [cp.reshape(cp.diag(x), (cp.diag(x).shape[0], 1)), remove_diag(x)]])


# Construct the r vector
r = 4 / 3 * np.ones(n + 1)
r[0] = 1

# Decision variables
theta = cp.Variable((n, n), symmetric=True)                   # the actual decision variables
v = cp.Variable(n + 1)
theta_bar = theta + remove_diag(theta)  # auxiliary representation of theta

# Construct A_bar (without the constant terms)
A_bar = -0.5 * (v.T @ r + log_det(-R(theta_bar) - cp.diag(v)))

# Construct auxiliary mean and second-order moment terms
Z = M - np.diag(np.diag(M)) + np.diag(mu)

# Objective function
obj = cp.Minimize(A_bar - cp.vec(theta).T @ np.reshape(Z, -1) + rho * norm(cp.vec(remove_diag(theta)), 1))

# Specify solver settings, run solver, and retrieve optimal solution
prob = cp.Problem(obj)
prob.solve(solver=cp.MOSEK, verbose=True)

# fix the adjacency matrix
theta = theta.value
theta[np.abs(theta) <= 1e-4] = 0
theta = theta - np.diag(np.diag(theta))

# plot the Adjacency Matrix
plt.matshow(theta)
plt.title('Adjacency Matrix')
plt.show()


# plot the Graph
G = nx.from_numpy_array(theta)
G = nx.relabel_nodes(G, dict(enumerate(words)))
pos = nx.spring_layout(G, seed=0)
nx.draw(G, pos, with_labels=True, font_size=9, edge_color='gray')

# Depending on your networkx verison, the 'nx.draw' might not work. If that's the case use the below code.
# nx.draw_networkx(G, pos, with_labels=True, font_size=9, edge_color='gray')
plt.show()
