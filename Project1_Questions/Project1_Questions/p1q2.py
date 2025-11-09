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

# +------------------+
r = cp.Variable((n, 1))
theta = cp.Variable((n, n), symmetric=True, PSD=True)
epsilon = 0.01  # small regularization to ensure PD
A_barre = x = cp.Variable(1)
A_barre = n/2*np.log(np.e * np.pi / 2) - (1/2)*(n-1) - (1/2)*(cp.Maximize(cp.sum(epsilon)) + cp.log_det(-theta + epsilon*np.eye(n)))
objective = cp.Minimize(
    - cp.sum(cp.multiply(mu, cp.diag(theta)))       # linear term
    - cp.sum(cp.multiply(M, theta))                # linear term for off-diagonals
    + rho * cp.norm(theta - cp.diag(cp.diag(theta)), 1)   # L1 penalty off-diag
    - cp.log_det(-theta + epsilon*np.eye(n))      # surrogate convex term
)


constraints = [
    -theta + epsilon*np.eye(n) >> 0,
    epsilon >= r
]

prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.MOSEK)
# theta = theta.value.copy()
prob.solve(solver=cp.MOSEK)

if theta.value is None:
    raise ValueError("Le problème n'a pas été résolu correctement. Vérifiez les contraintes ou le solver.")

prob = cp.Problem(cp.Minimize(objective), constraints)
prob.solve(solver=cp.MOSEK, verbose=True)
theta = theta.value.copy()
# +------------------+

# fix the adjacency matrix
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
