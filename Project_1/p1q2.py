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
# |  Your Code HERE  |

# helpers
def remove_diag(X):
    return X - cp.diag(cp.diag(X))  # affine

def R(X):
    n = X.shape[0]
    d = cp.diag(X)                  # (n,)
    return cp.bmat([[np.array([[0]]),           cp.reshape(d, (1, n))],
                    [cp.reshape(d, (n, 1)),     remove_diag(X)]])        # (n+1) x (n+1), symmetric

# variables
theta = cp.Variable((n, n), symmetric=True)
v = cp.Variable(n+1)

# constants
r = 4/3 * np.ones(n+1); r[0] = 1
Z = np.diag(mu) + M - np.diag(np.diag(M))  # all-NumPy constant
z = np.reshape(Z, -1)                       # (n*n, 1) constant
theta_bar = theta + theta - cp.diag(cp.diag(theta))           # necessary !     

# objective
#const_term = n/2 * np.log(np.e*np.pi/2) - 0.5*(n+1)          # optional constant
A_bar = -0.5 * (v.T @ r + cp.log_det(-(R(theta_bar) + cp.diag(v))))                     # use cp.log_det
lin_term = cp.vec(theta).T @ z
L1_term = rho * norm(cp.vec(remove_diag(theta)), 1)

objective = A_bar - lin_term + L1_term

prob = cp.Problem(cp.Minimize(objective))
prob.solve(solver=cp.MOSEK, verbose=True)
theta = theta.value


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
nx.draw_networkx(G, pos, with_labels=True, font_size=9, edge_color='gray')
plt.show()
