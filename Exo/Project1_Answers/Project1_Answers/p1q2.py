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
plt.show()
