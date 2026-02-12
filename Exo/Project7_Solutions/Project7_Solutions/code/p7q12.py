import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# load data
A = np.load('data/A.npy')
B = np.load('data/B.npy')
d = np.load('data/d.npy')
R = np.load('data/R.npy')
D = np.load('data/D-mat.npy')
k = np.load('data/k.npy')
N = np.load('data/N.npy').item()
Loc = np.load('data/Loc.npy')

# +-------------------+
# |  Your Code Here!  |
# +-------------------+

# Save optimal decisions to mat-file
np.save('xval.npy', x_val)
np.save('uval.npy', u_val)
np.save('Vval.npy', V_val)


# Visualization

# Visualize the locations of the stores and the stock allocations to each store
plt.figure(figsize=(12, 10))
plt.scatter(Loc[0,:], Loc[1,:], s=10 * x_val[0:N]+0.01, color='b')  # the +0.01 is to avoid errors in the scatter plot if some x(i)=0
plt.scatter(Loc[0,:], Loc[1,:], facecolors='none', edgecolors='b')
plt.title('Stock Allocations from Affine Policies', fontsize=18)
plt.show()
