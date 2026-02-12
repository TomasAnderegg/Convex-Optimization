import numpy as np
import cvxpy as cp
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
img = Image.open('unknown.png')

# Convert to double and resize
img = np.array(img).astype(float)
m = 256
np.array(img.resize((m, m, 3)))
I_keep = ~np.all(img == 255, axis=2)

# Decision variables
X = [cp.Variable((256, 256)) for i in range(3)]

# Compute differences for each color channel
diff_vert = [x[1:] - x[:-1] for x in X]
diff_hor = [x[:, 1:] - x[:, :-1] for x in X]

# Objective function
objective = cp.Minimize(
    sum([cp.norm(cp.reshape(dvi, 256 * 255), 1)
         + cp.norm(cp.reshape(dhi, 256 * 255), 1)
         for dvi, dhi in zip(diff_vert, diff_hor)])
)

# Constraints
constraints = [
    X[0][i, j] == img[i, j, 0] for i, j in zip(*np.where(I_keep))
] + [
    X[1][i, j] == img[i, j, 1] for i, j in zip(*np.where(I_keep))
] + [
    X[2][i, j] == img[i, j, 2] for i, j in zip(*np.where(I_keep))
] + [
    X[0] >= 0, X[1] >= 0, X[2] >= 0,
    X[0] <= 255, X[1] <= 255, X[2] <= 255
]

# Define the problem and solve
prob = cp.Problem(objective, constraints)
prob.solve(verbose=True)

res = np.stack([x.value for x in X], axis=2)

# Visualization


plt.subplot(121)
plt.imshow(img.astype(np.uint8))
plt.axis('off')
plt.title('partial image')

plt.subplot(122)
plt.imshow(res.astype(int))
plt.axis('off')
plt.title('reconstructed image')

plt.show()
