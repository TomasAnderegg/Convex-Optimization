import numpy as np
import cvxpy as cp
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the image
img_true = Image.open('dog.png')
# Convert to double and resize
img_true = np.array(img_true).astype(float)
m = 256
img_true.resize((m, m))

# Add noise
gamma = 20
img_noisy = img_true + gamma * np.random.randn(*img_true.shape)

# Adjust the pixel values such that they are in [0,255]
img_noisy[img_noisy > 255] = 255
img_noisy[img_noisy < 0] = 0

# Regularization weight
rho = cp.Parameter(nonneg=True)

# Decision variable
x = cp.Variable((m, m))

# Compute differences
diff_vert = x[1:] - x[:-1]
diff_hor = x[:, 1:] - x[:, :-1]

# Objective function
objective = cp.Minimize(cp.norm(img_noisy - x, 'fro') + rho * (cp.norm(cp.reshape(diff_vert, 256 * 255), 1) + cp.norm(cp.reshape(diff_hor, 256 * 255), 1)))

# Define the problem and solve for value of rho
prob = cp.Problem(objective)
performances = []
rho_values = np.logspace(-4, -1, 40)
for rho_value in tqdm(rho_values):
    rho.value = rho_value
    prob.solve(solver=cp.MOSEK, verbose=False)
    performances.append(np.linalg.norm(x.value - img_true) / np.linalg.norm(img_noisy - img_true))

plt.plot(rho_values, performances)
plt.xscale('log')
plt.show()

# Visualization
plt.figure(figsize=(15, 5))
for plot, label, rho_value in zip([131, 132, 133], ['weak reg.', 'strong reg.', 'best reg.'], [1e-4, 1e-1, 0.0029]):
    rho.value = rho_value
    prob.solve(solver=cp.MOSEK, verbose=False)
    plt.subplot(plot)
    plt.imshow(x.value, cmap='gray')
    plt.axis('off')
    plt.title(label)
plt.show()
