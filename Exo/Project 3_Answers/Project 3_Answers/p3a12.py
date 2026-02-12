import numpy as np
import cvxpy as cp
from PIL import Image
import matplotlib.pyplot as plt

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
rho = 0.0025

# Decision variable
x = cp.Variable((m, m))

# Compute differences
diff_vert = x[1:] - x[:-1] #x[1:] toutes les lignes sauf la premier, x[:-1] toutes les lignes sauf la derniere
#on part avec une matrice mxm, comme on enleve la premiere et derniere on finit avec m-1xm
diff_hor = x[:, 1:] - x[:, :-1] #on fait la meme choe pour les colonnes.

# Objective function
objective = cp.Minimize(cp.norm(img_noisy - x, 'fro') + rho * (cp.norm(cp.reshape(diff_vert, 256 * 255), 1) + cp.norm(cp.reshape(diff_hor, 256 * 255), 1)))

# Define the problem and solve
prob = cp.Problem(objective)
prob.solve(verbose=True)

# Visualization
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(img_true, cmap='gray')
plt.axis('off')
plt.title('true image')

plt.subplot(132)
plt.imshow(img_noisy, cmap='gray')
plt.axis('off')
plt.title('noisy image')

plt.subplot(133)
plt.imshow(x.value, cmap='gray')
plt.axis('off')
plt.title('denoised image')

plt.show()
