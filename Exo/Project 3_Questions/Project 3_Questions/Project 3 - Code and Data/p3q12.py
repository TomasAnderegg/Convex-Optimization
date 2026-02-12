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

# +------------------+
# |  Your Code HERE  |
# +------------------+

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
