import numpy as np
import cvxpy as cp
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
img_true = Image.open('monalisa.png')

# Convert to double and resize
img_true = np.array(img_true).astype(float)
m = 256
np.array(img_true.resize((m, m, 3)))

# Generate the partial image (denote by img)
# Solve the problem to denoise the image (denote by res)

# +------------------+
# |  Your Code HERE  |
# +------------------+

# Visualization
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(img_true.astype(np.uint8))
plt.axis('off')
plt.title('true image')

plt.subplot(132)
plt.imshow(img.astype(np.uint8))
plt.axis('off')
plt.title('partial image')

plt.subplot(133)
plt.imshow(res.astype(int))
plt.axis('off')
plt.title('reconstructed image')

plt.show()
