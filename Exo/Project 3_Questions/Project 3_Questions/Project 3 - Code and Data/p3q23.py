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
# Solve the problem to denoise the image (denote by res)

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
