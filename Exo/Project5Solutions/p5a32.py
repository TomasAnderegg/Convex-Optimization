import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

with open("q32.pkl", "rb") as f:
    data = pickle.load(f)
    A = data["A"]
    L = data["L"]
    d = data["d"]
    

def d_opt_frank_wolfe(A, max_iter, tol):
    S = np.array([Ai @ Ai.T for Ai in A])
    L, d, _ = S.shape

    pi = np.ones(L) / L

    for t in range(1,max_iter):
        V = np.tensordot(pi, S, axes=1)
        Vinv = np.linalg.inv(V)
        s = np.sum(S * Vinv, axis=(1, 2))
        
        i_star = int(np.argmax(s))
        gap = s[i_star] - d

        if gap <= tol:
            break

        gamma = 2.0 / (t + 2.0) 
        eistar = np.zeros(L); 
        eistar[i_star] = 1.0
        pi = (1 - gamma) * pi + gamma * eistar

    return pi


pi_star = d_opt_frank_wolfe(A, max_iter=2000, tol=1e-3)

S = [Ai @ Ai.T for Ai in A]
V_star = np.tensordot(pi_star, S, axes=1)
V_star = V_star + 1e-6 * np.eye(d)  
s = np.sum(S * np.linalg.inv(V_star) , axis = (1,2)) 

k = 25

top_k_id = np.argsort(-s)[:k]

active_id = np.where(pi_star > 1e-4)[0]
s_active = s[active_id]
print(f"Active s_i -> Mean: {np.mean(s_active):.4f}")

list_vec = np.array([np.mean(Ai, axis=1) for Ai in A])

Z = PCA(n_components=2, random_state=1000).fit_transform(list_vec.astype(float))

plt.figure(figsize=(8, 6))

selected = np.zeros(L, dtype=bool); selected[top_k_id] = True
un_selected = ~(selected)

plt.scatter(Z[un_selected,0], Z[un_selected,1], s=10, alpha=0.1, color='gray', label="Other")
plt.scatter(Z[selected,0], Z[selected,1], s=40, alpha=0.9, color='green', label=f"Top-{k} (Most Informative)")

# Draw 2D Convex Hulls
for points, color in [(Z[selected], 'green')]:
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], color=color, linestyle='--', lw=2)
    plt.fill(points[hull.vertices,0], points[hull.vertices,1], color=color, alpha=0.1)

plt.legend()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("p5q32.png")
plt.tight_layout()
plt.show()

