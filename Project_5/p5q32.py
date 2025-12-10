import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

with open("q32.pkl", "rb") as f:
    data = pickle.load(f)
    A = data["A"] # shape 1000 x 382 x 2 = L x d x K
    L = data["L"]
    d = data["d"]

S = [Ai @ Ai.T for Ai in A]

# ------------------------------------
# Frank–Wolfe for D-opt
# ------------------------------------
# YOUR CODE HERE

# Part 1 : compute pi_star
def problem_fw(d, epsilon, T, L):
    pi = np.random.rand(L)
    pi /= np.sum(pi)
    for t in range(T+1):
        V = 0
        for i in range(L):
            V += pi[i]*S[i]
        V_inv = np.linalg.inv(V)
        traces = np.zeros(L) # we want to compare L traces
        for i in range(L):
            traces[i] = np.trace(S[i] @ V_inv)
        i_star = np.argmax(traces)
        g = traces[i_star] - d
        if g <= epsilon:
            break
        gamma = 2/(t+2)
        e_i_star = np.zeros(L)
        e_i_star[i_star] = 1
        pi = (1- gamma)*pi + gamma*e_i_star
    return pi

epsilon = 1e-3
T = 1000

pi_star = problem_fw(d, epsilon, T, L)
np.save('pi_star.npy', pi_star)

# Part 2 : compute s_i and check that all prompts i s.t. pi_star[i] > 1e-4 have s_i approx. equal to d
# pi_star = np.load("pi_star.npy")
V_star = sum(pi_star[i] * S[i] for i in range(L))
s = np.array([np.trace(S[i] @ np.linalg.inv(V_star)) for i in range(L)])
print(f"support size: {np.sum(pi_star > 1e-4)} / {L}")

# Part 3 : selection of the top k prompts for PCA
k=25
top_k_ids = np.argsort(-s)[:k] # argsort sorts in increasing order, we want the sorting to be in decreasing order to take the k first indices corresponding to the k largest elements of s


# --- Verify Optimality Condition ---
active_ids = np.where(pi_star > 1e-4)[0]
s_active = s[active_ids]
print(f"Active s_i -> Mean: {np.mean(s_active):.4f}, compared to d: {d}")
print(f"Active s_i -> Std: {np.std(s_active):.4f}")

list_vec = np.array([np.mean(Ai, axis=1) for Ai in A])

Z = PCA(n_components=2).fit_transform(list_vec)

plt.figure(figsize=(8, 6))

sel_top = np.zeros(L, dtype=bool); sel_top[top_k_ids] = True
sel_other = ~(sel_top)

plt.scatter(Z[sel_other,0], Z[sel_other,1], s=10, alpha=0.1, color='gray', label="Other")
plt.scatter(Z[sel_top,0], Z[sel_top,1], s=40, alpha=0.9, color='green', label=f"Top-{k} (Most Informative)")

# Draw 2D Convex Hulls
for points, color in [(Z[sel_top], 'green')]:
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], color=color, linestyle='--', lw=2)
    plt.fill(points[hull.vertices,0], points[hull.vertices,1], color=color, alpha=0.1)

plt.legend()
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.tight_layout()
plt.show()

plt.savefig("p5q32.png")