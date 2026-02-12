import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np, time

# ------------------------------------
# D-optimal (CVXPY) 
# ------------------------------------
def d_opt_cvxpy(A, solver="MOSEK", verbose=False):
    """
    Solve the D-optimal design in the per-column formulation:

        max log det( sum_i pi_i A_i A_i^T )
        s.t.  pi in Delta_L
    
    Hint: Use cp.log_det for the objective.
    """
    S = [Ai @ Ai.T for Ai in A]

    L = len(S)
    d = S[0].shape[0]
    Pi = cp.Variable(L, nonneg=True)
    
    # Build symbolic information matrix
    V = 0
    for i in range(L):
        V += Pi[i] * cp.Constant(S[i])

    obj = cp.Maximize(cp.log_det(V))
    cons = [cp.sum(Pi) == 1]
    prob = cp.Problem(obj, cons)
    t0 = time.time()
    prob.solve(solver=solver, verbose=verbose)
    t1 = time.time()

    return Pi.value, prob.value, t1 - t0

# --------------------------------------------------------
# G-optimal (CVXPY)
# --------------------------------------------------------
def g_opt_cvxpy(A, solver="MOSEK", verbose=False):
    """
    Solve the SDP reformulation of the G-optimal design problem:

        min s
        s.t.  pi in Delta_L, s in R, s_{i,k} in R
              sum_k s_{i,k} <= s               for all i
              [ s_{i,k}            a_{i,k}^T ]
              [ a_{i,k}   sum_j pi_j A_j A_j^T ]  >= 0  for all i,k

    Rank constraint on sum_j pi_j A_j A_j^T is dropped.
    """
    L = len(A)
    d, K = A[0].shape

    # Variables
    pi = cp.Variable(L, nonneg=True)      # design probabilities
    s = cp.Variable()                     # scalar objective
    s_var = cp.Variable((L, K))           # s_{i,k} variables (free)

    V = 0
    for i in range(L):
        V += pi[i] * (A[i] @ A[i].T)

    constraints = []

    # Simplex constraint for pi
    constraints.append(cp.sum(pi) == 1)

    # Sum_k s_{i,k} <= s   for all i
    for i in range(L):
        constraints.append(cp.sum(s_var[i, :]) <= s)

    # Block PSD constraints:
    # [ s_{i,k}            a_{i,k}^T ]
    # [ a_{i,k}            V        ]  >= 0   for all i,k
    for i in range(L):
        Ai = A[i]  # d x K
        for k in range(K):
            a_ik = Ai[:, k]  # d-dimensional column for (i,k)

            # Make them CVXPY constants/vectors with correct shapes
            a_col = cp.Constant(a_ik.reshape((d, 1)))
            a_row = cp.Constant(a_ik.reshape((1, d)))

            s_ik = s_var[i, k]

            constraints.append(cp.bmat([
                [cp.reshape(s_ik, (1, 1), order='F'), a_row],
                [a_col,                    V    ]
            ]) >> 0)

    # Objective
    obj = cp.Minimize(s)
    prob = cp.Problem(obj, constraints)

    t0 = time.time()
    prob.solve(solver=solver, verbose=verbose)
    t1 = time.time()

    return pi.value, s.value, t1 - t0

# -------------------------------------------------
# Frank–Wolfe for D-opt
# -------------------------------------------------
def d_opt_frank_wolfe(A, max_iter, tol):
    """
    FW for: maximize f(pi) = log det( sum_i pi_i S_i ) over simplex.
    Oracle: i* = argmax_i tr(S_i V^{-1})
    Gap:    g = max_i tr(S_i V^{-1}) - d
    """
    
    S = np.array([Ai @ Ai.T for Ai in A])
    
    L, d, _ = S.shape

    pi = np.ones(L) / L

    for t in range(1,max_iter):
        
        # Vectorized V computation
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

    V = np.tensordot(pi, S, axes=1)
    return pi, np.linalg.slogdet(V)[1]

# ---------------------------
# Benchmarks & comparison

L_list = (100,200,300,400)
d = 5
K = 2
n_repeats = 5
results = []

A_all = np.load("q31.npy")

for L in L_list:
    print(f"Running L={L}...")
    A_gen = A_all[:L]

    # Individual lists instead of metrics dict
    fw_times, fw_supports, fw_logdets = [], [], []
    dcvx_times, dcvx_supports, dcvx_logdets = [], [], []
    gcvx_times, gcvx_supports = [], []
    
    for rep in range(n_repeats):        
        # FW
        t0 = time.time()
        fw_pi, fw_logdet = d_opt_frank_wolfe(A_gen, max_iter=1000, tol=1e-3)
        t1 = time.time()
        fw_times.append(t1 - t0)
        fw_supports.append(int(np.sum(fw_pi > 1e-4)))
        fw_logdets.append(fw_logdet)
        
        # D-opt CVXPY
        dcvx_pi, dcvx_val, dcvx_time = d_opt_cvxpy(A_gen)
        dcvx_times.append(dcvx_time)
        dcvx_supports.append(int(np.sum(dcvx_pi > 1e-4)))
        dcvx_logdets.append(dcvx_val)

        # G-opt CVXPY
        gcvx_pi, gcvx_val, gcvx_time = g_opt_cvxpy(A_gen)
        gcvx_times.append(gcvx_time)
        gcvx_supports.append(int(np.sum(gcvx_pi > 1e-4)))

    # Aggregate
    row = {"L": L}
    
    def add_stats(name, data):
        arr = np.array(data, dtype=float)
        row[f"{name}_mean"] = np.mean(arr)
        row[f"{name}_std"] = np.std(arr)

    add_stats("FW_time", fw_times)
    add_stats("FW_support", fw_supports)
    add_stats("FW_logdet", fw_logdets)
    
    add_stats("Dcvx_time", dcvx_times)
    add_stats("Dcvx_support", dcvx_supports)
    add_stats("Dcvx_logdet", dcvx_logdets)
    
    add_stats("Gcvx_time", gcvx_times)
    add_stats("Gcvx_val", gcvx_val)
    add_stats("Gcvx_support", gcvx_supports)
    
    results.append(row)
    
    print(f"  FW time: {row['FW_time_mean']:.3f} +/- {row['FW_time_std']:.3f} s")
    print(f"  D-cvx time: {row['Dcvx_time_mean']:.3f} +/- {row['Dcvx_time_std']:.3f} s")
    print(f"  G-cvx time: {row['Gcvx_time_mean']:.3f} +/- {row['Gcvx_time_std']:.3f} s")
    print(f"  FW logdet: {row['FW_logdet_mean']:.4f} ")
    print(f"  D-cvx logdet: {row['Dcvx_logdet_mean']:.4f}")
    print(f"  G-cvx logdet: {row['Gcvx_val_mean']:.4f}")
    print()

xs = [r["L"] for r in results]

def plot_with_conf(ax, xs, means, stds, label, color, marker):
    means = np.array(means)
    stds = np.array(stds)
    # Filter out NaNs
    mask = ~np.isnan(means)
    if not np.any(mask): return
    ax.plot(np.array(xs)[mask], means[mask], marker=marker, label=label, color=color, linewidth=2)
    ax.fill_between(np.array(xs)[mask], means[mask]-stds[mask], means[mask]+stds[mask], color=color, alpha=0.2)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

plt.figure(figsize=(8, 6))
plot_with_conf(plt.gca(), xs, [r["FW_time_mean"] for r in results], [r["FW_time_std"] for r in results], "FW (D-opt)", "blue", "o")
plot_with_conf(plt.gca(), xs, [r["Dcvx_time_mean"] for r in results], [r["Dcvx_time_std"] for r in results], "CVXPY D-opt", "orange", "s")
plot_with_conf(plt.gca(), xs, [r["Gcvx_time_mean"] for r in results], [r["Gcvx_time_std"] for r in results], "CVXPY G-opt", "green", "^")

plt.yscale("log")
plt.xlabel(r"$L$")
plt.ylabel("Runtime (s)")
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig("p5q31.png")
plt.show()

print("\nAverage Support Sizes:")
print(f"{'L':<6} | {'FW':<10} | {'D-cvx':<10} | {'G-cvx':<10}")
print("-" * 44)
for r in results:
    fw_s = f"{r['FW_support_mean']:.2f}"
    d_s = f"{r['Dcvx_support_mean']:.2f}" if not np.isnan(r['Dcvx_support_mean']) else "NaN"
    g_s = f"{r['Gcvx_support_mean']:.2f}" if not np.isnan(r['Gcvx_support_mean']) else "NaN"
    print(f"{r['L']:<6} | {fw_s:<10} | {d_s:<10} | {g_s:<10}")