import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import time

L_list = [100, 200, 300, 400]
d = 5
K = 2
n_repeats = 10 # run independent experiments to get stable runtime comparisons

A_all = np.load("q31.npy")

# ------------------------------------
# D-optimal (CVXPY) 
# ------------------------------------
"""
Solve the D-optimal design in the per-column formulation:

    max log det( sum_i pi_i A_i A_i^T )
    s.t.  pi in Delta_L

Hint: Use cp.log_det for the objective.
"""
# YOUR CODE HERE

# ========================= #
#          UTILS            #
# ========================= #

import matplotlib as mpl
import os
import pickle

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.unicode_minus": False 
    })

def create_plots(problem, results, L):
    os.makedirs(f"Project_5/Q3_1/Problem_{problem}", exist_ok=True)
    pi, runtime, support = results
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(range(L), pi)
    ax.set_xlabel("Index i")
    ax.set_ylabel(r"$\pi_i$")    
    fig.suptitle(f"Solution of ({problem}), L = {L}", fontsize=18, weight='heavy', y=0.98)
    
    ax.text(0.98, 0.05, f"Runtime : {runtime:.2f} s\nSupport  : {support}",
            ha='right', va='bottom', transform=ax.transAxes,
            fontsize=10, bbox=dict(facecolor='white', alpha=1.0),
            multialignment='left')
    
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(f"Project_5/Q3_1/Problem_{problem}/L_{L}.png", dpi=300)

def save_results(results, filename="results_q31.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(results, f)

def load_results(filename="results_q31.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)

def create_histogram(results, problem, L, bins=30, title="Histogram of pi values"):
    pi, runtime, support = results
    plt.figure(figsize=(8, 5))
    plt.hist(pi, bins=bins, edgecolor="black")
    plt.xlabel(r"$\pi_i$")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.savefig(f"Project_5/Q3_1/Histograms/{problem}_{L}.png", dpi=300)

# ========================= #
#          D-OPT            #
# ========================= #

def problem_d(A_all, d, K, n_repeats, L):
    L_max, _, _ = A_all.shape # = (400, 5, 2) which are L, d, K
    results = {}
    pi_values = np.zeros((n_repeats, L))
    runtimes = []
    for r in range(n_repeats):
        start = time.time()
        pi = cp.Variable(L)
        V = 0
        for i in range(L):
            V += pi[i]*(A_all[i]@A_all[i].T)
        objective = cp.Maximize(cp.log_det(V))
        constraints = [cp.sum(pi) == 1, pi >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.MOSEK, verbose=0)
        pi_values[r] = pi.value
        runtimes.append(time.time() - start)
    pi_mean = pi_values[-1]
    support = np.sum(pi_mean > 1e-4)
    runtimes_mean = np.mean(runtimes)
    results = (pi_mean, runtimes_mean, support) # stored in the dict with key L
    return results

# results = load_results()
# results.setdefault("L", {})
# for L in L_list:
#     results["D"][L] = problem_d(A_all, d, K, n_repeats, L)
#     save_results(results)

# results = load_results()
# for L in L_list:
#     create_plots("D", results["D"][L], L)

# results = load_results()
# for L in L_list:
#     create_histogram(results["D"][L], "D", L)

# --------------------------------------------------------
# G-optimal (CVXPY)
# --------------------------------------------------------
"""
Solve the SDP reformulation of the G-optimal design problem:

    min s
    s.t.  pi in Delta_L, s in R, s_{i,k} in R
            sum_k s_{i,k} <= s               for all i
            [ s_{i,k}            a_{i,k}^T ]
            [ a_{i,k}   sum_j pi_j A_j A_j^T ]  >= 0  for all i,k

Rank constraint on sum_j pi_j A_j A_j^T is dropped.
"""
# YOUR CODE HERE
def problem_g(A_all, d, k, n_repeats, L):
    L_max, _, _ = A_all.shape # = (400, 5, 2) which are L, d, K
    pi_values = np.zeros((n_repeats, L))
    runtimes = []
    for r in range(n_repeats):
        start = time.time()
        s = cp.Variable(1)
        S = cp.Variable((L,K))
        pi = cp.Variable(L)
        objective = cp.Minimize(s)
        constraints = [cp.sum(pi) == 1, pi >= 0, cp.sum(S, axis=1) <= s]
        V = 0
        for i in range(L):
            V += pi[i]*(A_all[i]@A_all[i].T)
        for i in range(L):
            for k in range(K):
                a_ik = cp.Constant(A_all[i][:, k].reshape((d,1))) # needs to be a cp constant
                s_ik = cp.reshape(S[i,k], (1,1))
                # Build the block matrix: [s_ik, a_ik^T; a_ik, V]
                top = cp.hstack([s_ik, a_ik.T])
                bottom = cp.hstack([a_ik, V])
                M = cp.vstack([top, bottom])
                constraints.append(M >> 0) # SDP constraint
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.MOSEK, verbose=0)
        pi_values[r] = pi.value
        runtimes.append(time.time() - start)
    pi_mean = pi_values[-1]
    support = np.sum(pi_mean > 1e-4)
    runtimes_mean = np.mean(runtimes)
    results = (pi_mean, runtimes_mean, support) # stored in the dict with key L
    return results

# results = load_results()
# results.setdefault("G", {})
# for L in L_list: # TO DO : run for 200, 300, 400
#     results["G"][L] = problem_g(A_all, d, K, n_repeats, L)
#     save_results(results)

# results = load_results()
# for L in L_list:
#     create_plots("G", results["G"][L], L)

# results = load_results()
# for L in L_list:
#     create_histogram(results["G"][L], "G", L, bins=50)


# -------------------------------------------------
# Frank–Wolfe for D-opt
# -------------------------------------------------
"""
FW (Algorithm 1)
"""
# YOUR CODE HERE

epsilon = 1e-3
T = 1000

def problem_fw(A_all, d, k, n_repeats, epsilon, T, L):
    L_max, _, _ = A_all.shape # = (400, 5, 2) which are L, d, K
    results = {}
    pi_values = np.zeros((n_repeats, L))
    runtimes = []
    for r in range(n_repeats):
        start = time.time()
        pi = np.random.rand(L)
        pi /= np.sum(pi)
        for t in range(T+1):
            V = 0
            for i in range(L):
                # print((A_all[i] @ A_all[i].T).shape)
                V += pi[i]*(A_all[i]@A_all[i].T)
            V += 1e-8 * np.eye(d)
            V_inv = np.linalg.inv(V)
            traces = np.zeros(L) # we want to compare L traces
            for i in range(L):
                traces[i] = np.trace(A_all[i].T @ V_inv @ A_all[i])
            i_star = np.argmax(traces)
            g = traces[i_star] - d
            if g <= epsilon:
                break
            gamma = 2/(t+2)
            e_i_star = np.zeros(L)
            e_i_star[i_star] = 1
            pi = (1- gamma)*pi + gamma*e_i_star
        pi_values[r] = pi           
        runtimes.append(time.time() - start)
    pi_mean = np.mean(pi_values, axis=0)
    support = np.sum(pi_mean > 1e-4)
    runtimes_mean = np.mean(runtimes)
    results = (pi_mean, runtimes_mean, support) # stored in the dict with key L
    return results

# results = load_results()
# results.setdefault("FW", {})
# for L in L_list:
#     results["FW"][L] = problem_fw(A_all, d, K, n_repeats, epsilon, T, L)
#     save_results(results)

# results = load_results()
# for L in L_list:
#     create_plots("FW", results["FW"][L], L)

# results = load_results()
# for L in L_list:
#     create_histogram(results["FW"][L], "FW", L, bins=50)


def compare_algos(results, L_list, save_dir="Project_5/Q3_1/Comparison"):
    os.makedirs(save_dir, exist_ok=True)
    algos = ["D", "G", "FW"]
    colors = {"D": "tab:blue", "G": "tab:orange", "FW": "tab:green"}

    for L in L_list:
        fig, ax = plt.subplots(figsize=(10,6))
        width = 0.25
        x = np.arange(L)

        for idx, algo in enumerate(algos):
            if L in results[algo]:
                pi_mean, runtime, support = results[algo][L]
                ax.bar(x + idx*width, pi_mean, width, label=f"{algo} (runtime={runtime:.2f}s, support={support})", color=colors[algo])
        
        ax.set_xlabel("Index i")
        ax.set_ylabel(r"$\pi_i$")
        ax.set_title(f"Comparison of Algorithms, L = {L}", fontsize=16)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/Comparison_L_{L}.png", dpi=300)

results = load_results()
compare_algos(results, L_list)