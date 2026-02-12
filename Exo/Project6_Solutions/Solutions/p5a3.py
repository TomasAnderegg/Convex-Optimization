import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.special import comb
import mask  # Assuming the mask function is in a file named mask.py

# Derivative matrices
D1 = np.zeros((15,10))
D2 = np.zeros((15,10))

D1[1, 0] = 1
D2[2, 0] = 1

D1[3, 1] = 2
D2[4, 1] = 1

D1[4, 2] = 1
D2[5, 2] = 2

D1[6, 3] = 3
D2[7, 3] = 1

D1[7, 4] = 2
D2[8, 4] = 2

D1[8, 5] = 1
D2[9, 5] = 3

D1[10, 6] = 4
D2[11, 6] = 1

D1[11, 7] = 3
D2[12, 7] = 2

D1[12, 8] = 2
D2[13, 8] = 3

D1[13, 9] = 1
D2[14, 9] = 4

# Dynamics vector
f1 = np.array([[0, 0, -1, -3/2, 0, 0, -1/2, 0, 0, 0]]).T
f2 = np.array([[0, 3, -1, 0, 0, 0, 0, 0, 0, 0]]).T

# Decision variables
p = cp.Variable((15,1))
q = cp.Variable((28,1))
P_tilde = cp.Variable((6,6), PSD=True)
Q_tilde = cp.Variable((10,10), PSD=True)

# Constraints initialization
constr = []

# Quadratic form of V_dot(x)
Q = D1.T @ p @ f1.T + D2.T @ p @ f2.T
Q = 0.5 * (Q + Q.T)

# Constraints on q and Q to ensure that both express the same polynomial
for k in range(1, 29):
    constr.append(q[k-1] == cp.trace(Q.T @ mask.mask(k, 6)))

# Constraints on p and q to ensure that V(0) = 0 and V_dot(0) = 0
constr.append(p[0] == 0)
constr.append(q[0] == 0)

# Constraints that link the decision variables p and P_tilde
for k in range(1, 16):
    if k == 4 or k == 6:
        constr.append(p[k-1] - 1 == cp.trace(P_tilde.T @ mask.mask(k, 4)))
    else:
        constr.append(p[k-1] == cp.trace(P_tilde.T @ mask.mask(k, 4)))

# Constraints that link the decision variables q and Q_tilde
for k in range(1, 29):
    if k == 4 or k == 6:
        constr.append(-q[k-1] - 1 == cp.trace(Q_tilde.T @ mask.mask(k, 6)))
    else:
        constr.append(-q[k-1] == cp.trace(Q_tilde.T @ mask.mask(k, 6)))

# Define the problem and solve
problem = cp.Problem(cp.Minimize(0), constr)
problem.solve(solver=cp.MOSEK)

# Display whether problem was feasible
print("Problem Status: ", problem.status)

# Retrieve optimal decisions
if problem.status == cp.OPTIMAL:
    opt_p = p.value
    opt_q = q.value
    opt_P_tilde = P_tilde.value
    opt_Q_tilde = Q_tilde.value

# Plot phase diagram of the dynamic system
plt.figure()
plt.axis('square')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis([-9,9,-9,9])

X1, X2 = np.meshgrid(np.arange(-7, 7, 1), np.arange(-7, 7, 1), indexing='ij')
dX1dt = -X2 - 3/2*X1**2 - 1/2*X1**3
dX2dt = 3*X1 - X2
plt.quiver(X1, X2, dX1dt, dX2dt)

# Plot some level curves of Lyapunov function
if problem.status == cp.OPTIMAL:
    x1val = np.arange(-7, 7, 0.2)
    x2val = np.arange(-7, 7, 0.2)
    X1, X2 = np.meshgrid(x1val, x2val, indexing='ij')
    Z = np.zeros(X1.shape)
    for i in range(len(x1val)):
        for j in range(len(x2val)):
            Z[i,j] = opt_p[0] + \
                    opt_p[1]*x1val[i] + opt_p[2]*x2val[j] + \
                    opt_p[3]*x1val[i]**2 + opt_p[4]*x1val[i]*x2val[j] + opt_p[5]*x2val[j]**2 + \
                    opt_p[6]*x1val[i]**3 + opt_p[7]*x1val[i]**2*x2val[j] + opt_p[8]*x1val[i]*x2val[j]**2 + opt_p[9]*x2val[j]**3 + \
                    opt_p[10]*x1val[i]**4 + opt_p[11]*x1val[i]**3*x2val[j] + opt_p[12]*x1val[i]**2*x2val[j]**2 + opt_p[13]*x1val[i]*x2val[j]**3 + opt_p[14]*x2val[j]**4
    plt.contour(X1, X2, Z, levels=np.arange(0, 1501, 100), linewidths=2)

# Integrate dynamics for some initial condition and plot trajectory
x0 = [5,5]
tspan = np.linspace(0, 20, 1000)  # 1000 points between 0 and 20

# dynamics function
def dynamics(x, t):
    return np.array([-x[1] - 3/2*(x[0])**2 - 1/2*(x[0])**3, 3*x[0] - x[1]])

# integrate dynamics
x = odeint(dynamics, x0, tspan)

plt.plot(x[:,0], x[:,1], 'k', linewidth=2)
plt.plot(x[0,0], x[0,1], 'ko', linewidth=2)
plt.show()


