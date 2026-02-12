import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.special import comb
import mask  # Assuming the mask function is in a file named mask.py

# Derivative matrices
D1 = ...
D2 = ...

# Dynamics vector
f1 = ...
f2 = ...

# Decision variables
p = cp.Variable((15,1))
q = cp.Variable((28,1))
P_tilde = cp.Variable((6,6), PSD=True)
Q_tilde = cp.Variable((10,10), PSD=True)

# Constraints initialization
constr = ...

# Quadratic form of V_dot(x)
Q = ...

# Constraints on q and Q to ensure that both express the same polynomial
...

# Constraints on p and q to ensure that V(0) = 0 and V_dot(0) = 0
...

# Constraints that link the decision variables p and P_tilde
...

# Constraints that link the decision variables q and Q_tilde
...

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


