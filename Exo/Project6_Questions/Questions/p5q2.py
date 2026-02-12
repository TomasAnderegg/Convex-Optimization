import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# CHOOSE A CONTROL GAIN (start with k = 20, later try to determine the
# smallest control gain for which this method still asserts stability)
k = 14

# System parameters
g = 9.81   # gravitational acceleration [m/s^2]
L = 0.3    # pendulum length [m]
m = 0.5    # pendulum mass [kg]
d = 0.02   # (linearized) frictional damping [Nms]
b = 2.5    # motor conversion constant [Nm]

# Linearized system and control matrices
As = ...
B = ...
K = ...

# Decision variables
P = cp.Variable((2,2), symmetric=True)

# Objective function
obj = cp.Minimize(0)

# Constraints
constr = ...

# Specify solver settings and run solver
prob = cp.Problem(obj, constr)
prob.solve(solver=cp.MOSEK, verbose=False)

# Display whether problem was feasible
print(prob.status)

# Retrieve optimal decisions
if prob.status == cp.OPTIMAL:
    opt_P = P.value

# Plot phase diagram of the dynamic system
plt.figure()
plt.axis('square')
plt.xlabel('$x_1$', fontsize=16)
plt.ylabel('$x_2$', fontsize=16)
plt.axis([-9,9,-9,9])
x1val = np.arange(-7, 8, 1)
x2val = np.arange(-7, 8, 1)
X1, X2 = np.meshgrid(x1val, x2val, indexing='ij')
dX1dt = X2
dX2dt = (g/L - b*k)*X1 - d/(m*L**2)*X2
plt.quiver(X1, X2, dX1dt, dX2dt)

# Plot some level curves of Lyapunov function
if prob.status == cp.OPTIMAL:
    x1val = np.arange(-7, 7.2, 0.2)
    x2val = np.arange(-7, 7.2, 0.2)
    X1, X2 = np.meshgrid(x1val, x2val, indexing='ij')
    Z = opt_P[0,0]*X1**2 + opt_P[0,1]*X1*X2 + opt_P[1,0]*X1*X2 + opt_P[1,1]*X2**2
    plt.contour(X1, X2, Z, levels=np.arange(0, 180, 30), linewidths=2)

# Integrate dynamics for some initial condition and plot trajectory
x0 = [4,0]
tspan = np.linspace(0, 20, 200)

def dynamics(x, t):
    return (As + B @ K) @ x

x = odeint(dynamics, x0, tspan)
plt.plot(x[:,0], x[:,1], 'k', linewidth=2)
plt.plot(x[0,0], x[0,1], 'ko', linewidth=2)

plt.show()
