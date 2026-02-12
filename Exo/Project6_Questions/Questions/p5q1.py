import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# System parameters
g = 9.81   # gravitational acceleration [m/s^2]
L = 0.3    # pendulum length [m]
m = 0.5    # pendulum mass [kg]
d = 0.02   # (linearized) frictional damping [Nms]

# Linearized system matrix
Ah = ...

# Decision variables
P = cp.Variable((2,2), symmetric=True)

# Objective function
obj = cp.Minimize(0)

# Constraints
constraints = ...

# Specify solver settings and run solver
prob = cp.Problem(obj, constraints)
prob.solve(solver=cp.MOSEK, verbose=False)

# Display whether problem was feasible
print("Problem status: ", prob.status)

# Retrieve optimal decisions
if prob.status == cp.OPTIMAL:
    opt_P = P.value

# Plot phase diagram of the dynamic system
plt.figure()
plt.axis('square')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis([-9,9,-9,9])
X1, X2 = np.meshgrid(np.arange(-7, 8), np.arange(-7, 8), indexing='ij')
dX1dt = X2
dX2dt = -g/L*X1 - d/(m*L**2)*X2
plt.quiver(X1, X2, dX1dt, dX2dt)

# Plot some level curves of Lyapunov function
if prob.status == cp.OPTIMAL:
    x1val = np.arange(-7, 7.2, 0.2)
    x2val = np.arange(-7, 7.2, 0.2)
    X1, X2 = np.meshgrid(x1val, x2val, indexing='ij')
    Z = np.zeros(X1.shape)
    for i in range(len(x1val)):
        for j in range(len(x2val)):
            x_vec = np.array([x1val[i], x2val[j]])
            Z[i,j] = x_vec @ opt_P @ x_vec
    plt.contour(X1, X2, Z, np.arange(0, 180, 30), linewidths=2)


# Integrate dynamics for some initial condition and plot trajectory
def dynamics(x, t):
    return np.dot(Ah, x)


# Integrate dynamics for some initial condition and plot trajectory
x0 = np.array([1, 0])
tspan = np.linspace(0, 20, num=1000)  # We need to define the points in time for which to solve the ODE
x = odeint(dynamics, x0, tspan)

print(opt_P)

# Plotting the results
plt.plot(x[:,0], x[:,1], 'k', linewidth=2)
plt.plot(x[0,0], x[0,1], 'ko', linewidth=2)
plt.show()
