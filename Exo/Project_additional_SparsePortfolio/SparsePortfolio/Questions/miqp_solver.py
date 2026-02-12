import cvxpy as cp 


def miqp_solver(Sigma, r, mu, cardinality):

    """
    a function to solve the MQIP formulation 
    Sigma: Covariance matrix
    r: expected returns
    mu: target return
    cardinality: desired cardinality level
    Returns: optimal x, optimal objective value and the runtime
    
    """
    n = Sigma.shape[0]
    M = 2
    x = cp.Variable(n)
    z = cp.Variable(n,  boolean=True)
    

    
    objective = cp.Minimize( 0.5 * cp.quad_form(x, Sigma) ) 
    constraints = [cp.abs(x) <= M*z, cp.sum(x) == 1, r.T@x == mu, cp.sum(z) <= cardinality]
    prob_miqp = cp.Problem(objective, constraints)
    prob_miqp.solve(solver=cp.MOSEK, verbose=False)
    
    return x.value, prob_miqp.value, prob_miqp.solver_stats.solve_time
