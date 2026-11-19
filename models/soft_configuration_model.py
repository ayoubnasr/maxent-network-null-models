import networkx as nx
import numpy as np

# We have the Karate Club graph as a measured sample from an ensemble we don't know yet. Our goal is to find the generating ensemble by constraining the average number
# of degrees of every node. This requires defining one constraing per node (34 constraints). 

# Define an array of constraints for all the 34 nodes: k[i] = G.degree[i] 

def link_probabilities(lam):
    """Update the link probabilities P_ij to match the given multipliers.
    """
    P = 1 / (1 + np.exp(lam[:, None] + lam[None, :]))
    np.fill_diagonal(P, 0)
    return P

def fit(k, eta=0.15, tol=1e-8, max_iter = 100000):
    lam = np.zeros(len(k)) # Initialize the lagrange multipliers vector

    for i in range(max_iter):
        P = link_probabilities(lam)
        expected = P.sum(axis=1) # Expected node degree == Sum of pij along row i

        error = np.max(np.abs(expected - k))
        if error < tol:
            return lam, P
        lam += eta * (expected - k)

    raise RuntimeError(f"No Convergence, error={error}")

def sample(P, rndg):
    A = np.triu(rndg.random(P.shape) < P, k=1)
    return (A | A.T).astype(int)
