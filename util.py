# Reference: 
#       Dimitri P. Bertsekas. "A Distributed Algorithm for the Assignment Problem" (1979)


import numpy as np
import networkit as nk

def auction_algorithm(C, eps=None, compute_total_cost=True):
    """Implementation of the Auction Algorithm by Bertsekas for solving the linear assignment problem. 

    Args:
        C: nxn-Cost matrix 
        eps: Minimum positive increment. Defaults to 1/(n+1).
        compute_total_cost (bool, optional): If true, the total cost of the assignment is computed. Defaults to True.

    Returns:
        An optimal assignment, the number of iterations and the total cost if compute_total_cost is True.
    """

    n = C.shape[0]

    if n == 1:
        return np.array([0]), 0, C[0][0]
    
    eps = 1 / (n + 1) if not eps else eps

    # Initializing
    prices = np.zeros(n)
    assignment = np.zeros(n, dtype=int) - 1

    # Bidding
    rounds = 0
    while (assignment == -1).any():

        rounds += 1

        unassigned = np.where(assignment == -1)[0][0]

        cost = C[unassigned] + prices
        indices = np.argsort(cost)
        
        if (assignment == indices[0]).any():
            assignment[np.where(assignment == indices[0])[0][0]] = -1
        assignment[unassigned] = indices[0]
        
        bidding_increment = cost[indices[1]] - cost[indices[0]] + eps
        prices[indices[0]] += bidding_increment

    total_cost = None
    if compute_total_cost:
        total_cost = C[np.arange(n), assignment].sum()
    
    return assignment, rounds, total_cost


def is_regular(G: nk.graph.Graph):
    """Checks if a given graph is regular

    Args:
        G (nk.graph.Graph): A given NetworKit graph

    Returns:
        bool: True if graph is regular
    """

    degrees = nk.centrality.DegreeCentrality(G).run().scores()
    return len(set(degrees)) <= 1