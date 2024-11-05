'''
Implementation to compute the Lin-Lu-Yau curvature and the Ollivier-Ricci curvature of a regular NetworkX graph.
'''

import multiprocessing as mp
import networkx as nx
import networkit as nk 
import numpy as np
from scipy.optimize import linear_sum_assignment
from errors import RegularityError


# ---Shared global variables for multiprocessing used.---
_Gk = nk.graph.Graph()
_d = 0
_apsp = {}

def calculate_edge_curvature_lly(edge):
    """Lin-Lu-Yau curvature computation for a given edge

    Args:
        edge (int, int): Edge in Networkit graph

    Returns:
        result (int, int, float): Lin-Lu-Yau curvature of the given edge in touple format, e.g. (vertex1, vertex2, Lin-Lu-Yau curvature)
    """
    x, y = edge
    S_1x = set(_Gk.iterNeighbors(x))
    S_1y = set(_Gk.iterNeighbors(y))

    triangles = S_1x.intersection(S_1y)

    if len(triangles) == _d-1:
        return [x,y, (_d+1)/_d]
    else:
        cost_matrix = np.array([[_apsp[l][k] for l in (S_1y - triangles) if not l == x] for k in (S_1x - triangles) if not k == y])
        row_ind, col_ind = linear_sum_assignment(cost_matrix=cost_matrix)
        return (x,y, (_d+1-cost_matrix[row_ind, col_ind].sum())/_d)
    

def calculate_edge_curvature_orc(edge):
    """Ollivier-Ricci curvature computation for a given edge

    Args:
        edge (int, int): Edge in Networkit graph

    Returns:
        result (int, int, float): Ollivier-Ricci curvature of the given edge in touple format, e.g. (vertex1, vertex2, Ollivier-Ricci curvature)
    """
    x, y = edge
    S_1x = set(_Gk.iterNeighbors(x))
    S_1y = set(_Gk.iterNeighbors(y))

    triangles = S_1x.intersection(S_1y)

    cost_matrix = np.array([[_apsp[l][k] for l in (S_1y - triangles)] for k in (S_1x - triangles)])
    row_ind, col_ind = linear_sum_assignment(cost_matrix=cost_matrix)
    return (x,y, (_d-cost_matrix[row_ind, col_ind].sum())/_d)



def lin_lu_yau_curvature(G: nx.Graph):
    """Compute Lin-Lu-Yau curvature of the graph

    Args:
        G (NetworkX Graph): A given regular NetworkX graph.

    Raises:
        RegularityError: Graph not regular

    Returns:
        output dict[(int,int), float]: A dictionary of the Lin-Lu-Yau curvature, e.g., {(vertex1, vertex2): Lin-Lu-Yau curvature}.
    """

    # ---Set global variables for multiprocessing.---
    global _Gk
    global _d
    global _apsp

    output = {}

    # Convert networkx.graph to a networkit.graph
    _Gk = nk.nxadapter.nx2nk(G)
    _d = _Gk.degree(0)

    # Check that graph is regular
    if not all(_Gk.degree(v) == _d for v in _Gk.iterNodes()):
        raise RegularityError()

    # Mapping between nx and nk nodes
    mapping = {}
    for idx, n in enumerate(G.nodes()):
        mapping[idx] = n
    
    # Compute all-pair-shortest-path
    _apsp = np.array(nk.distance.APSP(_Gk).run().getDistances())

    
    # No multiprocessing needed on small graphs
    if _Gk.numberOfEdges() <= 1300:
        for x,y in _Gk.iterEdges():
            output[(mapping[x],mapping[y])] = calculate_edge_curvature_lly((x,y))[2]
    
    # Multiprocessing needed on large graphs
    else:
        args = [(x,y) for x, y in _Gk.iterEdges()]

        with mp.get_context('fork').Pool(mp.cpu_count()) as pool:

            chunksize, extra = divmod(len(args), mp.cpu_count() * 4)
            if extra:
                chunksize += 1

            # Compute Ricci curvature for edges
            result = pool.imap_unordered(calculate_edge_curvature_lly, args, chunksize=chunksize)
            pool.close()
            pool.join()

            for v in result:
                output[(mapping[v[0]],mapping[v[1]])] = v[2]

    return output


def ollivier_ricci_curvature(G: nx.Graph):
    """Compute Ollivier-Ricci curvature of the graph

    Args:
        G (NetworkX Graph): A given regular NetworkX graph.

    Raises:
        RegularityError: Graph not regular

    Returns:
        output dict[(int,int), float]: A dictionary of the Ollivier-Ricci curvature, e.g., {(vertex1, vertex2): Lin-Lu-Yau curvature}.
    """

    # ---Set global variables for multiprocessing.---
    global _Gk
    global _d
    global _apsp

    output = {}

    # Convert networkx.graph to a networkit.graph
    _Gk = nk.nxadapter.nx2nk(G)
    _d = _Gk.degree(0)

     # Check that graph is regular
    if not all(_Gk.degree(v) == _d for v in _Gk.iterNodes()):
        raise RegularityError()

    # Mapping between nx and nk nodes
    mapping = {}
    for idx, n in enumerate(G.nodes()):
        mapping[idx] = n
    
    # Compute all-pair-shortest-path
    _apsp = np.array(nk.distance.APSP(_Gk).run().getDistances())

    
    # No multiprocessing needed on small graphs
    if _Gk.numberOfEdges() <= 1300:
        for x,y in _Gk.iterEdges():
            output[(mapping[x],mapping[y])] = calculate_edge_curvature_orc((x,y))[2]
    
    # Multiprocessing needed on large graphs
    else:
        args = [(x,y) for x, y in _Gk.iterEdges()]

        with mp.get_context('fork').Pool(mp.cpu_count()) as pool:

            chunksize, extra = divmod(len(args), mp.cpu_count() * 4)
            if extra:
                chunksize += 1

            # Compute Ricci curvature for edges
            result = pool.imap_unordered(calculate_edge_curvature_orc, args, chunksize=chunksize)
            pool.close()
            pool.join()

            for v in result:
                output[(mapping[v[0]],mapping[v[1]])] = v[2]

    return output
