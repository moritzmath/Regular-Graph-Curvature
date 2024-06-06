'''
Implementation to compute the Lin-Lu-Yau curvature and 0-Ollivier-Ricci curvature of a given regular NetworkX or Networkit graph.
'''

import numpy as np
import time
import networkx as nx
import networkit as nk
from util import auction_algorithm, is_regular
from errors import EdgeError, RegularityError, GraphError


def lly_curvature_regular_edge(G, e):
    """Compute Lin-Lu-Yau curvature for a given edge with same endpoint degrees

    Args:
        G (NetworkX or NetworKit graph): A given simple NetworkX or NetworKit graph
        e (Tuple): Edge in the graph

    Raises:
        GraphError: Graph format not supported
        EdgeError: Graph does not contain the edge
        RegularityError: Endpoints have different degrees

    Returns:
        float: Lin-Lu-Yau curvature of the edge
    """
    if isinstance(G, nx.Graph) and not nx.is_directed(G) and not nx.is_weighted(G):
        # Convert nx graph to nk graph
        Gk = nk.nxadapter.nx2nk(G)
        
        # Mapping between nx and nk nodes
        mapping = {}
        for idx, n in enumerate(G.nodes()):
            mapping[n] = idx
        x, y = mapping[e[0]], mapping[e[1]] 

    elif isinstance(G, nk.graph.Graph) and not G.isWeighted() and not G.isDirected():
        Gk = G
        x,y = e[0], e[1]

    else:
        raise GraphError()

    # Check that G contains the edge e
    if not Gk.hasEdge(x,y):
        raise EdgeError(e)
    
    # Check if the endpoints of e have different degrees
    if not Gk.degree(x) == Gk.degree(y):
        raise RegularityError(edge=True)
    
    # Compute the Lin-Lu-Yau curvature 
    d = Gk.degree(x)

    neighbors_x = set(Gk.iterNeighbors(x))
    neighbors_y = set(Gk.iterNeighbors(y))

    triangles = neighbors_x.intersection(neighbors_y)

    if len(triangles) == d-1:
        return (d + 1)/d
    
    else:
        R_x = neighbors_x - triangles
        R_y = neighbors_y - triangles
        R_x.remove(y)
        R_y.remove(x)
        cost_matrix = np.array([[nk.distance.BidirectionalBFS(G=Gk, source=i,target=j).run().getDistance() for j in R_y] for i in R_x])
        _, _, dist_sum = auction_algorithm(C=cost_matrix)

        return (d + 1 -dist_sum)/d


def lly_curvature_regular_graph(G):
    """Compute Lin-Lu-Yau curvature for every edge in a regular graph

    Args:
        G (NetworkX or NetworKit graph): A given simple, regular NetworkX or NetworKit graph

    Raises:
        GraphError: Graph format not supported
        RegularityError: Graph not regular

    Returns:
        dict[(int,int), float]: A dictionary of the Lin-Lu-Yau curvature, e.g., {(node1, node2): Lin-Lu-Yau curvature}.
    """
    if isinstance(G, nx.Graph) and not nx.is_directed(G) and not nx.is_weighted(G):
        # Convert nx graph to nk graph
        Gk = nk.nxadapter.nx2nk(G)
        
        # Mapping between nx and nk nodes
        mapping = {}
        for idx, n in enumerate(G.nodes()):
            mapping[idx] = n

    elif isinstance(G, nk.graph.Graph) and not G.isWeighted() and not G.isDirected():
        Gk = G
        mapping = range(Gk.numberOfNodes())

    else:
        raise GraphError()

    # Check that G is regular
    if not is_regular(Gk):
        raise RegularityError()
    
    # Compute all pairs shortest path
    t = time.time()
    apsp = nk.distance.APSP(Gk)
    apsp.run()
    # print("{} secs to compute all shortest-paths by NetworKit.".format(time.time() - t))

    # Compute the Lin-Lu-Yau curvature
    curvature = {}
    d = Gk.degree(0)
    for x,y in Gk.iterEdges():
        neighbors_x = set(Gk.iterNeighbors(x))
        neighbors_y = set(Gk.iterNeighbors(y))

        triangles = neighbors_x.intersection(neighbors_y)

        if len(triangles) == d-1:
            val = (d + 1)/d
    
        else:
            R_x = neighbors_x - triangles
            R_y = neighbors_y - triangles
            R_x.remove(y)
            R_y.remove(x)
            cost_matrix = np.array([[apsp.getDistance(i,j) for j in R_y] for i in R_x])
            _, _, dist_sum = auction_algorithm(C=cost_matrix)
            val = (d + 1 - dist_sum)/d

        curvature[(mapping[x],mapping[y])] = val
        curvature[(mapping[y],mapping[x])] = val

    return curvature

def zero_ollivier_regular_edge(G, e):
    """Compute the 0-Ollivier-Ricci curvature for a given edge with same endpoint degrees

    Args:
        G (NetworkX or NetworKit graph): A given simple NetworkX or NetworKit graph
        e (Tuple): Edge in the graph

    Raises:
        GraphError: Graph format not supported
        EdgeError: Graph does not contain the edge
        RegularityError: Endpoints have different degrees

    Returns:
        float: 0-Ollivier-Ricci curvature of the edge
    """
    if isinstance(G, nx.Graph) and not nx.is_directed(G) and not nx.is_weighted(G):
        # Convert nx graph to nk graph
        Gk = nk.nxadapter.nx2nk(G)
        
        # Mapping between nx and nk nodes
        mapping = {}
        for idx, n in enumerate(G.nodes()):
            mapping[n] = idx
        x, y = mapping[e[0]], mapping[e[1]] 

    elif isinstance(G, nk.graph.Graph) and not G.isWeighted() and not G.isDirected():
        Gk = G
        x,y = e[0], e[1]

    else:
        raise GraphError()

    # Check that G contains the edge e
    if not Gk.hasEdge(x,y):
        raise EdgeError(e)
    
    # Check if the endpoints of e have different degrees
    if not Gk.degree(x) == Gk.degree(y):
        raise RegularityError(edge=True)
    
    # Calculate the zero-Ollivier-Ricci curvature
    d = Gk.degree(x)

    neighbors_x = set(Gk.iterNeighbors(x))
    neighbors_y = set(Gk.iterNeighbors(y))

    triangles = neighbors_x.intersection(neighbors_y)

    R_x = neighbors_x - triangles
    R_y = neighbors_y - triangles
    cost_matrix = np.array([[nk.distance.BidirectionalBFS(G=Gk, source=i,target=j).run().getDistance() for j in R_y] for i in R_x])
    _, _, dist_sum = auction_algorithm(C=cost_matrix)

    return (d - dist_sum)/d

def zero_ollivier_regular_graph(G):
    """Compute 0-Ollivier-Ricci curvature for every edge in a regular graph

    Args:
        G (NetworkX or NetworKit graph): A given simple, regular NetworkX or NetworKit graph

    Raises:
        GraphError: Graph format not supported
        RegularityError: Graph not regular

    Returns:
        dict[(int,int), float]: A dictionary of the 0-Ollivier-Ricci curvature, e.g., {(node1, node2): 0-Ollivier-Ricci curvature}.
    """
    if isinstance(G, nx.Graph) and not nx.is_directed(G) and not nx.is_weighted(G):
        # Convert nx graph to nk graph
        Gk = nk.nxadapter.nx2nk(G)
        
        # Mapping between nx and nk nodes
        mapping = {}
        for idx, n in enumerate(G.nodes()):
            mapping[idx] = n

    elif isinstance(G, nk.graph.Graph) and not G.isWeighted() and not G.isDirected():
        Gk = G
        mapping = range(Gk.numberOfNodes())

    else:
        raise GraphError()

    # Check that G is regular
    if not is_regular(Gk):
        raise RegularityError()
    
    # Compute all pairs shortest path
    t = time.time()
    apsp = nk.distance.APSP(Gk)
    apsp.run()
    # print("{} secs to compute all shortest-paths by NetworKit.".format(time.time() - t))

    # Compute the zero-Ollivier-Ricci curvature
    curvature = {}
    d = Gk.degree(0)
    for x,y in Gk.iterEdges():
        neighbors_x = set(Gk.iterNeighbors(x))
        neighbors_y = set(Gk.iterNeighbors(y))

        triangles = neighbors_x.intersection(neighbors_y)
        R_x = neighbors_x - triangles
        R_y = neighbors_y - triangles
        cost_matrix = np.array([[apsp.getDistance(i,j) for j in R_y] for i in R_x])
        _, _, dist_sum = auction_algorithm(C=cost_matrix)
        val = (d - dist_sum)/d

        curvature[(mapping[x],mapping[y])] = val
        curvature[(mapping[y],mapping[x])] = val

    return curvature
