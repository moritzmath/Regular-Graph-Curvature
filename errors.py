class EdgeError(Exception):
    
    def __init__(self, e,msg=None):
        """Raises an error if the edge is not contained in the graph.

        Args:
            e: Edge
            msg (str, optional): Message to be printed instead of default one. Defaults to None.
        """
        
        if msg is None:
            msg = 'The graph does not contain the edge {}.'.format(e)

        super().__init__(msg)


class RegularityError(Exception):
    
    def __init__(self, edge=False,msg=None):
        """Raises an error if the graph or the edge is not regular.

        Args:
            edge (bool, optional): If true, the edge is not regular. Defaults to False.
            msg (str, optional): Message to be printed instead of default one. Defaults to None
        """
        
        if msg is None:
            if edge:
               msg = 'The endpoints of the edge have different degrees.'
            else:
                msg = 'The graph is not regular.'

        super().__init__(msg)


class GraphError(Exception):
    
    def __init__(self, edge=False,msg=None):
        """Raises an error if the graph is not a simple NetworkX or NetworKit graph.

        Args:
            msg (str, optional): Message to be printed instead of default one. Defaults to None
        """
        
        if msg is None:
            msg = 'The graph format is not supported. Only simple NetworkX and NetworKit graphs are supported.'

        super().__init__(msg)