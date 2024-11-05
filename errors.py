class RegularityError(Exception):
    
    def __init__(self, msg=None):
        """Raises an error if the graph is not regular.

        Args:
            msg (str, optional): Message to be printed instead of default one. Defaults to None
        """
        
        if msg is None:
            msg = 'The graph is not regular.'

        super().__init__(msg)
