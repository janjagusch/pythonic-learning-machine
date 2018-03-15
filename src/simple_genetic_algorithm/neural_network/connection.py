from copy import deepcopy

class Connection(object):
    """
    Class represents connection between two nodes in neural network.

    Attributes:
        from_node: Origin of the connection
        weight: Weight of the connection

    Notes:
        To node does not need to be stored in object, since connection is directly added to to node connections.
    """

    def __init__(self, from_node, to_node, weight):
        self.from_node = from_node
        self.weight = weight
        # Adds connection to 'to_node'.
        to_node.input_connections.append(self)


