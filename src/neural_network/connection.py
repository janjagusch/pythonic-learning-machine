class Connection(object):
    """
    Class represents connection between two nodes in neural network

    Attributes:
        from_node: Origin of the connection
        to_node: Destination of the connection
        weight: Weight of the connection
    """

    def __init__(self, from_node, to_node, weight):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        to_node.input_connections.append(self)

    def __eq__(self, other_connection):
        return self.from_node == other_connection.from_node and self.to_node == other_connection.to_node
