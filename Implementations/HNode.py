class HNode:
    """
    Class which represents node in a hash tree.
    """

    def __init__(self):
        self.children = {}
        self.isLeaf = True
        self.bucket = {}
