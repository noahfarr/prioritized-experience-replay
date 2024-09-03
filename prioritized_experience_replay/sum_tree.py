import numpy as np


class SumTree:

    nodes: np.ndarray
    data: np.ndarray
    size: int
    num_entries: int
    real_size: int

    def __init__(self, size: int):
        self.nodes = np.zeros((2 * size - 1))
        self.data = np.zeros(size, dtype=object)

        self.size = size
        self.num_entries = 0
        self.real_size = 0

    @property
    def total(self):
        return self.nodes[0]

    def add(self, value: float, data: np.ndarray):
        self.data[self.num_entries] = data
        self.update(self.num_entries, value)

        self.num_entries = (self.num_entries + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum: float):
        assert cumsum <= self.total

        node_idx: int = 0
        while 2 * node_idx + 1 < len(self.nodes):
            left = 2 * node_idx + 1
            right = 2 * node_idx + 2

            if cumsum <= self.nodes[left]:
                node_idx = left
            else:
                node_idx = right
                cumsum = cumsum - self.nodes[left]

        data_idx = node_idx - self.size + 1

        return data_idx, self.nodes[node_idx], self.data[data_idx]

    def update(self, data_idx: int, value: float):
        node_idx = data_idx + self.size - 1
        change = value - self.nodes[node_idx]
        self.nodes[node_idx] = value

        parent = (node_idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2
