import numpy as np
from dataclasses import dataclass, field

@dataclass
class Solution:
    x: np.ndarray
    qubo: np.ndarray = None
    const: float = 0
    N: int = None  # 問題サイズ
    energy_all: float = field(init=False)
    energy_obj: float = field(default=0)
    energy_constraint: float = field(default=0)
    constraint: bool = field(init=False)

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=int).flatten()
        if self.qubo is not None:
            xTx = np.dot(self.x, self.qubo @ self.x)
            self.energy_obj = float(xTx)
            self.energy_constraint = float(self.const)
            self.energy_all = self.energy_obj + self.energy_constraint
        else:
            self.energy_all = float("inf")
            self.energy_obj = float("inf")
            self.energy_constraint = float("inf")
        if self.N is not None:
            self.constraint = self.is_one_hot_assignment(self.x, self.N)
        else:
            self.constraint = None

    @staticmethod
    def is_one_hot_assignment(x: np.ndarray, N: int) -> bool:
        x = np.asarray(x).reshape(N, N)
        return np.all(np.sum(x, axis=1) == 1) and np.all(np.sum(x, axis=0) == 1)