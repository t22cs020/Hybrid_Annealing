
import numpy as np
from dataclasses import dataclass

@dataclass
class Solution:
    x: np.ndarray
    energy_all: float = 0
    energy_obj: float = 0
    energy_constraint: float = 0
    constraint: bool = True

    
    @classmethod
    def energy(cls, qubo: np.ndarray, x: np.ndarray, const=0) -> float:
        return float(np.dot(np.dot(x, qubo), x) + const)

    @classmethod
    def check_constraint(cls, qubo: np.ndarray, x: np.ndarray, const=0) -> bool:
        return cls.energy(qubo, x, const) == 0
