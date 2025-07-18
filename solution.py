import numpy as np
from dataclasses import dataclass, field

@dataclass
class Solution():
    """
    Solution information.

    Attributes:
        x (np.ndarray): n-sized solution composed of binary variables
        energy_all (float): energy value obtained from QUBO-matrix of all term
        energy_obj (float): energy value obtained from QUBO-matrix of objective term
        energy_constraint (float): energy value obtained from QUBO-matrix of constraint term
        constraint (bool): flag whether the solution satisfies the given constraint
    """
    x: np.ndarray
    energy_all: float = 0
    energy_obj: float = 0
    energy_constraint: float = 0
    constraint: bool = True

    @classmethod
    def energy(cls, qubo:np.ndarray, x: np.ndarray, const=0) -> float:
        """
        Calculate the enrgy from the QUBO-matrix & solution x

        Args:
            qubo (np.ndarray): n-by-n QUBO-matrix
            x (np.ndarray): n-sized solution composed of binary variables
            const (int, optional): _description_. Defaults to 0.

        Returns:
            float: Energy value.
        """
        return float(np.dot(np.dot(x, qubo), x) + const)

    @classmethod
    def check_constraint(cls, qubo: np.ndarray, x: np.ndarray, const=0) -> bool:
        """
        Check whether the solution satisfies the constraints.

        Args:
            qubo (np.ndarray): QUBO-model of the constraint term.
            x (np.ndarray): solution that you want to check.
            const (int, optional): constant of the constraint term. Defaults to 0.

        Returns:
            bool: Return True if the solution satisfy.
                  Return False otherwise.
        """
        return True if cls.energy(qubo, x, const) == 0 else False