
import numpy as np
from qap_qubo import load_qap_qubo
from qap_amplify import create_qap_model
from hybrid_annealing import HybridAnnealing

if __name__ == "__main__":
    filepath = "QAPLIB/tai20a.dat.txt"
    
    model, constraints = create_qap_model(filepath)
    qubo, const = load_qap_qubo(filepath, alpha=1)
    """
    ha = HybridAnnealing(
        qubo_obj=model,
        qubo_constraint=np.zeros_like(model),
        const_constraint=constraints,
        num_spin=len(model.variables)
    )
    """
    ha = HybridAnnealing(
        qubo_obj=qubo,
        qubo_constraint=np.zeros_like(qubo),
        const_constraint=const,
        num_spin=qubo.shape[0]
    )
    
    final_pool, best_solution = ha.run()
    print("Best Solution:", best_solution)
