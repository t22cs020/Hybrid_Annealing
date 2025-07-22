
import numpy as np
import re
from qap_qubo import create_qap_bqm
from hybrid_annealing import HybridAnnealing
from amplify import solve, FixstarsClient
from datetime import timedelta


if __name__ == "__main__":
    filepath = "QAPLIB/tai20a.dat.txt"
    
    client = FixstarsClient()
    client.token = "AE/eTNJSueTP0TJJa9NsloAvTMyxjQIoJ2X"
    client.parameters.timeout = timedelta(milliseconds = 1000)
    
    bqm, q, model, N, qubo_matrix, qubo_constraints, qubo_obj, name_to_index, flow, dist = create_qap_bqm(filepath)
    
    
    ha = HybridAnnealing(
        bqm = bqm, # QUBO (BQM形式)
        qubo_matrix = qubo_matrix, # QUBO行列
        qubo_obj = qubo_obj,
        qubo_constraints = qubo_constraints,
        const_constraint = np.sum(qubo_constraints), # 定数項
        num_spin = len(model.variables), # QUBOのサイズ (N * N)
        N_I = 20, # 解インスタンスの数
        N_E = 10, # サブQUBO構築数
        N_S = 5,  # 選択する解インスタンスの数（N_S < N_I）
        sub_qubo_size = 50, # サブQUBOのサイズ
        spin = q, # Amplify形式の変数
        client = client,
        name_to_index = name_to_index,
        flow = flow, 
        dist = dist
    )
    
    
    
    facility_violation, location_violation, best_solution = ha.run()
    print(f"施設側制約違反数: {facility_violation}")
    print(f"場所側制約違反数: {location_violation}")
    print(f"energy_all = {best_solution.energy_all}")
    print(f"energy_obj = {best_solution.energy_obj}")
    print(f"energy_constraint = {best_solution.energy_constraint}")
    print(best_solution.x.reshape(flow.shape[0], flow.shape[0]))
    