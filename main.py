

from qap_qubo import create_qap_bqm
from hybrid_annealing import HybridAnnealing
from ans import load_qap_answer, get_assignment_from_x, compare_assignments
from amplify import solve, FixstarsClient
from datetime import timedelta


if __name__ == "__main__":
    pro_filepath = "QAPLIB/tai20a.dat.txt"
    ans_filepath = "QAPLIB/tai20a_ans.dat.txt"
    
    
    client = FixstarsClient()
    client.token = "AE/eTNJSueTP0TJJa9NsloAvTMyxjQIoJ2X"
    client.parameters.timeout = timedelta(milliseconds = 1000)
    
    bqm, model, qubo_constraints, qubo_obj, flow, dist, const_constraint = create_qap_bqm(pro_filepath)
    
    ha = HybridAnnealing(
        bqm = bqm,
        qubo_obj = qubo_obj,
        qubo_constraints = qubo_constraints,
        const_constraint = const_constraint, # 定数項
        num_spin = len(model.variables), # QUBOのサイズ (N * N)
        N_I = 40, # 解インスタンスの数
        N_E = 20, # サブQUBO構築数
        N_S = 10,  # 選択する解インスタンスの数（N_S < N_I）
        sub_qubo_size = 100, # サブQUBOのサイズ
        client = client,
        flow = flow, 
        dist = dist
    )
    
    facility_violation, location_violation, best_solution = ha.run()
    
    
    print(f"施設 制約違反数: {facility_violation}")
    print(f"場所 制約違反数: {location_violation}")
    print(f"energy_all = {best_solution.energy_all}")
    print(f"energy_obj = {best_solution.energy_obj}")
    print(f"energy_constraint = {best_solution.energy_constraint}")
    print(best_solution.x.reshape(flow.shape[0], flow.shape[0]))
    
    # --- QAP最適解ファイルの読み込み ---
    ans_assignment, ans_energy = load_qap_answer(ans_filepath)
    N = flow.shape[0]
    my_assignment = get_assignment_from_x(best_solution.x, N)

    matches = compare_assignments(ans_assignment, my_assignment)
    print(f"一致施設数: {matches}/{N}")

    # 目的関数値（コスト）の比較
    print(f"最適値: {ans_energy}")
    print(f"得られた解のコスト: {best_solution.energy_all}")
    print(f"コスト差分: {best_solution.energy_all - ans_energy}")
    print(f"解の精度: {ans_energy / best_solution.energy_all}")
    