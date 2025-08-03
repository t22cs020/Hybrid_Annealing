
from qap_qubo import create_qap_bqm
from hybrid_annealing import HybridAnnealing
from ans import load_qap_answer, get_assignment_from_x, compare_assignments
from amplify import solve, FixstarsClient
from datetime import timedelta


if __name__ == "__main__":
    # 60以上は無理
    pro_filepath = "QAPLIB/tai50a.dat.txt"
    ans_filepath = "QAPLIB/tai50a_ans.dat.txt"
    
    
    client = FixstarsClient()
    client.token = "AE/eTNJSueTP0TJJa9NsloAvTMyxjQIoJ2X"
    client.parameters.timeout = timedelta(milliseconds = 1000)
    
    (bqm, 
     model, 
     qubo_constraints, 
     qubo_obj, 
     flow, dist, 
     const_constraint
    ) = create_qap_bqm(pro_filepath)
    
    print("file success")
    
    ha = HybridAnnealing(
        bqm = bqm,
        qubo_obj = qubo_obj,
        qubo_constraints = qubo_constraints,
        const_constraint = const_constraint, # 定数項
        num_spin = len(model.variables), # QUBOのサイズ (N * N)
        N_I = 20, # 解インスタンスの数
        N_E = 10, # サブQUBO構築数
        N_S = 5,  # 選択する解インスタンスの数（N_S < N_I）
        # サブQUBOサイズは70以上が望ましい
        sub_qubo_size = 50, # サブQUBOのサイズ
        client = client,
        flow = flow, 
        dist = dist
    )
    
    """
    (facility_violation, 
     location_violation, 
     best_solution, 
     ae_total_time, 
     tabu_total_time, 
     loop_count,
     tabu_result
    )= ha.run()
    
    print(f"施設 制約違反数: {facility_violation}")
    print(f"場所 制約違反数: {location_violation}")
    print(f"energy_all = {best_solution.energy_all}")
    print(f"energy_obj = {best_solution.energy_obj}")
    print(f"energy_constraint = {best_solution.energy_constraint}")
    print(best_solution.x.reshape(flow.shape[0], flow.shape[0]))
    """
    
    ans_assignment, ans_energy = load_qap_answer(ans_filepath)
    
    num_measur = 10
    total_sol = 0
    total_loop = 0
    for i in range(num_measur):
        (facility_violation, 
         location_violation, 
         best_solution, 
         ae_total_time, 
         tabu_total_time, 
         loop_count,
         tabu_result
        )= ha.run()
        print(f"古典ソルバの解のコスト： {tabu_result.energy_all}")
        print(f"得られた解のコスト: {best_solution.energy_all}")
        print(f"コスト差分: {best_solution.energy_all - ans_energy}")
        print(f"解の精度（f(X*)/f(X)）: {(ans_energy / best_solution.energy_all):.3f}")
        print(f"解の精度（f(X)/f(X*)）: {(best_solution.energy_all / ans_energy):.3f}")
        print(f"タブーサーチ合計時間: {tabu_total_time:.3f}秒")
        print(f"AE合計時間: {ae_total_time:.3f}秒")
        print(f"ループ回数: {loop_count}")
        print("-----")
        total_sol += best_solution.energy_all
        total_loop += loop_count
        
    print(f"平均解: {total_sol / num_measur}")
    print(f"平均ループ: {total_loop / num_measur}")
    