
import numpy as np
import re
from qap_qubo import create_qap_bqm
from hybrid_annealing import HybridAnnealing
from amplify import solve, FixstarsClient
from dwave.samplers import TabuSampler
from datetime import timedelta


if __name__ == "__main__":
    filepath = "QAPLIB/tai20a.dat.txt"
    
    client = FixstarsClient()
    client.token = "AE/eTNJSueTP0TJJa9NsloAvTMyxjQIoJ2X"
    client.parameters.timeout = timedelta(milliseconds = 100)
    
    bqm, q, model, N, constraints, qubo_matrix, name_to_index = create_qap_bqm(filepath)
    
    # bqm {('{name: q_{i,j}, id: 398, type: Binary}', )}
    
    #デバック用
    #print("bqm", bqm)
    #print("constaraints", constraints)
    """
    sampler = TabuSampler()
    sampleset = sampler.sample(bqm, num_reads=100)
    best_sample = sampleset.first.sample
    best_energy = sampleset.first.energy
    #print(sampleset)

    
    assignment_i = {}  # 施設ごと
    assignment_p = {}  # 場所ごと

    # best_sampleから割り当て抽出
    for v, val in best_sample.items():
        if val == 1:
            # q_{i,p}の抽出（正規表現でi,pを抜き出す）
            m = re.search(r'q_\{(\d+),(\d+)\}', v)
            if m:
                i, p = map(int, m.groups())
                # 施設ごと
                assignment_i.setdefault(i, []).append(p)
                # 場所ごと
                assignment_p.setdefault(p, []).append(i)
            else:
                print(f"変数名抽出失敗: {v}")

    # 判定へ（ここから下はあなたのロジックをそのまま利用）
    row_ok = all(len(assignment_i.get(i, [])) == 1 for i in range(N))
    col_ok = all(len(assignment_p.get(p, [])) == 1 for p in range(N))

    if row_ok and col_ok:
        print("one-hot制約を全て満たしています")
    else:
        for i in range(N):
            if len(assignment_i.get(i, [])) != 1:
                print(f"施設{i}が複数の場所に割り当て or 未割当: {assignment_i.get(i,[])}")
        for p in range(N):
            if len(assignment_p.get(p, [])) != 1:
                print(f"場所{p}に複数施設が割り当て or 未割当: {assignment_p.get(p,[])}")
    
    #print("Best Energy", best_energy)
    #print("Best Assignment", best_sample)
    """
    ha = HybridAnnealing(
        bqm = bqm, # QUBO (BQM形式)
        qubo_matrix = qubo_matrix, # QUBO行列
        const_constraint = np.sum(qubo_matrix), # 定数項
        num_spin = len(model.variables), # QUBOのサイズ (N * N)
        N_I = 20, # 解インスタンスの数
        N_E = 10, # サブQUBO構築数
        N_S = 5,  # 選択する解インスタンスの数（N_S < N_I）
        sub_qubo_size = 50, # サブQUBOのサイズ
        spin = q, # Amplify形式の変数
        client = client,
        name_to_index = name_to_index
    )
    
    #pool = ha._initialize_pool()
    #print(pool)
    #final_pool, best_solution = ha.run()
    #print("Best Solution:", best_solution)
    
    # デバック用
    """
    
    result = solve(model, client)
    print(result.best.objective)
    q_values = q.evaluate(result.best.values)
    print(q_values)
    """