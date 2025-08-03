import numpy as np
import dimod
from amplify import VariableGenerator, Model, one_hot

def create_qap_bqm(filepath):
    # --- QAPLIBファイル読み込み ---
    with open(filepath, "r") as f:
        lines = [line for line in f.readlines() if line.strip()]
    n = int(lines[0])
    flow = np.array([list(map(int, lines[i + 1].split())) for i in range(n)])
    dist = np.array([list(map(int, lines[i + 1 + n].split())) for i in range(n)])
    N = flow.shape[0]
    penalty = np.max(dist) * np.max(flow) * (N - 1)

    # --- Amplify変数生成 ---
    gen = VariableGenerator()
    matrix = gen.matrix("Binary", N, N)
    q = matrix.variable_array

    # --- QUBO行列生成 ---
    qubo_obj = np.zeros((N*N, N*N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    qubo_obj[i*N + k, j*N + l] += flow[i, j] * dist[k, l]

    # --- QUBO制約項（厳密なone-hot形式） ---
    lam = penalty
    qubo_constraints = np.zeros((N*N, N*N))
    # 施設側
    for i in range(N):
        for k in range(N):
            idx_ik = i * N + k
            qubo_constraints[idx_ik, idx_ik] += lam * (-2)
            for l in range(N):
                idx_il = i * N + l
                qubo_constraints[idx_ik, idx_il] += lam
    # 場所側
    for k in range(N):
        for i in range(N):
            idx_ik = i * N + k
            qubo_constraints[idx_ik, idx_ik] += lam * (-2)
            for j in range(N):
                idx_jk = j * N + k
                qubo_constraints[idx_ik, idx_jk] += lam

    # --- 定数項（違反ゼロならenergy_constraint=0）
    const_constraint = lam * N * 2

    # --- Amplifyモデル・BQM作成 ---
    constraints = one_hot(q, axis=1) + one_hot(q, axis=0)
    penalty_weight = lam
    model = matrix + penalty_weight * constraints
    poly = model.to_unconstrained_poly()
    variables = poly.variables
    qubo_dict = poly.as_dict()

    linear = {}
    quadratic = {}
    for key, coeff in qubo_dict.items():
        if len(key) == 1:
            name = str(variables[key[0]])
            linear[name] = linear.get(name, 0) + coeff
        elif len(key) == 2:
            name1 = str(variables[key[0]])
            name2 = str(variables[key[1]])
            pair = tuple(sorted((name1, name2)))
            quadratic[pair] = quadratic.get(pair, 0) + coeff

    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)

    
    return (bqm, 
            model, 
            qubo_constraints, 
            qubo_obj, 
            flow, 
            dist, 
            const_constraint
            )