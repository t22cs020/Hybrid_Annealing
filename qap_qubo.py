import numpy as np
import dimod
from amplify import VariableGenerator, Model, one_hot

def create_qap_bqm(filepath, penalty=1000):
    # --- QAPLIBファイル読み込み ---
    with open(filepath, "r") as f:
        lines = [line for line in f.readlines() if line.strip()]
    n = int(lines[0])
    flow = np.array([list(map(int, lines[i + 1].split())) for i in range(n)])
    dist = np.array([list(map(int, lines[i + 1 + n].split())) for i in range(n)])
    N = flow.shape[0]

    # --- Amplify変数生成 ---
    gen = VariableGenerator()
    matrix = gen.matrix("Binary", N, N)
    q = matrix.variable_array

    # --- 制約定義 (one-hot) ---
    constraints = one_hot(q, axis=1) + one_hot(q, axis=0)
    penalty_weight = np.max(dist) * np.max(flow) * (N - 1)
    model = matrix + penalty_weight * constraints

    # --- Poly取得 & QUBO変換 ---
    poly = model.to_unconstrained_poly()
    variables = poly.variables
    qubo_dict = poly.as_dict()

    # --- Ocean SDK 用に変換 ---
    linear = {}
    quadratic = {}
    name_to_index = {}  # 変数名 -> (i,j)
    index_to_name = {}  # (i,j) -> 変数名

    for i in range(N):
        for j in range(N):
            var = q[i][j]
            var_str = str(var)
            name_to_index[var_str] = (i, j)
            index_to_name[(i, j)] = var_str

    for key, coeff in qubo_dict.items():
        if len(key) == 1:
            name = str(variables[key[0]])
            linear[name] = linear.get(name, 0) + coeff
        elif len(key) == 2:
            name1 = str(variables[key[0]])
            name2 = str(variables[key[1]])
            pair = tuple(sorted((name1, name2)))
            quadratic[pair] = quadratic.get(pair, 0) + coeff

    # --- QUBO行列生成 ---
    qubo_matrix = np.zeros((n*n, n*n))
    qubo_obj = np.zeros((n*n, n*n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    qubo_obj[i*n + k, j*n + l] += flow[i, j] * dist[k, l]
                    qubo_matrix[i*n + k, j*n + l] += flow[i, j] * dist[k, l]

    qubo_constraints = np.zeros((n*n, n*n))
    # 施設ごとの one-hot
    for i in range(n):
        for k in range(n):
            qubo_constraints[i*n + k, i*n + k] += -2 * penalty_weight
            qubo_matrix[i*n + k, i*n + k] += -2 * penalty_weight
            for l in range(k + 1, n):
                qubo_constraints[i*n + k, i*n + l] += 2 * penalty_weight
                qubo_matrix[i*n + k, i*n + l] += 2 * penalty_weight

    # 場所ごとの one-hot
    for k in range(n):
        for i in range(n):
            qubo_constraints[i*n + k, i*n + k] += -2 * penalty_weight
            qubo_matrix[i*n + k, i*n + k] += -2 * penalty_weight
            for j in range(i + 1, n):
                qubo_constraints[i*n + k, j*n + k] += 2 * penalty_weight
                qubo_matrix[i*n + k, j*n + k] += 2 * penalty_weight

    # --- BQM作成 ---
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)

    # デバック用
    #print("flow",flow)
    #print("dist",dist)
    #print("qubo_matrix", qubo_matrix)
    
    return bqm, q, model, N, qubo_matrix, qubo_constraints, qubo_obj, name_to_index