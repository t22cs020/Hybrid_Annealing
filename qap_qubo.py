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
    qubo_matrix = np.zeros((N * N, N * N))
    for (u, v), coeff in quadratic.items():
        if u in name_to_index and v in name_to_index:
            i1, j1 = name_to_index[u]
            i2, j2 = name_to_index[v]
            idx_u = i1 * N + j1
            idx_v = i2 * N + j2
            qubo_matrix[idx_u, idx_v] += coeff
            if idx_u != idx_v:
                qubo_matrix[idx_v, idx_u] += coeff

    for u, coeff in linear.items():
        if u in name_to_index:
            i, j = name_to_index[u]
            idx = i * N + j
            qubo_matrix[idx, idx] += coeff

    # --- BQM作成 ---
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)

    print("flow",flow)
    print("dist",dist)
    print("qubo_matrix", qubo_matrix)
    
    return bqm, q, model, N, constraints, qubo_matrix, name_to_index