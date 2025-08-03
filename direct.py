from amplify import solve, FixstarsClient, VariableGenerator, Model, one_hot
import numpy as np
from ans import load_qap_answer, get_assignment_from_x, compare_assignments

if __name__ == "__main__":
    # 60以上は無理
    pro_filepath = "QAPLIB/tai20a.dat.txt"
    ans_filepath = "QAPLIB/tai20a_ans.dat.txt"
    
    client = FixstarsClient()
    client.token = "AE/eTNJSueTP0TJJa9NsloAvTMyxjQIoJ2X"
    
    with open(pro_filepath, "r") as f:
        lines = [line for line in f.readlines() if line.strip()]
    n = int(lines[0])
    flow = np.array([list(map(int, lines[i + 1].split())) for i in range(n)])
    dist = np.array([list(map(int, lines[i + 1 + n].split())) for i in range(n)])
    N = flow.shape[0]
    penalty = np.max(dist) * np.max(flow) * (N - 1)
    
    gen = VariableGenerator()
    matrix = gen.matrix("Binary", N, N)  # coefficient matrix
    q = matrix.variable_array
    constraints = one_hot(q, axis=1) + one_hot(q, axis=0)
    penalty_weight = np.max(dist) * np.max(flow) * (N - 1)
    model = matrix + penalty_weight * constraints
    
    ans_assignment, ans_energy = load_qap_answer(ans_filepath)
        
    num_measur = 10
    total_sol = 0
    for i in range(num_measur):
        result = solve(model, client)
        print(f"得られた解のコスト: {result.best.objective}")
        print(f"コスト差分: {result.best.objective - ans_energy}")
        print(f"解の精度（f(X*)/f(X)）: {ans_energy / result.best.objective}")
        print(f"解の精度（f(X)/f(X*)）: {result.best.objective / ans_energy}")
        print(f"AE合計時間: {result.execution_time.total_seconds():.3f}秒")
        print("-----")
        total_sol += result.best.objective
        
    print(f"平均解: {total_sol / num_measur}")