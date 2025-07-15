
import numpy as np

def load_qap_qubo(filepath: str, alpha: int = 1):
    with open(filepath, "r") as f:
        lines = [line for line in f.readlines() if line.strip()]

    n = int(lines[0])
    flow = np.array([list(map(int, lines[i + 1].split())) for i in range(n)])
    dist = np.array([list(map(int, lines[i + 1 + n].split())) for i in range(n)])

    Q = np.zeros((n*n, n*n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    Q[i*n + k, j*n + l] += flow[i, j] * dist[k, l]

    for i in range(n):
        for k in range(n):
            Q[i*n + k, i*n + k] += -2 * alpha
            for l in range(k + 1, n):
                Q[i*n + k, i*n + l] += 2 * alpha

    for k in range(n):
        for i in range(n):
            Q[i*n + k, i*n + k] += -2 * alpha
            for j in range(i + 1, n):
                Q[i*n + k, j*n + k] += 2 * alpha

    const = 2 * alpha * n
    return Q, const
