import numpy as np

def load_qap_answer(filepath):
    """
    QAP最適解ファイルの施設→場所割当リストを返す（0-indexed）
    """
    with open(filepath, "r") as f:
        lines = [line for line in f.readlines() if line.strip()]
    # 1行目: Nとコスト
    first_line = lines[0].strip().split()
    if len(first_line) == 2:
        # 例: "20    703482"
        N = int(first_line[0])
        cost = int(first_line[1])
        answer_lines = lines[1:]
    else:
        # 1行目にコストがない場合（フォーマット違いに備えて）
        N = int(first_line[0])
        cost = None
        answer_lines = lines[1:]

    assignment = []
    for line in answer_lines:
        nums = [int(x)-1 for x in line.strip().split() if x]
        assignment.extend(nums)
    return assignment, cost  # (長さNリスト, コスト値)

def get_assignment_from_x(x, N):
    """
    QUBO解ベクトルx（長さN*N）から施設→場所割当リスト（0-indexed）を返す
    """
    assignment = []
    for i in range(N):
        row = x[i*N:(i+1)*N]
        place = np.argmax(row)  # 1が立っている場所
        assignment.append(place)
    return assignment  # 長さN

def compare_assignments(ans, sol):
    """
    2つの施設→場所割当リスト（長さN）を比較し、一致数を返す
    """
    matches = sum([a==s for a,s in zip(ans,sol)])
    return matches
