import random
import numpy as np
from solution import Solution
from amplify import solve, FixstarsClient, Model, VariableGenerator
from dwave.samplers import TabuSampler
from datetime import timedelta


class HybridAnnealing:
    def __init__(self, bqm, qubo_matrix, const_constraint, num_spin,
                 N_I, N_E, N_S, sub_qubo_size, spin, client, name_to_index):
        self.bqm = bqm # model（QUBO式）
        self.qubo_matrix = qubo_matrix
        self.const_constraint = const_constraint
        self.num_spin = num_spin # QUBO size
        self.N_I = N_I # solution instances
        self.N_E = N_E # num of subQUBO
        self.N_S = N_S # num of select solution instamces(N_S < N_I )
        self.client = client
        self.sub_qubo_size = sub_qubo_size # subQUBO size 
        self.spin = spin # amplify spins
        self.name_to_index = name_to_index
        self.pool = self._initialize_pool() # solution instances pool(N_I)
         
        

     # プールを初期化
    def _initialize_pool(self):
        pool = []
        N = int(np.sqrt(self.num_spin))
        for _ in range(self.N_I):
            # one-hotな割当（パーミュテーション行列）を生成
            perm = np.random.permutation(N)
            x_vec = np.zeros((N, N), dtype=int)
            x_vec[np.arange(N), perm] = 1
            x_vec = x_vec.flatten()
            s = Solution(x_vec, qubo=self.qubo_matrix, const=self.const_constraint, N=N)
            pool.append(s)
        return pool

    # Amplify モデルと変数 q を取得して，解いた変数を返す   
    def _evaluate(self, model, spin):
        result = solve(model, self.client)
        q_values = spin.evaluate(result.best.values)
        return q_values
    
    # d-wave タブーサーチ
    def _tabu_search(self, bqm):
        sampler = TabuSampler()
        sampleset = sampler.sample(bqm, num_reads=100)
        return sampleset

    # メイン
    def run(self):
        # 一旦，１
        # 本来は，プール内の平均ハミング距離で判定
        for epoch in range(1):
            for i in range(self.N_I):
                solution_tmp = Solution(self.num_spin)
                solution_tmp.x = np.copy(self.pool[i].x)

            # ランダムにサブQUBO選択
            extracted_idx = sorted(random.sample(range(self.num_spin), self.sub_qubo_size))
            non_extracted_idx = sorted(list(set(range(self.num_spin)) - set(extracted_idx)))

            # サブQUBO作成
            subqubo_obj = self.qubo_obj[np.ix_(extracted_idx, extracted_idx)]
            for idx_i, val_i in enumerate(extracted_idx):
                subqubo_obj[idx_i, idx_i] += sum(self.qubo_obj[val_i, j] * solution_tmp.x[j]
                                                for j in non_extracted_idx)

            subqubo_constraint = self.qubo_matrix[np.ix_(extracted_idx, extracted_idx)]
            
            for idx_i, val_i in enumerate(extracted_idx):
                subqubo_constraint[idx_i, idx_i] += sum(self.qubo_matrix[val_i, j] * solution_tmp.x[j]
                                                        for j in non_extracted_idx)

                const = self.const_constraint - sum(sum(self.qubo_matrix[i, j] * solution_tmp.x[i] * solution_tmp.x[j]
                                                        for j in non_extracted_idx) for i in non_extracted_idx)

                # Amplify用のModelを構築
                gen = VariableGenerator()
                matrix = gen.matrix("Binary", self.sub_qubo_size, self.sub_qubo_size)
                q = matrix.variable_array
                sub_model = sum(subqubo_obj[i, j] * q[i] * q[j] for i in range(self.sub_qubo_size) for j in range(self.sub_qubo_size))
                sub_model += (sum(subqubo_constraint[i, j] * q[i] * q[j] for i in range(self.sub_qubo_size) for j in range(self.sub_qubo_size)) - const) ** 2

                result = self.solve(sub_model, self.client)
                if result.status.name != "SUCCESS":
                    continue

                values = result[0].values
                for idx, var in enumerate(q):
                    bit = values.get(var, 0)
                    solution_tmp.x[extracted_idx[idx]] = bit

                solution_tmp.energy = self._evaluate(solution_tmp.x)
                if solution_tmp.energy < self.pool[i].energy:
                    self.pool[i] = solution_tmp

        best_solution = min(self.pool, key=lambda s: s.energy)
        return best_solution.x, best_solution.energy