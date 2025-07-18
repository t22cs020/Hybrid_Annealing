import random
import numpy as np
from solution import Solution
from amplify import solve, FixstarsClient, Model, VariableGenerator
from dwave.samplers import TabuSampler
from datetime import timedelta


class HybridAnnealing:
    def __init__(self, bqm, qubo_matrix, qubo_obj, qubo_constraints, const_constraint, num_spin,
                 N_I, N_E, N_S, sub_qubo_size, spin, client, name_to_index):
        self.bqm = bqm # model（QUBO式）
        self.qubo_matrix = qubo_matrix # QUBO行列
        self.qubo_obj = qubo_obj # QUBO行列（コスト項）
        self.qubo_constraints = qubo_constraints # QUBO行列（制約項）
        self.const_constraint = const_constraint # 制約定数
        self.num_spin = num_spin # QUBO size
        self.N_I = N_I # solution instances
        self.N_E = N_E # num of subQUBO
        self.N_S = N_S # num of select solution instamces(N_S < N_I )
        self.client = client
        self.sub_qubo_size = sub_qubo_size # subQUBO size 
        self.spin = spin # amplify spins
        self.name_to_index = name_to_index
        self.pool = self._initialize_pool() # solution instances pool(N_I)
         
        

    # プール初期化（Tabuサーチを使う版）
    def _initialize_pool(self):
        sampler = TabuSampler()
        pool = []
        # Tabuサーチで連続N_I回サンプリング
        sampleset = sampler.sample(self.bqm, num_reads= self.N_I)
        # サンプルセットから N_I 個のサンプルを取得
        for sample in sampleset.samples():
            # Ocean形式 {'x_{i}_{j}': 0/1, ...} から1次元配列xに変換 (変数ソート順を合わせる)
            # name_to_index: {'x_{i}_{j}': (i, j)}
            # num_spin = N*N
            x = np.zeros(len(self.qubo_obj), dtype=int)
            for idx, var in enumerate(sorted(sample.keys())):
                x[idx] = sample[var]
            energy_obj = Solution.energy(self.qubo_obj, x)
            energy_constraint = Solution.energy(qubo = self.qubo_matrix, x=x, const = self.const_constraint)
            pool.append(
                Solution(
                    x=x,
                    energy_all=energy_obj + energy_constraint,
                    energy_obj=energy_obj,
                    energy_constraint=energy_constraint,
                    constraint=Solution.check_constraint(qubo = self.qubo_matrix, x=x, const=self.const_constraint)
                )
            )
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