import random
import numpy as np
from solution import Solution
from amplify import solve, FixstarsClient, Model, VariableGenerator
from dwave.samplers import TabuSampler
from datetime import timedelta


class HybridAnnealing:
    def __init__(self, bqm, qubo_obj, qubo_constraints, const_constraint, num_spin,
                 N_I, N_E, N_S, sub_qubo_size, client, flow, dist):
        self.bqm = bqm
        self.qubo_obj = qubo_obj # QUBO行列（コスト項）
        self.qubo_constraints = qubo_constraints # QUBO行列（制約項）
        self.const_constraint = const_constraint # 制約定数
        self.num_spin = num_spin # QUBO size
        self.N_I = N_I # solution instances
        self.N_E = N_E # num of subQUBO
        self.N_S = N_S # num of select solution instamces(N_S < N_I )
        self.client = client # Amplify client情報
        self.sub_qubo_size = sub_qubo_size # subQUBO size 
        self.flow = flow # QAP flow
        self.dist = dist # QAP dist
        self.pool = [] # solution instances pool(N_I)
         
        

    # プール初期化（ランダムに0,1を生成）
    def _initialize_pool(self):
        self.pool = []
        for _ in range(self.N_I):
            # 0,1をランダムに生成
            x = np.random.randint(0, 2, self.num_spin)
            energy_obj = Solution.energy(self.qubo_obj, x)
            energy_constraint = Solution.energy(self.qubo_constraints, x, self.const_constraint)
            self.pool.append(
                Solution(
                    x=x,
                    energy_all=energy_obj + energy_constraint,
                    energy_obj=energy_obj,
                    energy_constraint=energy_constraint,
                    constraint=Solution.check_constraint(self.qubo_constraints, x, self.const_constraint)
                )
            )
        return self.pool
    
    # d-wave タブーサーチ
    def _tabu_search(self):
        sampler = TabuSampler()
        new_pool = []
        for sol in self.pool:
            # Ocean BQMの変数名順にdictを生成
            # bqm.variables: ['x_{i}_{j}', ...]
            initial_state = {v: int(sol.x[idx]) for idx, v in enumerate(self.bqm.variables)}
            # Tabuサーチを初期解から1回実行
            sampleset = sampler.sample(self.bqm, initial_state=initial_state)
            sample = sampleset.first.sample
            # dict→1次元配列xに変換（変数名順）
            x = np.array([sample[v] for v in self.bqm.variables], dtype=int)
            energy_obj = Solution.energy(self.qubo_obj, x)
            energy_constraint = Solution.energy(qubo=self.qubo_constraints, x=x, const=self.const_constraint)
            new_pool.append(
                Solution(
                    x=x,
                    energy_all=energy_obj + energy_constraint,
                    energy_obj=energy_obj,
                    energy_constraint=energy_constraint,
                    constraint=Solution.check_constraint(qubo=self.qubo_constraints, x=x, const=self.const_constraint)
                )
            )
        return new_pool

    def _AE_subQUBO(self, sub_spin_idx, non_sub_spin_idx):
        # Amplifyの変数生成
        gen = VariableGenerator()
        sub_q = gen.array("Binary", self.sub_qubo_size)
                
        # 目的関数（コスト）
        cost = 0
        N = self.flow.shape[0]
        for i, idx_i in enumerate(sub_spin_idx):
            i_fac, i_pos = divmod(idx_i, N)
            for j, idx_j in enumerate(sub_spin_idx):
                j_fac, j_pos = divmod(idx_j, N)
                cost += self.flow[i_fac, j_fac] * self.dist[i_pos, j_pos] * sub_q[i] * sub_q[j]

        # サブQUBO用 制約項 (サブQUBO内で同じ施設/場所が複数回現れる場合のみone-hot制約を課す)
        fac_list = []
        pos_list = []
        
        constraint_terms = []
        for fac in set(fac_list):
            idxs = [i for i, f in enumerate(fac_list) if f == fac]
            if len(idxs) > 1:
                constraint_terms.append((sum(sub_q[i] for i in idxs) - 1) ** 2)
        for pos in set(pos_list):
            idxs = [i for i, p in enumerate(pos_list) if p == pos]
            if len(idxs) > 1:
                constraint_terms.append((sum(sub_q[i] for i in idxs) - 1) ** 2)
        constraints = sum(constraint_terms) if constraint_terms else 0
                
        penalty = np.max(self.flow) * np.max(self.dist) * self.sub_qubo_size 
        sub_model = cost + penalty * constraints
                
        
        sub_result = solve(sub_model, self.client)
        sub_best_sol = sub_result.best.values
        sub_qubo_assignment = [sub_best_sol[sub_q[i]] for i in range(self.sub_qubo_size)]
               
        # 解インスタンスからランダムに解インスタンスを１つ選択 
        X_t = random.choice(self.pool)
                
        # subQUBO と X_t を組み合わせた新たな解
        new_X = np.zeros(self.num_spin, dtype = int)
        for i, idx in enumerate(sub_spin_idx):
            new_X[idx] = sub_qubo_assignment[i]
                    
        for idx in non_sub_spin_idx:
            new_X[idx] = X_t.x[idx]
                
        energy_obj = Solution.energy(self.qubo_obj, new_X)
        energy_constraint = Solution.energy(qubo=self.qubo_constraints, x=new_X, const=self.const_constraint)
        new_sol = Solution(
            x=new_X,
            energy_all=energy_obj + energy_constraint,
            energy_obj=energy_obj,
            energy_constraint=energy_constraint,
            constraint=Solution.check_constraint(qubo=self.qubo_constraints, x=new_X, const=self.const_constraint)
        )
        return new_sol
    
    # one-hot 制約の判定
    def count_qap_violations(self, x):
        facility_violation = 0
        location_violation = 0
        N = self.flow.shape[0]

        # 施設ごと
        for i in range(N):
            assigned = sum(x[i*N + p] for p in range(N))
            if assigned != 1:
                facility_violation += 1

        # 場所ごと
        for p in range(N):
            assigned = sum(x[i*N + p] for i in range(N))
            if assigned != 1:
                location_violation += 1

        return facility_violation, location_violation
    
    # プール内の解同士の平均ハミング距離を計算
    def _average_hamming_distance(self, pool):
        if len(pool) <= 1:
            return 0
        total = 0
        count = 0
        for i in range(len(pool)):
            for j in range(i+1, len(pool)):
                total += np.sum(pool[i].x != pool[j].x)
                count += 1
        return total / count if count > 0 else 0


    # メイン
    def run(self):
        self.pool = self._initialize_pool()
        
        # プール内の平均ハミング距離で判定
        # Line 6 
        while True:
            avg_hamming = self._average_hamming_distance(self.pool)
            if avg_hamming < self.sub_qubo_size:
                break
            
            # Line 8
            self.pool = self._tabu_search()
            
            # 本来は，N_E
            # Line 9
            for i in range(self.N_E):
            
                # Line 10
                # 解インスタンスを NS 個ランダムに選択
                selected_pool = []
                selected_pool = random.sample(self.pool, self.N_S)
            
                # 変数のばらつき
                # Line 11~14
                c = np.zeros(self.num_spin)
                d = np.zeros(self.num_spin)
            
                for j in range(self.num_spin):
                    for k in range(len(selected_pool)):
                        c[j] += selected_pool[k].x[j]
                    d[j] = abs(c[j] - self.N_S / 2)
            
                select_idx = np.argsort(d)
                
                sub_spin_idx = select_idx[:self.sub_qubo_size]
                non_sub_spin_idx = select_idx[self.sub_qubo_size:]
                
                # サブQUBO を AE で解く
                # Line 15
                new_sol = self._AE_subQUBO(sub_spin_idx, non_sub_spin_idx)
                
                # Line 16
                self.pool.append(new_sol)
                
            self.pool = sorted(self.pool, key=lambda sol: sol.energy_all)[:self.N_I]
        
        # 最適解
        best_solution = min(self.pool, key=lambda sol: sol.energy_all)
        # 制約違反数
        facility_violation, location_violation = self.count_qap_violations(best_solution.x)
        
        return facility_violation, location_violation, best_solution