
import random
import numpy as np
from solution import Solution

class HybridAnnealing:
    def __init__(self, qubo_obj, qubo_constraint, const_constraint, num_spin,
                 N_I=20, N_E=10, N_S=10, sub_qubo_size=5):
        self.pool = []
        self.N_I = N_I
        self.N_E = N_E
        self.N_S = N_S
        self.sub_qubo_size = sub_qubo_size
        self.num_spin = num_spin
        self.qubo_obj = qubo_obj
        self.qubo_constraint = qubo_constraint
        self.const_constraint = const_constraint
        self._initialize_pool()

    def _initialize_pool(self):
        for _ in range(self.N_I):
            x = np.random.randint(0, 2, self.num_spin)
            energy_obj = Solution.energy(self.qubo_obj, x)
            energy_constraint = Solution.energy(self.qubo_constraint, x, self.const_constraint)
            self.pool.append(Solution(
                x=x,
                energy_all=energy_obj + energy_constraint,
                energy_obj=energy_obj,
                energy_constraint=energy_constraint,
                constraint=Solution.check_constraint(self.qubo_constraint, x, self.const_constraint)
            ))
        self.pool.sort(key=lambda sol: sol.energy_all)

    def run(self):
        x_best = self.pool[0]
        for _ in range(1):
            for sol in self.pool:
                x = np.random.randint(0, 2, self.num_spin)
                sol.x = x
                sol.energy_obj = Solution.energy(self.qubo_obj, x)
                sol.energy_constraint = Solution.energy(self.qubo_constraint, x, self.const_constraint)
                sol.energy_all = sol.energy_obj + sol.energy_constraint
                sol.constraint = Solution.check_constraint(self.qubo_constraint, x, self.const_constraint)

            for _ in range(self.N_E):
                n_s_pool = random.sample(self.pool, self.N_S)
                vars_of_x = np.array([sum(sol.x[j] for sol in n_s_pool) - self.N_S / 2 for j in range(self.num_spin)])
                solution_tmp = random.choice(n_s_pool)
                extracted_idx = np.argsort(vars_of_x)[:self.sub_qubo_size]
                non_extracted_idx = np.argsort(vars_of_x)[self.sub_qubo_size:]

                subqubo_obj = self.qubo_obj[np.ix_(extracted_idx, extracted_idx)]
                subqubo_constraint = self.qubo_constraint[np.ix_(extracted_idx, extracted_idx)]

                for idx_i, val_i in enumerate(extracted_idx):
                    subqubo_obj[idx_i, idx_i] += sum(self.qubo_obj[val_i, j] * solution_tmp.x[j] for j in non_extracted_idx)
                    subqubo_constraint[idx_i, idx_i] += sum(self.qubo_constraint[val_i, j] * solution_tmp.x[j] for j in non_extracted_idx)

                x_sub = np.random.randint(0, 2, self.sub_qubo_size)
                for idx, val in enumerate(extracted_idx):
                    solution_tmp.x[val] = x_sub[idx]

                energy_obj = Solution.energy(self.qubo_obj, solution_tmp.x)
                energy_constraint = Solution.energy(self.qubo_constraint, solution_tmp.x, self.const_constraint)
                solution_tmp.energy_all = energy_obj + energy_constraint
                solution_tmp.energy_obj = energy_obj
                solution_tmp.energy_constraint = energy_constraint
                solution_tmp.constraint = Solution.check_constraint(self.qubo_constraint, solution_tmp.x, self.const_constraint)

                self.pool.append(solution_tmp)

            self.pool.sort(key=lambda sol: sol.energy_all)
            x_best = self.pool[0]
            self.pool = self.pool[:self.N_I]

        return self.pool, x_best
