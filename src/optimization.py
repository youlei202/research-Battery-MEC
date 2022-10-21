from typing import Optional, Tuple

import numpy as np

from src.sp1 import NetworkFlowProblem
from src.sp2 import DischargingProblem


class MetaProblem:

    def __init__(self,
                 alpha: float, kp: int,
                 lam: np.array, mu: np.array, p: np.array,
                 cG: np.array, cD: np.array, pS: np.array, C: np.matrix,
                 ) -> None:
        self.n = len(lam)
        self.alpha = alpha
        self.kp = kp
        self.lam = lam
        self.mu = mu
        self.p = p
        self.cG = cG
        self.cD = cD
        self.pS = pS
        self.C = C

        self.pD = np.zeros(self.n)
        self.X = np.zeros((self.n, self.n))

    def optimize(self, **kwargs) -> Tuple[float, np.array, np.matrix]:
        pass

    def _objective_function(self, pD: np.array, X: np.matrix) -> float:
        return np.sum(
            (self.lam - np.asarray(np.dot(X, np.ones(self.n))).squeeze() +
             np.asarray(np.dot(np.ones(self.n), X)).squeeze()) * self.p * self.cG / self.mu
            - pD * (self.cG - self.cD) + self.alpha * (pD / self.pS) ** self.kp +
            np.asarray(np.dot(X*self.C, np.ones(self.n))).squeeze()
        )


class Problem(MetaProblem):

    def __init__(self,
                 alpha: float, kp: int,
                 lam: np.array, mu: np.array, p: np.array,
                 cG: np.array, cD: np.array, pS: np.array, C: np.matrix,
                 ) -> None:
        super().__init__(alpha=alpha, kp=kp, lam=lam, mu=mu,
                         p=p, cG=cG, cD=cD, pS=pS, C=C)
        self.best_dual = -np.inf
        self.dual_var = np.zeros(self.n)
        self.best_obj = np.inf

        self.best_X = self.X
        self.best_pD = self.pD

    def optimize(self, max_iter: Optional[int] = 100) -> Tuple[float, np.array, np.matrix]:

        MAX_NO_PROGRESS_STEPS = 10
        no_progress_steps = 0

        last_best_dual = -np.inf

        for i in range(max_iter):

            self.subproblem_1, self.subproblem_2 = self._lagrangian_decomposition(
                self.dual_var)
            _, _ = self.subproblem_1.solve()

            self.X = self.subproblem_1.X
            self.pD = self.subproblem_2.solve()

            self.pD, self.X = self._heuristic(self.pD, self.X)

            penalty = self._lagrangian_penalty(pD=self.pD, X=self.X)
            dual = self._lagrangian_dual_function(
                dual_var=self.dual_var, pD=self.pD, X=self.X)

            obj = self._objective_function(pD=self.pD, X=self.X)

            numerator = abs(obj - dual)
            denominator = np.linalg.norm(penalty, 2)**2
            iter_rate = min(
                numerator/denominator if denominator > 0 else 0.02, 2)

            # print(
            #     f'Round={i}, dual={dual}, obj={obj}, iter_rate={iter_rate}'
            # )

            if numerator < 1e-15:
                break

            if last_best_dual - self.best_dual < 1e-15:
                no_progress_steps += 1
            else:
                no_progress_steps = 0

            if no_progress_steps > MAX_NO_PROGRESS_STEPS:
                iter_rate /= 2

            last_dual_var = self.dual_var
            self.dual_var = (self.dual_var + iter_rate*penalty).clip(min=0)

            if np.linalg.norm(last_dual_var-self.dual_var, 2) < 1e-15:
                break

        self.best_obj = obj
        self.best_X = self.X
        self.best_pD = self.pD
        self.best_dual = dual
        print(f'Best Obj={self.best_obj}, Best Dual={self.best_dual}')

        return self.best_obj, self.best_pD, self.best_X

    def _lagrangian_decomposition(self, dual_var) -> None:

        weights = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                weights[i][j] = (
                    (self.p[j]*self.cG[j]/self.mu[j] - self.p[j]*dual_var[j]/self.mu[j]) +
                    (self.C[i][j] - self.p[i]*self.cG[i]/self.mu[i] +
                     self.p[i]*dual_var[i]/self.mu[i])
                )

        subproblem_1 = NetworkFlowProblem(
            arrival_rates=self.lam, weights=weights)

        subproblem_2 = DischargingProblem(
            dual_var=dual_var,
            alpha=self.alpha, kp=self.kp,
            cG=self.cG, cD=self.cD, pS=self.pS,
        )
        return subproblem_1, subproblem_2

    def _lagrangian_dual_function(self, dual_var: np.array, pD: np.array, X: np.matrix) -> float:
        return self._objective_function(pD=pD, X=X) + sum(dual_var * self._lagrangian_penalty(pD=pD, X=X))

    def _lagrangian_penalty(self, pD: np.array, X: np.matrix) -> float:
        return pD - (self.lam - np.asarray(np.dot(X, np.ones(self.n))).squeeze() + np.asarray(np.dot(np.ones(self.n), X)).squeeze()) * self.p / self.mu

    def _heuristic(self, pD, X) -> Tuple[np.array, np.matrix]:
        for i in range(self.n):
            rhs = (self.lam[i] - np.sum(X[i]) +
                   np.sum(X[:, i])) * self.p[i] / self.mu[i]
            pD[i] = max(min(pD[i], rhs, self.pS[i]), 0)
        return pD, X


class BaselineProblem(MetaProblem):

    def __init__(self,
                 alpha: float, kp: int,
                 lam: np.array, mu: np.array, p: np.array,
                 cG: np.array, cD: np.array, pS: np.array, C: np.matrix,
                 ) -> None:
        super().__init__(alpha=alpha, kp=kp, lam=lam, mu=mu,
                         p=p, cG=cG, cD=cD, pS=pS, C=C)
        weights_baseline = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                weights_baseline[i][j] = (
                    p[j]*cG[j]/mu[j] + C[i][j] - p[i]*cG[i]/mu[i])

        self.problem = NetworkFlowProblem(
            arrival_rates=lam, weights=weights_baseline)

    def optimize(self) -> Tuple[float, np.array, np.matrix]:
        min_cost, _ = self.problem.solve()
        self.X = self.problem.X
        self.pD = np.zeros(self.n)

        obj = self._objective_function(pD=self.pD, X=self.X)
        # print(f'Baseline objective = {obj}')

        return obj, self.pD, self.X
