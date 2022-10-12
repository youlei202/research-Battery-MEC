from typing import Optional, Tuple

import numpy as np

from src.sp1 import NetworkFlowProblem
from src.sp2 import DischargingProblem


class Problem():

    def __init__(self,
                 alpha: float, kp: int,
                 lam: np.array, mu: np.array, p: np.array,
                 cG: np.array, cD: np.array, pS: np.array, C: np.matrix,
                 ) -> None:
        self.n = len(lam)
        self.best_dual = -10000

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
        self.dual_var = np.ones(self.n)

    def optimize(self, max_iter: Optional[int] = 100) -> Tuple[float, np.array, np.matrix]:

        for i in range(max_iter):
            print(f'Iteration {i+1}')
            self.subproblem_1, self.subproblem_2 = self._lagrangian_decomposition(
                self.dual_var)
            _, _ = self.subproblem_1.solve()
            print('\t SP1 solved')

            self.X = self.subproblem_1.X
            self.pD = self.subproblem_2.solve()
            print('\t SP2 solved')

            self.pD, self.X = self._heuristic(self.pD, self.X)

            dual = self._lagrangian_dual_function(
                dual_var=self.dual_var, pD=self.pD, X=self.X)

            if dual > self.best_dual:
                self.best_dual = dual

            penalty = self._lagrangian_penalty(pD=self.pD, X=self.X)

            numerator = abs(self.best_dual - dual)
            denominator = np.linalg.norm(penalty, 2)**2

            iter_rate = max(
                min(numerator/denominator if denominator > 0 else 0.01, 0.01), 2)
            print(
                '\t', f'best_dual={self.best_dual}, dual={dual}, iter_rate={iter_rate}')

            self.dual_var = (self.dual_var + iter_rate*penalty).clip(min=0)

        obj = self._objective_function(pD=self.pD, X=self.X)
        print(f'objective={obj}')

        return obj, self.pD, self.X

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

    def _objective_function(self, pD: np.array, X: np.matrix) -> float:
        return np.sum(
            (self.lam - np.asarray(np.dot(X, np.ones(self.n))).squeeze() +
             np.asarray(np.dot(np.ones(self.n), X)).squeeze()) * self.p * self.cG / self.mu
            - pD * (self.cG - self.cD) + self.alpha * (pD / self.pS) ** self.kp +
            np.asarray(np.dot(X*self.C, np.ones(self.n))).squeeze()
        )

    def _heuristic(self, pD, X) -> Tuple[np.array, np.matrix]:
        for i in range(self.n):
            rhs = (self.lam[i] - np.sum(X[i]) +
                   np.sum(X[:, i])) * self.p[i] / self.mu[i]
            pD[i] = min(pD[i], rhs)
        return pD, X
