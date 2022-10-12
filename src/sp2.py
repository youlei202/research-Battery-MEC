import numpy as np


class DischargingProblem(object):

    def __init__(self,
                 dual_var: np.array,
                 alpha: float, kp: int,
                 cG: np.array, cD: np.array,  pS: np.array,
                 ) -> None:
        self.kp = kp
        self.a = alpha / pS**kp
        self.b = cG - cD - dual_var

    def solve(self):
        n = len(self.a)
        p = []

        if self.kp == 1:
            for i in range(n):
                p.append(0 if self.a[i] - self.b[i] >= 0 else np.inf)
        else:
            for i in range(n):
                if self.b[i] < 0:
                    p.append(0)
                else:
                    p.append((self.b/(self.a*self.kp))
                             ** (1/(self.kp-1)) - self.b)

        return np.array(p)
