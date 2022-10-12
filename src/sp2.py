import numpy as np


class DischargingProblem(object):

    def __init__(self, alpha: float, Ps: float, kp: float, cG: float, cD: float, pi: float) -> None:
        self.kp = kp
        self.a = alpha / Ps**kp
        self.b = cG - cD - pi

    def solve(self):
        if self.b < 0:
            return 0
        if self.kp == 1:
            if self.a * self.kp - self.b >= 0:
                return 0
            else:
                return np.inf

        return (self.b/(self.a*self.kp)) ** (1/(self.kp-1)) - self.b
