
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

class SegRefineEnv:
    """State = (M_t, P, X_in optional).
    Actions: +1 (expand), -1 (contract). This is a *minimal* reference implementation.
    """
    def __init__(self, P, M0, X_in=None, iters=4):
        self.P = P.astype(np.float32)  # (D,H,W)
        self.M = M0.astype(np.uint8)
        self.X = X_in
        self.iters = iters
        self.t = 0

    def step(self, action):
        if action == 1:   # expansion
            self.M = binary_dilation(self.M, iterations=1).astype(np.uint8)
        elif action == -1: # contraction
            self.M = binary_erosion(self.M, iterations=1).astype(np.uint8)
        self.t += 1
        done = self.t >= self.iters
        return self.M, done
