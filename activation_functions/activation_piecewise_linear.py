import context
from activation_functions.base import ActivationFunction

class PiecewiseLinear(ActivationFunction):
    """
    Piecewise Linear
    alpha: scalar determining the slope and window of inner piecewise function
    """
    def __init__(self, alpha:float):
        self.alpha = alpha
        if self.alpha <= 0:
            raise("Parameter alpha must be positive.")

    def run(self, x:float):
        if x <= -self.alpha:
            return -1
        elif -self.alpha < x and x < self.alpha:
            return x / self.alpha
        else:
            return 1