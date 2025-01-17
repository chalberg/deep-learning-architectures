import context
import numpy as np
from activation_functions.base import ActivationFunction

class Linear(ActivationFunction):
    """
    f(x) = scalar * x
    alpha: scalar value to multiply input by (float)
    """
    def __init__(self, alpha:float):
        self.alpha = alpha

    def run(self, x:float|np.array):
        return self.alpha * x
    
    def derivative(self, x:float|np.array):
        return np.ones_like(x) * self.alpha