import context
import numpy as np
from activation_functions.base import ActivationFunction

class Softmax(ActivationFunction):
    """
    Softmax
    x: an array of floats of length n
    f(x_i) = e^{alpha * x_i} / sum_n e^{alpha * x_j}
    """
    def __init__(self, alpha:float):
        self.alpha = alpha
        if self.alpha <= 0:
            raise("Parameter alpha must be positive.")

    def run(self, x:np.array):
        denominator = np.sum([np.exp(self.alpha*x_i) for x_i in x])
        weights = [np.exp(self.alpha*x_i) / denominator for x_i in x]
        return np.where(weights == np.max(weights))