import context
from activation_functions.base import ActivationFunction

class PReLU(ActivationFunction):
    """
    Parametric Rectified Linear Unit
    threshold: a scalar about which the slope changes
    alpha: scalar which multiplies the output past the threshold
    """
    def __init__(self, alpha:float, threshold:float=0.0):
        self.alpha = alpha
        self.threshold = threshold

    def run(self, x:float):
        if x <= self.threshold:
            return x
        else:
            return self.alpha * x