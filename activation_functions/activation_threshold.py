import context
from activation_functions.base import ActivationFunction

class Threshold(ActivationFunction):
    """
    threshold: a scalar value about which the step function is centered
    """
    def __init__(self, threshold:float=0.0):
        self.threshold = threshold

    def run(self, x:float):
        if x <= self.threshold:
            return 0
        else:
            return 1