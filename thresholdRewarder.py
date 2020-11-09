"""
The threshold rewarder only gives a cost of taking a measure
when we reach a certain number of measurements (threshold) then
the error becomes -cost-error such that taking a new measure 
is too costly for the agent.
"""
import numpy as np
from rewarder import Rewarder

class ThresholdRewarder(Rewarder):

    def __init__(self, threshold, windowSize, cost=2):
        """
        Arguments
        ---------
        threshold : the maximum number of measures allowed (must be > 0)
        cost : the cost of taking a measure when sum of measure >= threshold
        """
        self._threshold = threshold
        self._cost = cost
        self._winSize = windowSize
        self.reset()


    def reset(self):
        """
        Rest internal parameters
        """
        self._errors = []
        self._window = [0] * self._winSize


    def get(self, error, action, *args, **kwargs):
        """ Return the reward for the RL agent given a certain criteria
        Arguments
        ---------
        error : the error vector made by the predictor
        action : the action taken by the agent (1 means measure taken and 0 none)
        Returns
        -------
        reward : the reward send to the agent
        """
        #MSE of error
        error = (np.square(error).mean())

        #Update window
        del(self._window[0])
        self._window.append(action)

        if sum(self._window) > self._threshold:
            err = -self._cost - error
            self._errors.append(err)
            return err

        self._errors.append(-error)
        return -error


    def info(self):
        print("threshold: " + str(self._threshold))
        print("cost: " + str(self._cost))
        print("number of measures: " + str(sum(self._window)))
        print("error for measure: " + str(self.get(0,0)))
        print("error history: " + str(self._errors))
        
