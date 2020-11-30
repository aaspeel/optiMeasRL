"""
The linear rewarder will add a step error, defined as cost/threshold, multiplied by the number
of measurement taken to the error made by the predictor to have a linearly increasing negative reward.
"""
import numpy as np
from rewarders.rewarder import Rewarder

class LinearRewarder(Rewarder):

    def __init__(self, threshold, windowSize, cost=1):
        """
        Arguments
        ---------
        threshold : the maximum number of measures allowed (must be > 0)
        cost : the cost of taking a measure when numberOfMeasure = threshold
        """
        self._threshold = threshold
        self._cost = cost
        assert(self._threshold > 0)
        self._winSize = windowSize
        self.reset()


    def reset(self):
        """
        Rest internal parameters
        """
        self._errors = []
        self._window = [0] * self._winSize
        self._errorSum = 0
        self._numError = 0


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
        error = (np.square(error).sum())

        #Update window
        del(self._window[0])
        self._window.append(action)

        #Update step (estimated mean error over threshold)
        self._numError += 1
        self._errorSum += error
        #print("mean error:")
        #print(self._errorSum/self._numError)
        step = (self._errorSum/self._numError)/self._threshold

        err = -error - 2 * sum(self._window) * step
        self._errors.append(err)
        return err


    def summarize(self):
        print("threshold: " + str(self._threshold))
        print("cost: " + str(self._cost))
        print("number of measures: " + str(sum(self._window)))
        print("error for measure: " + str(self.get(0,0)))
        print("error history: " + str(self._errors))
        