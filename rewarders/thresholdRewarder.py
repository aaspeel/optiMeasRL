"""
The threshold rewarder only gives a cost of taking a measure
when we reach a certain number of measurements (threshold) then
the reward becomes -cost-error such that taking a new measure 
is too costly for the agent.
"""
import numpy as np
from rewarders.rewarder import Rewarder

class ThresholdRewarder(Rewarder):

    def __init__(self, cost=1, threshold=0, windowSize=1):
        """
        Arguments
        ---------
        threshold : the maximum number of measures allowed (must be >= 0)
        windowSize : Size of the sliding history window (must be >=1)
        cost : the 'cost' of taking a measurement when there is more than 'threshold' in the last 'windowSize'.
        NB: threshold=0 and windowSize=1 associates the same 'cost' at each measurement.
        """
        self._threshold = threshold
        self._cost = cost
        self._winSize = windowSize
        self.reset()


    def reset(self):
        """
        Reset internal parameters
        """
        self._window = [0] * self._winSize


    def get(self, error, action, *args, **kwargs):
        """
        Return the reward for the RL agent given a certain criteria
        Arguments
        ---------
        error : the error vector made by the predictor
        action : the action taken by the agent (1 means measure taken and 0 none)
        Returns
        -------
        reward : the reward send to the agent
        """
        # Squared norm of the error
        reward = -np.square(error).sum()

        # Update window
        del(self._window[0])
        self._window.append(action)
        
        # If threshold exceded
        if sum(self._window) > self._threshold:
            reward -= self._cost
        
        return reward


    def summarize(self):
        print("threshold rewarder")
        print("window size: " + str(self._winSize))
        print("threshold: " + str(self._threshold))
        print("cost: " + str(self._cost))
        print("number of measures in the window: " + str(sum(self._window)))
        
