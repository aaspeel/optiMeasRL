"""
Rewarder class
The default behavior is to just use -error of the predictor as the
reward (the RL agent should then always take a measure)
"""

class Rewarder:

    def __init__(self):
        """
        Implement if needed.
        """

    def reset(self):
        """
        Rest internal parameters
        """

    def get(self, error, action, *args, **kwargs):
        """ Return the reward for the RL agent given a certain criteria
        Arguments
        ---------
        error : the error made by the predictor
        action : the action taken by the agent (1 means measure taken and 0 none)
        Returns
        -------
        reward : the reward send to the agent
        """
        
        return NotImplementedError()
        
