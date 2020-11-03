"""
Penalization class
The default behavior is to just use -error of the predictor as the
penalization (the RL agent should then always take a measurement)
"""

class Penalization:

    def __init__(self):
        """
        Implement if needed.
        """

    def reset(self):
        """
        Rest internal parameters
        """

    def get(self, error, *args, **kwargs):
        """ Return the penalization for the RL agent given a certain criteria
        Arguments
        ---------
        error : the error made by the predictor
        -------
        penalization : the penalization send to the agent
        """
        return -error
        
