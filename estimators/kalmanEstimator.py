""" 
Class kalmanEstimator
"""

from estimators.estimator import Estimator
from utils.linear_systems import *

class KalmanEstimator(Estimator):
    
    def __init__(self):
        """
        Construct the estimator.
        """
        self._kf=loadKF()
        
        self._n_dim_state=self._kf.n_dim_state
        self._n_dim_obs=self._kf.n_dim_obs
        self._n_dim_obj=self._kf.n_dim_obj
        
        self.reset()
        
    def reset(self):
        """
        Initialize the constructor if necessary.
        """
        self._filtered_state=None
        self._filtered_state_covariance=None
    
    def estimate(self,observation_corrupted):
        """
        Return the estimate from corrupted informations.
        """
        self._filtered_state, self._filtered_state_covariance, current_objective_pred, _ = KFFilterOne(self._kf, observation_corrupted.reshape(-1), state_mean=self._filtered_state, state_mean_covariance=self._filtered_state_covariance)
        
        return current_objective_pred
    
    def estimateAll(self,observations_corrupted):
        """
        Return the estimate from corrupted informations.
        """
        estimated_objectives, objective_covariances, estimated_states, state_covariances = KFFilterAll(self._kf,observations_corrupted)
        return estimated_objectives
        
    def generateSequence(self, T, numberSamples=1):
        """
        Facultative, generate sequences for which the estimator is designed.
        """
        (objectives,observations,_)=sampleKFSequence(self._kf,T,numberSamples=numberSamples)
        
        return (objectives,observations)
    