""" 
Class kalmanEstimator
"""

from estimators.estimator import Estimator
from utils.linear_systems import *

class KalmanEstimator(Estimator):
    
    def __init__(self,seeAction=True,seeMeasurement=True,seeEstimate=False,seeTime=False,seeCovariance=False):
        """
        Construct the estimator.
        """
        self._kf=loadKF()
        
        self._n_dim_state=self._kf.n_dim_state
        self._n_dim_meas=self._kf.n_dim_obs # change of notation
        self._n_dim_obj=self._kf.n_dim_obj
        
        self._seeAction=seeAction
        self._seeMeasurement=seeMeasurement
        self._seeEstimate=seeEstimate
        self._seeTime=seeTime
        self._seeCovariance=seeCovariance
        
        self.reset()
    
    
    def reset(self):
        """
        Initialize the constructor if necessary.
        """
        self._filtered_state=None
        self._filtered_state_covariance=None
        
        # reset observations
        self._last_action=0
        self._last_measurement_outOfRange=self._n_dim_meas*[self.outOfRangeValue()]
        self._last_estimate=self._n_dim_obj*[self.outOfRangeValue()] # could be different
        self._time=-1
    
    
    def estimate(self,measurement_corrupted):
        """
        Return the estimate from corrupted informations.
        """
        # compute the estimate and update the internal state for Kalman filtering
        self._filtered_state, self._filtered_state_covariance, current_objective_est, _ = KFFilterOne(self._kf, measurement_corrupted.reshape(-1), state_mean=self._filtered_state, state_mean_covariance=self._filtered_state_covariance)
        
        # storage for observation
        if measurement_corrupted.mask[0]: # masked
            self._last_action=0
        else:
            self._last_action=1
        self._last_measurement_outOfRange=measurement_corrupted.filled(self.outOfRangeValue())
        self._last_estimate=current_objective_est
        self._time+=1
        
        return current_objective_est
    
    
    def observe(self):
        """
        Return an observation to help the reinforcement learning agent.
        """
        observation=[]
        if self._seeAction:
            observation.append( self._last_action )
        if self._seeMeasurement:
            observation.append( self._last_measurement_outOfRange )
        if self._seeEstimate:
            observation.append( self._last_estimate )
        if self._seeTime:
            observation.append( 1-1/(self._time+2) ) # 1-1/(self._time+2) to represent the current time in [0,1[
        if self._seeCovariance:
            pass
        
        return observation
    
    
    def observationsDimensions(self):
        """
        Facultative
        Return the shape of an obsevation (including the action and the history size).
        """
        sigmaHistorySize=12 # T-1
        measurementHistorySize=12
        estimateHistorySize=12
        
        dim=[]
        if self._seeAction:
            dim.append( (sigmaHistorySize,) )
        if self._seeMeasurement:
            dim.append( (measurementHistorySize,self._n_dim_meas) )
        if self._seeEstimate:
            dim.append( (estimateHistorySize,self._n_dim_obj) )
        if self._seeTime:
            dim.append( (1,) )
        if self._seeCovariance:
            print('Option to see covariance not implemented')
        
        return dim
    
    
    def estimateAll(self,measurements_corrupted):
        """
        Return the estimate from corrupted informations.
        """
        estimated_objectives, objective_covariances, estimated_states, state_covariances = KFFilterAll(self._kf,measurements_corrupted)
        return estimated_objectives
    
    
    def generateSequence(self, T, numberSamples=1):
        """
        Facultative, generate sequences for which the estimator is designed.
        """
        (objectives,measurements,_)=sampleKFSequence(self._kf,T,numberSamples=numberSamples)
        
        return (objectives,measurements)
    
    
    def summarize(self):
        """
        Facultative
        Print a summary of the predictor.
        """
        print('Kalman estimator')
        print('  observationsDimensions:',self.observationsDimensions())
        print('  seeAction=',self._seeAction)
        print('  seeMeasurement=',self._seeMeasurement)
        print('  seeEstimate=',self._seeEstimate)
        print('  seeTime=',self._seeTime)
        print('  seeCovariance=',self._seeCovariance)
        
    