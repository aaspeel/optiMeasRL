""" 
Class rnnEstimator
"""

import numpy as np

from estimators.estimator import Estimator
from utils.sequences_treatment import generateSequence

# used in function convert_to_inference_model
import json
from keras.models import model_from_json

class RnnEstimator(Estimator):
    
    def __init__(self,model,generatorType,outOfRangeValue=-1,seeAction=True,seeMeasurement=True,seeEstimate=False,seeTime=False):
        """
        Construct the estimator.
        model and generatorType must have compatible dimensions
        """
        # Could be nice to add defaults values for model and generatorType (but not easy)
        
        self._n_dim_meas=model._feed_input_shapes[0][2]-1 # we don't count sigma
        self._n_dim_obj=model._feed_output_shapes[0][2]
        
        self._model_stateless=model # store estimateAll() and possible re-training of the model
        
        # convert the stateless model to a stateful model (required for online prediction)
        self._model=convert_to_inference_model(model)
        
        self._generatorType=generatorType
        self._outOfRangeValue=outOfRangeValue
        
        self._seeAction=seeAction
        self._seeMeasurement=seeMeasurement
        self._seeEstimate=seeEstimate
        self._seeTime=seeTime
        
        self.reset()
        
    def reset(self):
        """
        Reset the estimator if necessary.
        """
        self._model.reset_states()
        
        # reset observations
        self._last_action=0
        self._last_measurement_outOfRange=self._n_dim_meas*[self.outOfRangeValue()]
        self._last_estimate=self._n_dim_obj*[self.outOfRangeValue()] # could be different
        self._time=-1
        
    def estimate(self,measurement_corrupted):
        """
        Return the estimate from corrupted measurement.
        measurement_corrupted.shape=(1,1,n_dim_meas)
        """
        sigma=1-measurement_corrupted.mask[0,0,0]
        measurement_corrupted_outOfRange=measurement_corrupted.filled(self._outOfRangeValue)
        
        # input of the rnn is the sigma and the corrupted measurement
        inputRNN=np.concatenate(([[[sigma]]],measurement_corrupted_outOfRange),axis=2)
        
        # convert the corruption with mask to a corruption with outOfRangeValue
        current_objective_est=self._model.predict(inputRNN)
        
        # storage for observation
        self._last_action=sigma  
        self._last_estimate=current_objective_est
        self._last_measurement_outOfRange=measurement_corrupted_outOfRange
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
            observation.append( 1-1/(self._time+2) ) # to represent the current time in [0,1[
        
        return observation
    
    def observationsDimensions(self):
        """
        Facultative
        Return the shape of an obsevation (including the action and the history size).
        """
        sigmaHistorySize=5
        measurementHistorySize=5
        estimateHistorySize=5
        
        dim=[]
        if self._seeAction:
            dim.append( (sigmaHistorySize,) )
        if self._seeMeasurement:
            dim.append( (measurementHistorySize,self._n_dim_meas) )
        if self._seeEstimate:
            dim.append( (estimateHistorySize,self._n_dim_obj) )
        if self._seeTime:
            dim.append( (1,) )
        
        return dim
    
    def estimateAll(self,measurements_corrupted):
        """
        ! ! ! Use the stateless model (do not update the internal state).
        Return the estimate from corrupted informations.
        """
        # convert the corruption with mask to a corruption with outOfRangeValue
        sigmas=1-measurements_corrupted.mask # has 3 dimensions
        inputRNN=np.concatenate( (sigmas,measurements_corrupted.filled(self._outOfRangeValue)) ,axis=2)
        
        objectives_est=self._model_stateless.predict(inputRNN)
        return objectives_est
    
    
    def outOfRangeValue(self):
        """
        Return a value out of the range of the sequence of measurements.
        """
        return self._outOfRangeValue
    
    
    def generateSequence(self,T,numberSamples=1):
        """
        Facultative, generate sequences ( for which the estimator is designed.
        Return (objectives,measurements) with shapes (numberSamples,T,:)
        """
        (objectives,measurements)=generateSequence(T,numberSamples=numberSamples,generatorType=self._generatorType)
        
        return (objectives,measurements)
    
    def summarize(self):
        """
        Facultative
        Print a summary of the predictor.
        """
        print('RNN estimator')
        print('  generatorType:',self._generatorType)
        print('  observationsDimensions:',self.observationsDimensions())
        print('  seeAction=',self._seeAction)
        print('  seeMeasurement=',self._seeMeasurement)
        print('  seeEstimate=',self._seeEstimate)
        print('  seeTime=',self._seeTime)
    
def convert_to_inference_model(original_model):
    """
    Static function.
    Function to convert a Keras LSTM model trained as stateless to a stateful model expecting
    a single sample and time step as input to use in inference.
    https://gist.github.com/rpicatoste/02cecac1ed52524301e3ab423dac888b
    """
    original_model_json = original_model.to_json()
    inference_model_dict = json.loads(original_model_json)

    layers = inference_model_dict['config']['layers']
    for layer in layers:
        if 'stateful' in layer['config']:
            layer['config']['stateful'] = True

        if 'batch_input_shape' in layer['config']:
            layer['config']['batch_input_shape'][0] = 1
            layer['config']['batch_input_shape'][1] = None

    inference_model = model_from_json(json.dumps(inference_model_dict))
    inference_model.set_weights(original_model.get_weights())

    return inference_model
