""" 
Class rnnEstimator
"""

from estimator import Estimator
from myUtils.sequences_treatment import generateSequence

# used in function convert_to_inference_model
import json
from keras.models import model_from_json

class RnnEstimator(Estimator):
    
    def __init__(self,model,generatorType,outOfRangeValue=-1):
        """
        Construct the estimator.
        """
        # Could be nice to add defaults values for model and generatorType (but not easy)
        
        self._model_stateless=model # store estimateAll() and possible re-training of the model
        
        # convert the stateless model to a stateful model (required for online prediction)
        self._model=convert_to_inference_model(model)
        
        self._generatorType=generatorType
        self._outOfRangeValue=outOfRangeValue
        self.reset()
        
    def reset(self):
        """
        Reset the estimator if necessary.
        """
        self._model.reset_states()
    
    def estimate(self,observation_corrupted):
        """
        Return the estimate from corrupted informations.
        """
        # convert the corruption with mask to a corruption with outOfRangeValue
        current_objective_pred=self._model.predict(observation_corrupted.filled(self._outOfRangeValue))
        return current_objective_pred
    
    def estimateAll(self,observations_corrupted):
        """
        ! ! ! Use the stateless model (do not update the internal state).
        Return the estimate from corrupted informations.
        """
        # convert the corruption with mask to a corruption with outOfRangeValue
        objectives_pred=self._model_stateless.predict(observations_corrupted.filled(self._outOfRangeValue))
        return objectives_pred
            
    def extraInfo(self):
        """
        Return extra information to help the reinforcement learning agent.
        """
        return None
    
    def shapeExtraInfo(self):
        """
        Return the shape of the extra information.
        """
        return ()
    
    def summarize(self):
        """
        Facultative, print a summary of the predictor.
        """
        print("No function summarize() implemented in default Estimator class.")
        
    def generateSequence(self,T,numberSamples=1):
        """
        Facultative, generate sequences ( for which the estimator is designed.
        Return (objectives,observations) with shapes (numberSamples,T,:)
        """
        (objectives,observations)=generateSequence(T,numberSamples=numberSamples,generatorType=self._generatorType)
        
        return (objectives,observations)

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
