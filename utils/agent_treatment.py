"""
To manipulate RL agent (from deer).
"""

import numpy as np

# import environment class (used in constructAgent)
from optimalIntermittency import OptimalIntermittency

# import deer classes (used in constructAgent)
from deer.learning_algos.q_net_keras import MyQNetwork
from deer.agent import NeuralAgent
import deer.experiment.base_controllers as bc

# import personal controller
from utils.interleavedValidEpochController import InterleavedValidEpochController

from utils.agentNetwork import AgentNetwork

def constructAgent(estimator,rewarder,objectives_train,measurements_train,objectives_valid,measurements_valid):
    """
    Return a new agent
    """
    (numberSamples_valid,T_valid,_)=np.shape(objectives_valid)
    
    batch_size=10
    
    rng=np.random.RandomState(123456)
    env=OptimalIntermittency(estimator,rewarder,objectives_train,measurements_train,objectives_valid,measurements_valid,rng)
    qnetwork=MyQNetwork(environment=env, random_state=rng, batch_size=batch_size, neural_network=AgentNetwork)
    agent=NeuralAgent(env, qnetwork, batch_size=batch_size, random_state=rng)
    
    #agent.setDiscountFactor(1.0)

    # load agent
    #agent.setNetwork('../../myFolder/myModels/retrainedAgent_KF')

    # --- Bind controllers to the agent ---
    # Before every training epoch, we want to print a summary of the agent's epsilon, discount and 
    # learning rate as well as the training epoch number.
    agent.attach(bc.VerboseController(modes=[]))

    # During training epochs, we want to train the agent after every action it takes.
    # Plus, we also want to display after each training episode (!= than after every training) the average bellman
    # residual and the average of the V values obtained during the last episode.
    agent.attach(bc.TrainerController(
        periodicity=1,
        evaluate_on='action',
        show_episode_avg_V_value=False, # show V value
        show_avg_Bellman_residual=False)) # show average training loss

    # All previous controllers control the agent during the epochs it goes through. However, we want to interleave a 
    # "test epoch" between each training epoch ("one of two epochs", hence the periodicity=2). We do not want these 
    # test epoch to interfere with the training of the agent, which is well established by the TrainerController, 
    # EpsilonController and alike. Therefore, we will disable these controllers for the whole duration of the test 
    # epochs interleaved this way, using the controllersToDisable argument of the InterleavedTestEpochController. 
    # The value of this argument is a list of the indexes of all controllers to disable, their index reflecting in 
    # which order they were added. Here, "0" is refering to the firstly attached controller, thus the 
    # VerboseController; "2" refers to the thirdly attached controller, thus the LearningRateController; etc. The order 
    # in which the indexes are listed is not important.
    # For each test epoch, we want also to display the sum of all rewards obtained, hence the showScore=True.
    # Finally, we want to call the summarizePerformance method of Toy_Env every [summarize_every] *test* epochs.
    agent.attach(InterleavedValidEpochController(
        id=0, # mode
        epoch_length=numberSamples_valid*T_valid,
        periodicity=1, #  “1 test epoch on [periodicity] epochs”. Minimum value: 2.
        show_score=True, # True
        summarize_every=1))
    
    return agent

def agentInference(agent, objectives, measurements):
    """
    Inference on the given data using the given agent. Return the corresponding results.
    """
    
    (numberSamples,T,_)=np.shape(objectives)
    
    # give test data to the environment (and erase the previous one)
    agent._environment.setTestData(objectives, measurements)
    
    # set the inference mode of the agent (and empties agent._tmp_dataset)
    agent.startMode(mode=1, epochLength=numberSamples*T)
    
    # run on the data
    agent.run(n_epochs=1, epoch_length=numberSamples*T)
    
    # take back the results of the running
    (sigmas, rewards, estimates)=agent._environment.getTestResults()
    
    # Back to training mode 
    agent.resumeTrainingMode()
    
    return (sigmas, rewards, estimates)

def agentForcedInference(agent, sigmas):
    (numberSamples,T,_)=agent._environment._objectives_test.shape # throw an error if setTestData has not been called before.
    
    # give regular sigma to the agent
    agent._environment.setForcedActions(sigmas)

    # set the forced mode of the agent (and empties agent._tmp_dataset)
    agent.startMode(mode=2, epochLength=numberSamples*T)

    # run on the test data
    agent.run(n_epochs=1, epoch_length=numberSamples*T)

    # take back the results of the running
    (sigmas_copy, rewards, estimates)=agent._environment.getTestResults()

    # Back to training mode 
    agent.resumeTrainingMode()
    
    if np.linalg.norm(sigmas-sigmas_copy)!=0:
        print("Warning: a problem happend in agent_treatment.py/agentForcedInference().")
    
    return (rewards, estimates)






    