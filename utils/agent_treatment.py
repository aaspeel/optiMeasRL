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


def constructAgent(estimator,rewarder,objectives,measurements):
    """
    Return a new agent
    """
    rng=np.random.RandomState(123456)
    env=OptimalIntermittency(estimator, rewarder, objectives, measurements,rng)
    qnetwork=MyQNetwork(environment=env,random_state=rng)
    agent=NeuralAgent(env,qnetwork,random_state=rng)

    # load agent
    #agent.setNetwork('../../myFolder/myModels/retrainedAgent_KF')

    # --- Bind controllers to the agent ---
    # Before every training epoch, we want to print a summary of the agent's epsilon, discount and 
    # learning rate as well as the training epoch number.
    agent.attach(bc.VerboseController(modes=[]))

    # During training epochs, we want to train the agent after every action it takes.
    # Plus, we also want to display after each training episode (!= than after every training) the average bellman
    # residual and the average of the V values obtained during the last episode.
    agent.attach(bc.TrainerController())

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
    agent.attach(bc.InterleavedTestEpochController(
        id=0, # mode
        epoch_length=10,
        periodicity=1,
        show_score=True,
        summarize_every=1))
    
    return agent

def agentInference(agent, objectives_test, measurements_test):
    """
    Inference on the given data using the given agent. Return the corresponding results.
    """
    
    (numberSamples,T,_)=np.shape(objectives_test)
    
    # give test data to the environment (and erase the previous one)
    agent._environment.setTestData(objectives_test, measurements_test)
    
    # set the inference mode of the agent (and empties agent._tmp_dataset)
    agent.startMode(mode=1, epochLength=numberSamples*T)
    
    # run on the data
    agent.run(n_epochs=1, epoch_length=numberSamples*T)
    
    # take back the results of the running
    (testResults_sigmas, testResults_rewards, testResults_estimates)=agent._environment.getTestResults()
    
    # Back to training mode 
    agent.resumeTrainingMode()
    
    return (testResults_sigmas, testResults_rewards, testResults_estimates)
    