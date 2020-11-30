"""
Old functions and code
"""

def corruptSequenceWithAgentPolicy(agent,observations):
    #  ! ! ! SLOW ! ! !
    #
    # corrupt observations according to the test policy of the agent
    # return sigma and observations_corrupted_outOfRange (outOfRangeValue of the agent)
    
    agentPolicy=agent._test_policy
    
    outOfRangeValue=agent._environment._outOfRangeValue
    sigmaMEMORY=agent._environment._sigmaMEMORY
    observationsMEMORY=agent._environment._observationsMEMORY
    
    (numberSamples,T,m)=np.shape(observations)
    
    sigma=np.zeros([numberSamples,T])
    observations_corrupted=observations.copy()
    for indSample,sample in enumerate(observations_corrupted):
        # initialize the state of the agent
        agentState=[sigmaMEMORY*[0],observationsMEMORY*[m*[outOfRangeValue]]]
        for t,obs_t in enumerate(sample):
            # compute action from agent state
            (action,_)=agentPolicy.bestAction(agentState)
            
            # for returning
            sigma[indSample,t]=action
            if not action: # no observation available
                observations_corrupted[indSample,t,:]=m*[outOfRangeValue]
            
            # observation of the agent
            agent_obs=[action,observations_corrupted[indSample,t,:]]
            for i in range(len(agent_obs)):
                agentState[i][0:-1] = agentState[i][1:]
                agentState[i][-1] = agent_obs[i]
            
    return sigma,observations_corrupted