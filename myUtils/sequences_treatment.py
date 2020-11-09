"""
To manipulate data sequences.
"""

import numpy as np
from matplotlib import pyplot

def generateSequence(T,numberSamples=1,n=1,m=1,generatorType='sinRandomFreq'):
    # outputs: x a numpy array of shape (numberSamples,T,n)
    #          y a numpy array of shape (numserSamples,T,m)
    
    if generatorType=='random01':
        t=np.random.random([numberSamples,T,1])
        x=np.repeat(t,n,axis=2)
        y=np.repeat(t,m,axis=2)
        
    elif generatorType=='sin':
        if (not n==1) or (not n==1):
            print('ERROR: m=n=1 is required when generatorType=sin')
            return
        
        # trajectories
        t=np.linspace(0,2*np.pi,T)
        x=(np.sin(t)+1)/2
        y=(np.sin(t)+1)/2
        
        # correct format
        x=x.reshape(1,T,1)
        y=y.reshape(1,T,1)
        
        # simulate many samples
        x=np.repeat(x,numberSamples,axis=0)
        y=np.repeat(y,numberSamples,axis=0)
        # add noise on y
        #y+=np.random.normal(size=[numberSamples,T,1])*0.2
        
    elif generatorType=='dynamicSystem':
        A=np.array([[1,0],[0,1]])
        C=np.array([[1,0],[0,1]])
        Q=np.array([[1,0],[0,1]])*0
        R=np.array([[1,0],[0,1]])*0
        
        (m,n)=np.shape(C) # To have correct reshape at the end of this function
        (x,y)=dynamicSequence(T,A,C,Q,R,numberSamples=numberSamples)
    
    elif generatorType=='sinRandomFreq':
        if (not n==1) or (not m==1):
            print('ERROR: m=n=1 is required when generatorType=sinRandomFreq')
            return
        # generates random sequence of frequencies
        p0=0.95
        p1=(1-p0)/2
        p2=(1-p0)/2
        gamma=np.random.choice([0,1,2], size=(numberSamples,T), p=[p0,p1,p2])
        gamma=np.cumsum(gamma,axis=1)
        gamma=np.mod(gamma,3)+1 # in {1,2,3}
        f=gamma*1

        t=np.arange(T)/(2*np.pi*10)
        x=(np.sin(2*np.pi*np.multiply(f,t))+1)/2
        y=(np.sin(2*np.pi*np.multiply(f,t))+1)/2
    
    elif generatorType=='constantPi':
        x=np.ones([numberSamples,T,n])*np.pi
        y=np.ones([numberSamples,T,m])*np.pi
    
    elif generatorType=='randomStep':
        if (not n==1) or (not m==1):
            print('ERROR: m=n=1 is required when generatorType=sinRandomFreq')
            return
        delta=4
        nStep=np.ceil(T/delta).astype(int)

        x=np.random.rand(numberSamples,nStep,1)
        x=np.repeat(x,delta,axis=1)
        x=x[:,:T,:]
        y=x.copy()
        
    else:
        print('ERROR: unknown generatorType\n')
        return
    
    if (y<-1).any():
        print('WARNING: y<-1. Clipping has been applied.')
        y=y.clip(min=-1)
        x=x.clip(min=-1)
    
    x=x.reshape(numberSamples,T,n)
    y=y.reshape(numberSamples,T,m)
    return x,y


def dynamicSequence(T,A,C,Q,R,numberSamples=1):
    # return a sequence of length T: x(t+1)=A*x(t)+w(t), w(t)~N(0,Q)
    #                                y(t)=C*x(t)+v(t), v(t)~N(0,R)
    
    #A=[[1,0],[0,1]]
    #C=[[1,0],[0,1]]
    #Q=[[1,0],[0,1]]
    #R=[[1,0],[0,1]]
    
    (m,n)=np.shape(C)
    x0=np.zeros((n,numberSamples))
    # construct x with shape (n,numberSamples,T). Will be transposed later
    x=np.zeros((n,numberSamples,T))
    x[:,:,0]=x0
    w=np.random.multivariate_normal(np.zeros(n),Q,size=(numberSamples,T)).transpose((2,0,1))
    for t in range(T-1):
        x[:,:,t+1]=np.matmul(A,x[:,:,t])+w[:,:,t]
        x[:,:,t+1]=x[:,:,t+1].clip(min=0) # TO REMOVE

    # construct y with shape (m,numberSamples,T). Will be transposed later
    v=np.random.multivariate_normal(np.zeros(m),R,size=(numberSamples,T)).transpose((2,0,1))
    y=np.tensordot(C,x,axes=([1],[0]))+v

    # change shapes, x: (numberSamples,T,n) and y: (numberSamples,T,m)
    x=x.transpose((1,2,0))
    y=y.transpose((1,2,0))
    
    if (y<-1).any():
        print('WARNING: y<-1. Clipping has been applied.')
        y=y.clip(min=-1)
        x=x.clip(min=-1)
    return x,y


def randomSigma(T,numberSamples=1,p0=1./2):
    # return a random binary signal of shape (numberSamples,T).
    # 0 has a probability p0 to occur, 1 has a probability (1-p0) to occur.
    sigma = np.random.choice([0, 1], size=(numberSamples,T), p=[p0, (1-p0)])
    return sigma

def corruptSequence_mask(observations,sigma):
    # transform sigma into a (numberSamples,T,m) array without changing sigma
    (numberSamples,T,n_dim_obs)=np.shape(observations)
    sigma2=sigma.reshape([numberSamples,T,1])
    sigma2=np.tile(sigma2,(1,1,n_dim_obs))
    observations_corrupted = np.ma.array(observations,mask=(1-sigma2))

    return observations_corrupted


def corruptSequence_outOfRange(y,sigma,outOfRangeValue=-1):
    # y has shape (numberSamples,T,m) and is a numpy array
    # sigma has shape (numberSaples,T) and is a binary numpy array
    # return yc, a corrupted version of y according to sigma
    if not np.array_equal(sigma, sigma.astype(bool)):
        print('ERROR in corruptSequence: sigma is not a binary sequence')
    
    (numberSamples,T,_)=np.shape(y)
    sigma=sigma.reshape([numberSamples,T,1]) # To permits element wise product
    yc=sigma*y + (1-sigma)*outOfRangeValue
    return yc


def plotRNNresults(history):
    pyplot.title('Mean squared error')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='valid')
    pyplot.legend()
    pyplot.show()
    return pyplot

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
    