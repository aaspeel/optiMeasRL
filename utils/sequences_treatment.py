"""
To manipulate data sequences.
"""

import numpy as np
from matplotlib import pyplot
from .linear_systems import loadKF,sampleKFSequence

def generateSequence(T,numberSamples=1,generatorType='sinRandomFreq'):
    # outputs: x a numpy array of shape (numberSamples,T,n) - The quantity to estimate
    #          y a numpy array of shape (numserSamples,T,m) - The measurement
    
    if generatorType=='random01':
        n=1
        m=1
        t=np.random.random([numberSamples,T,1])
        x=np.repeat(t,n,axis=2)
        y=np.repeat(t,m,axis=2)
        
    elif generatorType=='linearSystem':
        kf=loadKF()
        (x,y,_)=sampleKFSequence(kf,T,numberSamples=numberSamples)
        x=x.data # convert the masked array into an array
        
    elif generatorType=='sin':
        n=1
        m=1
        
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
        
    elif generatorType=='sinRandomFreq':
        n=1
        m=1
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
        n=1
        m=1
        x=np.ones([numberSamples,T,n])*np.pi
        y=np.ones([numberSamples,T,m])*np.pi
    
    elif generatorType=='randomStep':
        delta=4
        nStep=np.ceil(T/delta).astype(int)

        x=np.random.rand(numberSamples,nStep,1)
        x=np.repeat(x,delta,axis=1)
        x=x[:,:T,:]
        y=x.copy()
        
    else:
        print('ERROR: unknown generatorType\n')
        return
    
    return x,y

def randomSigma(T,numberSamples=1,p0=1./2):
    # return a random binary signal of shape (numberSamples,T).
    # 0 has a probability p0 to occur, 1 has a probability (1-p0) to occur.
    sigma = np.random.choice([0, 1], size=(numberSamples,T), p=[p0, (1-p0)])
    return sigma

def regularSigma(T,numberMeasurements,numberSamples=1):
    """
    Return a sigma sequence of shape (numberSamples,T) with numberMeasurements 1 regularly spaced in each sample
    """
    sigma=np.zeros((numberSamples,T),int)
    cols=np.array(range(numberMeasurements))
    cols=np.round(cols*(T-1)/(numberMeasurements-1)).astype(int)
    sigma[:,cols]=1
    return sigma

def corruptSequence_mask(measurements,sigma):
    # transform sigma into a (numberSamples,T,m) array without changing sigma
    (numberSamples,T,n_dim_meas)=np.shape(measurements)
    sigma2=sigma.reshape([numberSamples,T,1])
    sigma2=np.tile(sigma2,(1,1,n_dim_meas))
    measurements_corrupted = np.ma.array(measurements,mask=(1-sigma2))

    return measurements_corrupted

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

