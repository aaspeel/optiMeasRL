"""
Various plots
"""

from matplotlib import pyplot
import numpy as np
import pandas as pd
import joypy

def plotRNNresults(history):
    pyplot.title('Mean squared error')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='valid')
    pyplot.legend()
    pyplot.show()

def plotExperiment(objectives,estimates,sigmas,idx_sample=0,idx_objective=0):
    scaledSigmas=sigmas[idx_sample,:] * max(objectives[idx_sample,:,idx_objective]) + (1-sigmas[idx_sample,:]) * min(objectives[idx_sample,:,idx_objective])
    
    # plot objective, estimate and sigma
    pyplot.title('Objective')
    pyplot.plot(estimates[idx_sample,:,idx_objective], marker='o', color='blue', label='estimate')
    pyplot.plot(objectives[idx_sample,:,idx_objective], marker='o', color='orange', label='real')
    pyplot.plot(scaledSigmas, marker='o', color='red', label='sigma')
    pyplot.xlabel('Time t')
    pyplot.ylabel('z_i')
    pyplot.legend()
    pyplot.show()
    
    # plot estimation error and sigma
    absError=abs(objectives-estimates)
    pyplot.title('Error')
    pyplot.plot(absError[idx_sample,:,idx_objective], marker='o', color='blue', label='abs error')
    pyplot.plot(max(absError[idx_sample,:,idx_objective])*sigmas[idx_sample,:], marker='o', color='red', label='sigma')
    pyplot.xlabel('Time t')
    pyplot.ylabel('Error_i')
    pyplot.legend()
    pyplot.show()
    
def boxplotErrors(objectives,estimates):
    T=np.shape(estimates)[1]
    print('T:',T)
    errors=objectives-estimates
    squareErrorNorms=np.square(errors).sum(axis=2)
    pyplot.title('Square error norm')
    pyplot.boxplot(squareErrorNorms,positions=range(0,T))
    pyplot.xlabel('Time t')
    pyplot.ylabel('Square error norm')
    #pyplot.legend()
    pyplot.show()
    
def boxplotRewards(rewards):
    T=np.shape(rewards)[1]
    pyplot.title('Reward')
    pyplot.boxplot(rewards,positions=range(0,T))
    pyplot.xlabel('Time t')
    pyplot.ylabel('Reward r(t)')
    #pyplot.legend()
    pyplot.show()
    
def boxplotCumulatedRewards(cumulatedRewards):
    cumulatedRewards=np.array(cumulatedRewards)
    pyplot.title('Cumulated reward')
    pyplot.boxplot(cumulatedRewards.transpose())
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Cumulated reward')
    #pyplot.legend()
    pyplot.show()
    
def boxplotSigmas(sigmas):
    (_,T)=np.shape(sigmas)
    pyplot.title('Sigma')
    pyplot.boxplot(sigmas,positions=range(0,T))
    pyplot.xlabel('Time t')
    pyplot.ylabel('Actions sigma(t)')
    #pyplot.legend()
    pyplot.show()
    
def freqSigmas(sigmas):
    (_,T)=np.shape(sigmas)
    freq=np.mean(sigmas,axis=0)
    pyplot.title('Frequency')
    pyplot.plot(range(0,T),freq,marker='o')
    pyplot.xlabel('Time t')
    pyplot.ylabel('sigma(t)')
    pyplot.show()
    
def plotAllErrors(objectives,estimates):
    errors=objectives-estimates
    squareErrorNorms=np.square(errors).sum(axis=2)
    MSEs=squareErrorNorms.mean(axis=0)
    pyplot.title('Square error norm')
    pyplot.plot(squareErrorNorms.transpose(), color='blue',alpha=0.2)
    pyplot.plot(MSEs,color='black',label='Mean')
    pyplot.xlabel('Time t')
    pyplot.ylabel('Square error norm')
    pyplot.legend()
    pyplot.show()
    
def plotAllRewards(rewards):
    meanRewards=rewards.mean(axis=0)
    pyplot.title('Reward r(t)')
    pyplot.plot(rewards.transpose(), color='blue',alpha=0.2)
    pyplot.plot(meanRewards,color='black',label='Mean')
    pyplot.xlabel('Time t')
    pyplot.ylabel('Reward r(t)')
    pyplot.legend()
    pyplot.show()
    
def plotAllCumulatedRewards(cumulatedRewards):
    cumulatedRewards=np.array(cumulatedRewards)
    (n_epochs,_)=cumulatedRewards.shape
    meanCumulatedRewards=cumulatedRewards.mean(axis=1)
    
    pyplot.title('Cumulated reward')
    #pyplot.plot(range(1,n_epochs+1),cumulatedRewards, color='blue',alpha=0.2)
    pyplot.plot(range(1,n_epochs+1),meanCumulatedRewards,color='black',label='Mean')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Cumulated reward')
    pyplot.legend()
    pyplot.show()

def histoCumulatedRewards(rewards):
    cumulatedRewards=rewards.sum(axis=1)
    pyplot.hist(cumulatedRewards,density=True, bins=50)
    pyplot.title('Cumulated rewards histogram')
    pyplot.xlabel('Cumulated rewards')
    pyplot.show()
    
def allHistoCumulatedRewards(cumulatedRewards,mod=5):
    def color_gradient(x=0.0, start=(0, 0, 0), stop=(1, 1, 1)):
        r = np.interp(x, [0, 1], [start[0], stop[0]])
        g = np.interp(x, [0, 1], [start[1], stop[1]])
        b = np.interp(x, [0, 1], [start[2], stop[2]])
        return (r, g, b)

    n_epochs=np.shape(cumulatedRewards)[0]

    df = pd.DataFrame()
    for epoch in range(n_epochs):
        df[epoch+1] = cumulatedRewards[epoch]
    labels=[y+1 if (y+1)%mod==0 or (y+1)==1 or (y+1)==n_epochs else None for y in range(n_epochs)]

    fig,axes=joypy.joyplot(df, overlap=1,kind="normalized_counts", bins=100, colormap=lambda x: color_gradient(x, start=(.78, .25, .09), stop=(1.0, .64, .44)), linecolor='w', linewidth=.5, title='Cumulated rewards', labels=labels, grid="y")
    ax=axes[-1]
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Epoch")
    ax.yaxis.set_visible(True)
    ax.yaxis.set_ticks([])
    ax.set_xlabel("Cumulated reward")
    
        
def sigma_to_points(sigmas: list):
    x = []
    y = []
    for i in range(len(sigmas)):
        if sigmas[i] == 1:
            x.append(i)
            y.append(1)
    return x,y

