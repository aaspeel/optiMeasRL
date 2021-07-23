from deer.experiment.base_controllers import Controller

class InterleavedValidEpochController(Controller):
    """A controller that interleaves a valid/test epoch between training epochs of the agent (only in training mode, i.e., agent.mode() == -1).
    
    Parameters
    ----------
    id : int
        The identifier (>= 0) of the mode each test epoch triggered by this controller will belong to. 
        Can be used to discriminate between datasets in your Environment subclass (this is the argument that 
        will be given to your environment's reset() method when starting the test epoch).
    epoch_length : float
        The total number of transitions that will occur during a test epoch. This means that
        this epoch could feature several episodes if a terminal transition is reached before this budget is 
        exhausted.
    periodicity : int 
        How many train epochs are necessary before a valid/test epoch is ran.
    show_score : bool
        Whether to print an informative message on stdout at the end of each test epoch, about 
        the total reward obtained in the course of the test epoch.
    summarize_every : int
        How many of this controller's test epochs are necessary before the attached agent's 
        summarizeTestPerformance() method is called. Give a value <= 0 for "never". If > 0, the first call will
        occur just after the first test epoch.
    """

    def __init__(self, id=0, epoch_length=500, periodicity=1, show_score=True, summarize_every=10):
        """Initializer.
        """

        super(self.__class__, self).__init__()
        self._epoch_count = 0
        self._id = id
        self._epoch_length = epoch_length
        self._show_score = show_score
        self._periodicity = periodicity
        self._summary_counter = 0
        self._summary_periodicity = summarize_every
        self._currentlyValidating = False
        self.scores = []
        self.rewards = [] # cumulated rewards
        
        # by default, self._modes = [-1]
        
        
    def onStart(self, agent):
        #print('onStart: call')
        if (self._active == False) or (agent.mode() not in self._modes):
            #print(' onStart: quit directly')
            #if (self._active == False):
                #print('     inactive')
            #if (agent.mode() not in self._modes):
                #print('     not in modes')
            return
        #print(' onStart: in')

        self._epoch_count = 0
        self.rewards = []
        
        #print(' onStart: out')
        
        
    def onEpisodeEnd(self, agent, is_terminal, lastReward):
        #print('onEpisodeEnd: call')
        if (self._currentlyValidating == False):
            #print(' onEpisodeEnd: quit directly (not validating)')
            return
        #print(' onEpisodeEnd: in (currentlyValidating=',self._currentlyValidating,')')
        
        rewardEpisode = agent._total_mode_reward - sum(self._rewardsCurrentEpoch)
        self._rewardsCurrentEpoch.append(rewardEpisode.copy()) # store the content and not the pointer
        #print('In interleavedValidEpochController.onEpisodeEnd:\n self._epoch_count=', self._epoch_count, 'agent._mode=', agent._mode)
        #print('          rewardEpisode=',rewardEpisode)
        
        #print(' onEpisodeEnd: out')
        

    def onEpochEnd(self, agent):
        #print('onEpochEnd: call')
        if (self._active == False) or (agent.mode() not in self._modes) or (self._currentlyValidating == True):
            #print(' onEpochEnd: quit directly')
            #if (self._active == False):
                #print('     inactive')
            #if (agent.mode() not in self._modes):
                #print('     not in _modes')
            #if (self._currentlyValidating == True):
                #print('     currently validating')
            return
        #print(' onEpochEnd: in (epoch count=',self._epoch_count,')')

        mod = self._epoch_count % self._periodicity
        self._epoch_count += 1
        if mod == 0:
            #print(' modulo: in')
            #print('In myController.onEpochEnd:\n   self._epoch_count=',self._epoch_count, 'agent._mode=',agent._mode,'\n')
            self._currentlyValidating = True
            self._rewardsCurrentEpoch = []
            agent.startMode(self._id, self._epoch_length) # agent._total_mode_reward is reset to 0
            #print()
            #print('-------------------------')
            print("Validation epoch running... ", end="")
            agent._run_non_train(n_epochs=1, epoch_length=self._epoch_length)
            print('Done.')
            #print('-------------------------')
            #print()
            self._summary_counter += 1

            if self._show_score:
                #print('   show_score: in')
                score,nbr_episodes=agent.totalRewardOverLastTest()
                print("  Testing score per episode (id: {}) is {} (average over {} episode(s))".format(self._id, score, nbr_episodes))
                self.scores.append(score)
            if self._summary_periodicity > 0 and self._summary_counter % self._summary_periodicity == 0:
                #print('   summarize: in')
                agent.summarizeTestPerformance()
            agent.resumeTrainingMode()
            self.rewards.append(self._rewardsCurrentEpoch)
            self._currentlyValidating = False
            
        #print(' onEpochEnd: out (agent._mode:',agent._mode,')')
            

