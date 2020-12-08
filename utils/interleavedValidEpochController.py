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

        
    def onStart(self, agent):
        if (self._active == False) or (agent.mode() not in self._modes):
            return

        self._epoch_count = 0
        self.rewards = []
        
        
    def onEpisodeEnd(self, agent, is_terminal, lastReward):
        if (self._currentlyValidating == False):
            return
        
        rewardEpisode = agent._total_mode_reward - sum(self._rewardsCurrentEpoch)
        self._rewardsCurrentEpoch.append(rewardEpisode.copy()) # store the content and not the pointer
        #print('In interleavedValidEpochController.onEpisodeEnd:\n self._epoch_count=', self._epoch_count, 'agent._mode=', agent._mode)
        

    def onEpochEnd(self, agent):
        if (self._active == False) or (agent.mode() not in self._modes) or (self._currentlyValidating == True):
            return

        mod = self._epoch_count % self._periodicity
        self._epoch_count += 1
        if mod == 0:
            #print('In myController.onEpochEnd:\n   self._epoch_count=',self._epoch_count, 'agent._mode=',agent._mode,'\n')
            self._currentlyValidating = True
            self._rewardsCurrentEpoch = []
            agent.startMode(self._id, self._epoch_length) # agent._total_mode_reward is reset to 0
            agent._run_non_train(n_epochs=1, epoch_length=self._epoch_length)
            self._summary_counter += 1

            if self._show_score:
                score,nbr_episodes=agent.totalRewardOverLastTest()
                print("Testing score per episode (id: {}) is {} (average over {} episode(s))".format(self._id, score, nbr_episodes))
                self.scores.append(score)
            if self._summary_periodicity > 0 and self._summary_counter % self._summary_periodicity == 0:
                agent.summarizeTestPerformance()
            agent.resumeTrainingMode()
            self.rewards.append(self._rewardsCurrentEpoch)
            self._currentlyValidating = False
            

