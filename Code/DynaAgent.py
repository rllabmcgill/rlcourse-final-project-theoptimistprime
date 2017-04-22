# Author: Ayush Jain
import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from agent import Agent, LearningAgent
#from mcts import MCTS
from MCTSOptions_hashed import MCTSOption

class DYNAAgent(LearningAgent):
    name = "DYNA Agent"

    def __init__(self, environment, options, getFeatures, getState, featuresOn, K, noOfActions, 
        featureVectorLength=1000, behavior_policy=None, planning="Random"):
        super(DYNAAgent, self).__init__(environment, options, getFeatures, getState, featuresOn, 
            noOfActions, featureVectorLength, behavior_policy)
        """
        param K: number of times hypothetical experience is used in Planning Step

        """
        self.K = K
        self.planning = self.MCTSplanning if planning=='MCTS' else self.Randomplanning
        print "DynaAgent initiated with", planning, "planning"
        

    def randomAct(self, features):
        initializable_options = self.feasible_options(features)
        return np.random.choice(initializable_options)


    def MCTSplanning(self):
        # this is till termination of selected option
        env = self.env.__class__()
        i=0
        featureVector = self.getFeatures(env.observation_space.getState())

        mcts = MCTSOption(value_fn=self.valueFunction, policy_fn=self.policyFunction, rollout_policy_fn=self.fastPolicy,
            options=self.options, get_features=self.getFeatures, get_states=self.getState, expectedFeatures=self.expectedFutureFeatureVector,
            optionRewards=self.w_r, lmbda=0.5, c_puct=0.25, rollout_limit=10, playout_depth=5, n_playout=20)
        
        while not env.episodeEnded() and i<self.K :
            i+=1
            o = mcts.get_move(env)
            optionIndex = self.options.index(o)
            r = np.dot(self.w_r[optionIndex], featureVector)
            nextFeatureVector = self.expectedFutureFeatureVector(featureVector, optionIndex)
            self.QLearning_update(r, featureVector, nextFeatureVector, optionIndex)
            featureVector = nextFeatureVector
            env.reset()


    def Randomplanning(self):
        # this is till termination of selected option
        for epoch in range(self.K):
            featureVector = np.zeros(self.featureVectorLength).astype('int8')
            featureVector[np.random.randint(0,self.featureVectorLength, size=self.featuresOn)] = 1
            o = self.randomAct(featureVector)
            r = np.dot(self.w_r[o], featureVector)
            nextFeatureVector = self.expectedFutureFeatureVector(featureVector, o)       
            self.QLearning_update(r, featureVector, nextFeatureVector, o)

    def start(self, observation):
        self.reset()
        self.featureVector = self.getFeatures(observation)
        self.activeOption = self.mu()
        self.lastAction = self.activeOption.policy(self.featureVector)
        self.last_featureVector = self.featureVector
        return self.lastAction

    def step(self, observation, reward, done):
        # this one time step
        self.featureVector = self.getFeatures(observation)
        # self.model_update(reward)
        self.QLearning_update(reward, self.last_featureVector, self.featureVector)
        self.planning()
        return self.act(done)

    def end(self, observation, reward):
        # episode finished, this method is called to update the options with experience from
        # last step of the episode.
        self.featureVector = self.getFeatures(observation)
        # self.model_update(reward)
        self.QLearning_update(reward, self.last_featureVector, self.featureVector)
        self.planning()

