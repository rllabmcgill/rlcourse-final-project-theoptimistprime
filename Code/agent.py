# Author: Ayush Jain

import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# np.random.seed(seed=1581990)

class Agent(object):
    name = "Random Agent"

    def __init__(self, environment, options):
        self.env = environment
        #self.action_space = self.env.action_space
        self.init_parameters(options)
        self.noOfOptions = len(self.options)
        self.activeOption = None

    def init_parameters(self, options):
        """Initialize algorithm parameters. Will be called by constructor, and at the
        start of each new run. Parameters' initial values should be stored in
        self.params, and here instances of them should be copied into object variables
        which may or may not change during a particular run of the algorithm.
        """
        self.options = options #makePrimitiveOptions(self.action_space)
        pass

    def rho(self, state, action):
        """ Calculates the importance sampling ratio.
        param state: representation of current state (phi)
        param action: action selected according to behavior policy
        """
        pi = self.target_policy(state, action)
        mu = self.behavior_policy(state, action)

        if pi>0 and mu<=0:
            raise ZeroDivisionError(" Value of behavior policy can't be 0 where target policy is non-zero. Against assumption.")
        else:
            return pi/mu

    def i(self, state=None):
        """ Interest Function: returns a non-negetive relative interest in each state.
        param state: current state
        TO-DO: implement a function approximator to this
        """
        return 1

    def gamma(self, state=None):
        """ Termination function: returns thee probability of continuing from 'state'
        TO-DO: implement some kind of function to use this effectively
        """
        return 0.9

    def etd_lambda(self, state=None):
        """ Bootstraping in current 'state'  """
        return 0.8  


    def feasible_options(self, observation):
        """
        :returns: a subset of the options for which I_o(s) = True
        :rtype: list of int
        """
        return [idx for idx in xrange(self.noOfOptions) if self.options[idx].initiate(observation)]

    def reset(self):
        self.activeOption = None


    def act(self, observation, reward, done):
        if self.activeOption==None or self.activeOption.terminate(observation):
            availableOptions = self.feasible_options(observation)
            self.activeOption = np.random.choice(availableOptions)
        self.lastObservation = observation
        self.lastAction = self.activeOption.policy(observation)
        return self.lastAction


    def has_diverged(self):
        """Overwrite the function with one that checks the key values for your
        agent, and returns True if they have diverged (gone to nan or infty for ex.), and
        returns False otherwise.
        """
        return False



class LearningAgent(Agent):
    name = "ETD Agent"
    def __init__(self, environment, options, getFeatures, getState, featuresOn, noOfActions,
        featureVectorLength=1000, behavior_policy=None):
        super(LearningAgent, self).__init__(environment, options)
        """
        self.w_r: model weights to predict expected return taking an option 'o' in state 's'
        self.w_p: predict probability of ending in state 'x' when started in state 's' under option 'o'
        """
        self.getFeatures = getFeatures #this function is expected to return tile coded features
        self.getState = getState #function converts expected feature vector to state
        self.featureVectorLength = featureVectorLength
        self.noOfActions = noOfActions
        self.featuresOn = featuresOn
        self.w_r = 0.5*np.random.random((self.noOfOptions, self.featureVectorLength)).astype("float32")
        self.w_p = np.random.random((self.noOfOptions, self.featureVectorLength, 
            self.featureVectorLength)).astype("float32")
        # making sums less than one, since we'll be summing up for featuresOn number of features
        self.w_p /= self.featuresOn
        self.w_q = 0.5*np.random.random((self.noOfOptions, self.featureVectorLength, self.noOfActions)).astype("float32")
        self.e = np.zeros((self.noOfOptions, featureVectorLength, self.noOfActions), dtype="float32")
        self.f = np.zeros((self.noOfOptions), dtype="float32")
        self.epsilon = 0.15
        self.alpha_model = 0.01 
        self.alpha_q = 0.0001 
        self.behavior_option = behavior_policy


    def set_weights(self, w_r, w_p, w_q):
        self.w_r = w_r
        self.w_p = w_p
        self.w_q = w_q

    def get_weights(self):
        return self.w_r , self.w_p, self.w_q

    def reset(self):
        self.activeOption = None
        self.e[:] = 0.
        self.f[:] = 0.

    def fastPolicy(self, featureVector):
        """
        Function computes the distribution over feasible options based on probability of
        being in a state. Uses this distribution to pick an option.
        """ 
        V = [np.sum(np.dot(featureVector,self.w_q[o]*self.options[o].pi)) for o in self.feasible_options(featureVector)]
        return V.argmax()

    def policyVector(self, featureVector):
        """
        Function returns a soft probability vector over all options based on their values for given feature
        """
        temperature = 0.1
        V = np.array([np.sum(np.dot(featureVector, self.w_q[o]*self.options[o].pi)) for o in self.feasible_options(featureVector)])
        probability = np.exp(V/temperature)
        return probability/np.sum(probability)

    def policyFunction(self, featureVector):
        temperature = 0.1
        V = np.array([np.sum(np.dot(featureVector, self.w_q[o]*self.options[o].pi)) for o in self.feasible_options(featureVector)])
        prob = np.exp(V/temperature)
        prob/= np.sum(prob)
        return [(self.options[o],p) for o,p in zip(initializable_options, prob)]

    def egreedy(self, featureVector):
        if np.random.random() < self.epsilon:
            #self.e[:] = 0.
            #self.f[:] = 0.
            return np.random.randint(len(self.options))
        V = np.array([np.sum(np.dot(featureVector, self.w_q[o]*self.options[o].pi)) for o in self.feasible_options(featureVector)])
        return np.argmax(V)
        #return initializable_options[ np.dot(self.w_q[initializable_options,:],self.featureVector).argmax() ]

    def mu(self):
        self.activeOption = self.behavior_option
        ''' If active_option is None,
               agent was reset or
               activeOption terminate,
            choose an option greedily '''
        if self.activeOption == None or self.activeOption.terminate(self.featureVector):
            self.activeOption = self.options[self.egreedy(self.featureVector)]
            #print self.activeOption, observation
        return self.activeOption

    def valueFunction(self, featureVector):
        initializable_options = self.feasible_options(featureVector)
        return [np.sum(np.dot(featureVector, self.w_q[o]*self.options[o].pi)) for o in initializable_options].max()

    def consistent_options(self):
        return [idx for idx in xrange(self.noOfOptions) if self.options[idx].policy(self.last_featureVector)==self.lastAction]

    def expectedFutureFeatureVector(self, featureVector, option):
        temperature = 0.01
        dist = np.exp(np.dot(featureVector, self.w_p[option])/temperature)
        return dist/np.sum(dist)

    def model_update_old(self, reward):
        #reward = -100 if done else 0
        lastObservation = self.getState(self.last_featureVector)
        observation = self.getState(self.featureVector)
        for i in self.feasible_options(self.last_featureVector):
            rho = self.options[i].tran(lastObservation, self.lastAction) / self.activeOption.tran(lastObservation, self.lastAction)
            if rho==0:
                 continue
            beta = self.options[i].beta(observation)       
            # reward model
            # again, featureVector has been updated here!! Chill!!
            delta_reward = reward + self.gamma()*(1-beta)*np.dot(self.w_r[i,:], self.featureVector) - np.dot(self.w_r[i,:], self.last_featureVector)
            self.w_r[i,:] += self.alpha_model*delta_reward*self.e*rho
            
            # transition model for options
            last_OnFeatures = np.nonzero(self.last_featureVector)
            delta_prob = self.gamma()*((1-beta)*np.dot(self.featureVector, self.w_p[i]) + beta*self.featureVector)
            delta_prob -= np.dot(self.last_featureVector, self.w_p[i])
            self.w_p[i,last_OnFeatures,:] += self.alpha_model*delta_prob*self.e*rho


    def model_update(self, reward):
        beta = self.activeOption.beta(self.featureVector)
        i = self.options.index(self.activeOption)
        # reward model
        delta_reward = reward + self.gamma()*(1-beta)*np.dot(self.w_r[i], self.featureVector) - np.dot(self.w_r[i], self.last_featureVector)
        self.w_r[i] += self.alpha_model*self.last_featureVector*delta_reward
        # transition model for options
        delta_prob = self.gamma()*((1-beta)*np.dot(self.featureVector, self.w_p[i]) + beta*self.featureVector)
        delta_prob -= np.dot(self.last_featureVector, self.w_p[i])
        self.w_p[i] += self.alpha_model*np.outer(self.last_featureVector, delta_prob)


    def QLearning_update(self, reward, lastFeatureVector, futureFeatureVector=None, option=None):
        '''
        param reward is cumulated discounted reward received between selection of option 'o' in state s
        and termination 'k' timesteps later
        param observation
        '''
        if option==None:    #Direct RL update
            beta = np.array( [np.dot(o.b,futureFeatureVector) for o in self.options] )
            pi_o = np.outer(beta, self.policyVector(futureFeatureVector))
            for i,b in enumerate(beta):
                pi_o[i,i] = 1-b
            exp_future_val = [np.sum(np.dot(futureFeatureVector, self.w_q[i]*o.pi)) for i,o in enumerate(self.options)]
            curr_val = np.dot(self.w_q[:,:,self.lastAction], lastFeatureVector)
            delta_q = reward + self.gamma()*np.dot(pi_o, exp_future_val) - curr_val
            self.w_q += self.alpha_q*delta_q[:,None,None]*self.e

        else:               #Planning Update
            beta = np.dot(self.options[option].b, futureFeatureVector)
            pi_o = beta*self.policyVector(futureFeatureVector)
            pi_o[option] = 1-beta
            exp_future_val = [np.sum(np.dot(futureFeatureVector, self.w_q[i]*o.pi)) for i,o in enumerate(self.options)]
            exp_curr_val = np.sum(np.dot(lastFeatureVector, self.w_q[option]*self.options[option].pi))
            delta_q = reward + self.gamma()*np.dot(pi_o, exp_future_val) - exp_curr_val
            lastFeatureVector = lastFeatureVector.reshape(lastFeatureVector.shape[0],1)
            self.w_q[option] += self.alpha_q*delta_q*lastFeatureVector*self.options[option].pi
            # eligibility trace still retains the original value from episode, observation in planning step is randomly generated.

    def updateTrace(self):
        beta = np.dot(self.featureVector, self.previous_option.b)
        newOptionIndex = self.options.index(self.activeOption)
        if str(self.previous_option)==str(self.activeOption):
            pi_o = 1-beta
        else:
            pi_o = beta*self.policyVector(self.featureVector)[newOptionIndex]

        self.e *= self.gamma() * self.etd_lambda() * pi_o * self.activeOption.tran(self.featureVector, self.lastAction)
        self.e[newOptionIndex, np.nonzero(self.featureVector), self.lastAction] = 1

    def act(self, done):
        if done==True:
            self.reset()

        self.previous_option = self.activeOption
        if self.activeOption==None or self.activeOption.terminate(self.featureVector):
            '''if self.activeOption != None:
                print self.activeOption," ended in state ",observation'''
            self.activeOption = self.mu()

        self.lastAction = self.activeOption.policy(self.featureVector)
        self.last_featureVector = self.featureVector
        self.updateTrace()

        return self.lastAction

    def start(self, observation):
        self.reset()
        self.featureVector = self.getFeatures(observation)
        self.activeOption = self.mu()
        self.lastAction = self.activeOption.policy(self.featureVector)
        self.last_featureVector = self.featureVector
        return self.lastAction

    def step(self, observation, reward, done):
        self.featureVector = self.getFeatures(observation)
        self.model_update(reward)
        self.QLearning_update(reward, self.last_featureVector, self.featureVector)
        return self.act(done)

    def end(self, observation, reward):
        # episode finished, this method is called to update the options with experience from
        # last step of the episode.
        self.featureVector = self.getFeatures(observation)
        self.model_update(reward)
        self.QLearning_update(reward, self.last_featureVector, self.featureVector)

    def has_diverged(self):
        value = self.weights.sum()
        return numpy.isnan(value) or numpy.isinf(value)