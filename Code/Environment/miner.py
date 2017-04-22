import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import random
from options import *
from agent import *
from DynaAgent import *
import pickle

action_space = type('', (object,), {})()
action_space.n = 4
options = []#makePrimitiveOptions(action_space)

policy_behaviour = np.array([[0.25, 0.25, 0.25, 0.25],
                            [0.25, 0.25, 0.25, 0.25],
                            [0.25, 0.25, 0.25, 0.25],
                            [0.4, 0.2, 0.2, 0.2],
                            [0.4, 0.2, 0.2, 0.2],
                            [0.4, 0.2, 0.2, 0.2],
                            [0.4, 0.2, 0.2, 0.2],
                            [0.4, 0.2, 0.2, 0.2],
                            [0.2, 0.2, 0.4, 0.2],
                            [0.2, 0.2, 0.4, 0.2],
                            [0.2, 0.2, 0.4, 0.2],
                            [0.2, 0.2, 0.4, 0.2]])
behaviour_option = MarkovOptions("behaviour", [True for i in range(12)], policy_behaviour, np.array([0.2 for i in range(12)]))

policy_uniform = policy_behaviour.copy()
policy_uniform[:,:] = .25
options.append(MarkovOptions("uniform", [True for i in range(12)], policy_uniform, np.array([0.2 for i in range(12)])))

policy_headfirst = policy_uniform.copy()
policy_headfirst[:3,0] = 0.9
policy_headfirst[:3,1:] = 0.1/3.0
policy_headfirst[6:8,0] = 0.9
policy_headfirst[6:8,1:] = 0.1/3.0
options.append(MarkovOptions("headfirst", [True for i in range(12)], policy_headfirst, np.array([0.2 for i in range(12)])))

policy_cautious = np.array([[0.133, 0.133, 0.134, 0.6],
                           [0.133, 0.133, 0.134, 0.6],
                           [0.133, 0.133, 0.134, 0.6],
                           [0.6, 0.133, 0.133, 0.134],
                           [0.6, 0.133, 0.133, 0.134],
                           [0.6, 0.133, 0.133, 0.134],
                           [0.6, 0.133, 0.133, 0.134],
                           [0.6, 0.133, 0.133, 0.134],
                           [0.133, 0.133, 0.6, 0.134],
                           [0.133, 0.133, 0.6, 0.134],
                           [0.133, 0.133, 0.6, 0.134],
                           [0.133, 0.133, 0.6, 0.134]])
options.append(MarkovOptions("cautious", [True for i in range(12)], policy_cautious, np.array([0.2 for i in range(12)])))




class Environment(object):
    class observations(object):
        def sample(self):
            return np.random.randint(0,12,1)
    
    def __init__(self, trans, maxEntrapments):
        self.state = 0
        self.trans = trans
        self.trap_count = 0
        self.gold = 0
        self.maxEntrapments = maxEntrapments
        self.observation_space = self.observations()
        self.reset()

    def step(self, action):
        self.state = self.trans[self.state,action]
        r = 1 if self.state==11 else 0
        return self.state, r
    
    def getState(self):
        return self.state
    
    def reset(self):
        self.state=0
        self.time=0
        return self.state
        
    def trapper(self):
        if self.time<=0 and np.random.random()<0.25 :
            self.trap = np.random.randint(6,8)
            self.time = 2
        else:
            self.time-=1
    
    def ticktock(self):
        self.trapper()        
        if self.state == 11:
            self.gold+=1
            return True
        elif self.time>0 and self.state == self.trap:
            #goldTrace[self.trap_count]+=self.gold
            self.trap_count+= 1
            #state_sequence.append("Trapped in {0}".format(self.state))
            return True
        return False
    
    def stop(self):
        return True if self.trap_count>=self.maxEntrapments else False




iState = np.array([1,1,1,0,0,0,0,0,1,1,1,1])
#state_sequence = []
#goldTrace = np.zeros((MaxTraps))


# This is the transition matrix delta. At state s and action a, trans(s,a) = s' = next state
trans = np.array([[6,0,0,1],
              [1,1,0,2],
              [2,2,1,3],
              [4,3,2,3],
              [5,3,4,4],
              [8,4,5,5],
              [7,0,6,6],
              [11,6,7,7],
              [8,5,9,8],
              [9,9,10,8],
              [10,10,11,9],
              [11,11,11,11]]).astype("int8")

env = Environment(trans, 5000)

def getFeatures(observation):
    features = np.zeros(12).astype('int8')
    features[observation]=1
    return features

def getState(features):
    return np.flatnonzero(features)[0]

def policyEvaluate(policy, maxEntrapments, iterations):
    goldTrace = []
    for i in range(iterations):
        env = Environment(trans, maxEntrapments)
        w_q = np.load('w_q_miner.npy')[policy]
        while (not env.stop()):
            s = env.reset()
            feature = getFeatures(s)
            action = np.dot(w_q,features)
            s,r = env.step(agent.start(s))
            done = False
            while(not done):
                done = env.ticktock()
                action = agent.step(s,r,done)
                s,r = env.step(action)


agent = DYNAAgent(env, options, getFeatures, getState, 
        featuresOn=1, K=5, maxStepsInPlanning=1, featureVectorLength=12, behavior_policy=behaviour_option)     

if os.path.isfile('Data/Miner/w_r.npy'):
    agent.set_weights(np.load('Data/Miner/w_r.npy'), np.load('Data/Miner/w_p.npy'), np.load('Data/Miner/w_q.npy'))
gold=[]
if os.path.isfile('Data/Miner/gold'):
        with open('Data/Miner/gold','rb') as f:
            gold = pickle.load(f)


while (not env.stop()):
    s = env.reset()
    s,r = env.step(agent.start(s))
    done = env.ticktock()
    while(not done):
        action = agent.step(s,r,done)
        s,r = env.step(action)
        done = env.ticktock()
    agent.end(s,r)
    gold.append(env.gold)

    agent.end(s,r)
    if env.gold%5==0:
        w_r, w_p, w_q = agent.get_weights()
        np.save('Data/Miner/w_r', w_r)
        np.save('Data/Miner/w_p', w_p)
        np.save('Data/Miner/w_q', w_q)

w_r, w_p, w_q = agent.get_weights()
np.save('Data/Miner/w_r', w_r)
np.save('Data/Miner/w_p', w_p)
np.save('Data/Miner/w_q', w_q)
with open('Data/Miner/gold','wb') as f:
        pickle.dump(gold, f)
