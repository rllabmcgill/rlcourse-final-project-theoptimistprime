import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from options import *
from agent import *
from DynaAgent import *
import pickle
                                                       
"""  
______________________
|RRR  1 |  2   3  GGG|
|RRR    |         GGG|
| 5   6 |  7   8   9 |
|       |            |
|10  11   12  13  14 |
|                    |
|15 |16   17 |18  19 |
|   |        |       |
|YYY|21   22 |BBB 24 |
|YYY|________|BBB____|


Actions: Up, Down, Left, Right, Pickup, Drop
"""


RED = 0
GREEN = 1
YELLOW = 2
BLUE = 3
CAR = 4

location = {RED:0, GREEN:4, YELLOW:20, BLUE:23}


trans = np.array([[0,5,0,1],
              [1,6,0,1],
              [2,7,2,3],
              [3,8,2,4],
              [4,9,3,4],
              [0,10,5,6],
              [1,11,5,6],
              [2,12,7,8],
              [3,13,7,9],
              [4,14,8,9],
              [5,15,10,11],
              [6,16,10,12],
              [7,17,11,13],
              [8,18,12,14],
              [9,19,13,14],
              [10,20,15,15],
              [11,21,16,17],
              [12,22,16,17],
              [13,23,18,19],
              [14,24,18,19],
              [15,20,20,20],
              [16,21,21,22],
              [17,22,21,22],
              [18,23,23,24],
              [19,24,23,24]]).astype("int8")


class Taxi(object):
    class observations(object):
        def __init__(self):
            self.car = np.random.randint(25)
            self.passenger = np.random.choice([RED, GREEN, YELLOW, BLUE])
            self.destination = np.random.choice([RED, GREEN, YELLOW, BLUE])
        
        def reset(self):
            self.car = np.random.randint(25)
            self.passenger = np.random.choice([RED, GREEN, YELLOW, BLUE])
            self.destination = np.random.choice([RED, GREEN, YELLOW, BLUE])
            return self.car, self.passenger, self.destination    
        
        def getState(self):
            return self.car, self.passenger, self.destination
            
        def setState(self, features):
            onIndex = np.flatnonzero(features)[0]
            self.car = onIndex/20
            onIndex %= 20
            self.passenger = onIndex/4
            onIndex %= 4
            self.destination = onIndex 

        def sample(self):
            return self.reset()

    def __init__(self):
        self.trans = trans
        self.observation_space = self.observations()
        self.nActions = 6
        self.acc_reward = 0
        self.done = False
        self.reset()

    def reset(self):
        self.acc_reward = 0
        self.done = False
        return self.observation_space.reset()

    def totalReward(self):
        return self.acc_reward

    def goalState(self):
        if self.observation_space.passenger == CAR:
            return location[self.observation_space.destination]
        else:
            return location[self.observation_space.passenger]

    def step(self, action):
        reward = -1
        if action<4:
            self.observation_space.car = self.trans[self.observation_space.car,action]
            """if self.picked:
                self.observation_space.passenger = self.observation_space.car"""
        elif action == 4:
            if self.observation_space.passenger!=CAR and self.observation_space.car == location[self.observation_space.passenger]:
                reward = 0
                self.observation_space.passenger = CAR
        elif action == 5:
            if self.observation_space.car==location[self.observation_space.destination] and self.observation_space.passenger==CAR:
                reward = 1
                self.done = True
        self.acc_reward += reward
        return self.observation_space.getState(), reward, self.done

    def episodeEnded(self):
        return self.done


def getFeatures(observation):
    car, passenger, destination = observation
    features = np.zeros(500).astype('int8')
    index = destination + passenger*4 + car*5*4
    features[index]=1
    return features

def getState(features):
    stateDistribution = np.zeros(25)
    for index, probability in enumerate(features):
        stateDistribution[index/20] += probability
    onIndex = np.flatnonzero(stateDistribution)
    if len(onIndex)>1:
        return stateDistribution
    return onIndex[0] 


def learnOptionPolicy(destination, possibleActions, policy, n=5000):
    eps = 0.1
    alpha=0.1
    gamma=1
    lambd=0.8
    pi = np.zeros((policy.shape[0],possibleActions))
    e = np.zeros(pi.shape)
    temperature=0.1
    for i in range(n):
        e[:]=0.
        featureIndex = np.random.randint(500)
        while (featureIndex/20!=location[destination]):
            if np.random.random()>eps:
                action=np.argmax(pi[featureIndex])
            else:
                action=np.random.randint(possibleActions)
                e[:]=0
            newcar = trans[featureIndex/20,action]
            newfeatureIndex=newcar*20 + featureIndex%20
            e *= gamma*lambd
            e[featureIndex,action] += 1
            """ To avoid cycles, we need to set a reward of -1 for every
                time step. This guarantees shortest possible paths.
            """
            if np.random.random()>eps:
                nextaction=np.argmax(pi[newfeatureIndex])
            else:
                nextaction=np.random.randint(possibleActions)
            update = alpha*(-1 +gamma*pi[newfeatureIndex, nextaction] - pi[featureIndex,action])
            pi += update*e
            featureIndex=newfeatureIndex
            
    pi = np.exp(pi/temperature)
    policy[:,:possibleActions] = pi/np.sum(pi, axis=1).reshape(pi.shape[0],1)
    return policy



def getOptions():
    if os.path.isfile('Data/Taxi/options'):
                    with open('Data/Taxi/options','rb') as f:
                        return pickle.load(f)
    options = []
    # RED
    options.append(MarkovOptions("toRED",
        np.array([1 for i in range(500)]),
        learnOptionPolicy(RED,4,np.zeros((500,6)),200000),
        np.array([1 if int(i/20)==location[RED] else 0 for i in range(500)]),
        [0,1,2,3]))
    #GREEN
    options.append(MarkovOptions("toGREEN",
        np.array([1 for i in range(500)]),
        learnOptionPolicy(GREEN,4,0.5*np.zeros((500,6)),200000),
        np.array([1 if int(i/20)==location[GREEN] else 0 for i in range(500)]),
        [0,1,2,3]))
    #YELLOW
    options.append(MarkovOptions("toYELLOW",
        np.array([1 for i in range(500)]),
        learnOptionPolicy(YELLOW,4,0.5*np.zeros((500,6)),200000),
        np.array([1 if int(i/20)==location[YELLOW] else 0 for i in range(500)]),
        [0,1,2,3]))
    #BLUE
    options.append(MarkovOptions("toBLUE",
        np.array([1 for i in range(500)]),
        learnOptionPolicy(BLUE,4,0.5*np.zeros((500,6)),200000),
        np.array([1 if int(i/20)==location[BLUE] else 0 for i in range(500)]),
        [0,1,2,3]))
    # primitive action Up
    options.append(PrimitiveOption(0,"Up",
        np.array([1 for i in range(500)]),
        np.zeros((500,6)),
        np.ones(500)))
    # primitive action Down
    options.append(PrimitiveOption(1,"Down",
        np.array([1 for i in range(500)]),
        np.zeros((500,6)),
        np.ones(500)))
    # primitive action Left
    options.append(PrimitiveOption(2,"Left",
        np.array([1 for i in range(500)]),
        np.zeros((500,6)),
        np.ones(500)))
    # primitive action Right
    options.append(PrimitiveOption(3,"Right",
        np.array([1 for i in range(500)]),
        np.zeros((500,6)),
        np.ones(500)))
    # primitive action Pickup
    options.append(PrimitiveOption(4,"Pickup",
        np.array([1 for i in range(500)]),
        np.zeros((500,6)),
        np.ones(500)))
    # primitive action DropOff
    options.append(PrimitiveOption(5,"DropOff",
        np.array([1 for i in range(500)]),
        np.zeros((500,6)),
        np.ones(500)))
    
    with open('Data/Taxi/options','wb') as f:
        pickle.dump(options, f)
    return options



if __name__=="__main__":
    options = getOptions()
    
    planner = sys.argv[1]
    runs = int(sys.argv[2]) if len(sys.argv)>2 else 1
    MaxEpisodes = int(sys.argv[3]) if len(sys.argv)>3 else 520
    
    np.random.seed(seed=1581990)
    
    for ep in range(runs):
        env = Taxi()
        agent = DYNAAgent(env, options, getFeatures, getState, 
            featuresOn=1, K=5, noOfActions=6, featureVectorLength=500, planning=planner)     
        
        if os.path.isfile('Data/Taxi/rmodel.npy'):
            agent.set_weights(np.load('Data/Taxi/rmodel.npy'), np.load('Data/Taxi/pmodel.npy'), 0.5*np.random.random((len(options), 500, 6)))
        reward=[]
        
        # model=[]
        # vec = np.zeros(500)
        # vec[158]=1

        # if os.path.isfile('Data/Taxi/rewards'):
        #     with open('Data/Taxi/rewards','rb') as f:
        #         reward = pickle.load(f)
        
        for i in range(MaxEpisodes):
            s = env.reset()
            s,r,done = env.step(agent.start(s))

            while(not done):
                action = agent.step(s,r,done)
                s,r,done = env.step(action)
            agent.end(s,r)
            reward.append(env.acc_reward)
            # model.append(np.dot(vec,agent.w_r[0]))
            if i%20==0:
                print ep, i
                _, _, w_q = agent.get_weights()
                #np.save('Data/Taxi/w_r_'+planner+"_"+str(ep), w_r)
                #np.save('Data/Taxi/w_p_'+planner+"_"+str(ep), w_p)
                np.save('Data/Taxi/w_q_'+planner+"_"+str(ep), w_q)
                with open('Data/Taxi/rewards_'+planner+"_"+str(ep),'wb') as f:
                    pickle.dump(reward, f)
                # with open('Data/Taxi/model_'+planner+"_"+str(ep),'wb') as f:
                #     pickle.dump(model, f)

        _, _, w_q = agent.get_weights()
        #np.save('Data/Taxi/w_r_'+planner+"_"+str(ep), w_r)
        #np.save('Data/Taxi/w_p_'+planner+"_"+str(ep), w_p)
        np.save('Data/Taxi/w_q_'+planner+"_"+str(ep), w_q)
        with open('Data/Taxi/rewards_'+planner+"_"+str(ep),'wb') as f:
            pickle.dump(reward, f)
        # with open('Data/Taxi/model_'+planner+"_"+str(ep),'wb') as f:
        #     pickle.dump(model, f)