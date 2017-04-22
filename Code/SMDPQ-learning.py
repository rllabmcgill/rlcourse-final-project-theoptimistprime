import numpy as np

class SMDP-Q(object):

	def __init(self, options, nFeatures, getState):
		self.epsilon = 0.1
		self.alpha = 0.0001
		self.gamma = 0.9
		self.activeOption = None
		self.r=0
		self.k=0
		self.startingfeature = None
		self.getState = getState
		self.options = options
		self.nFeatures = nFeatures
		self.qOption = np.random.random((len(options), nFeatures))

	def set_weights(self, q):
        self.qOption = q

    def get_weights(self):
        return self.qOption

    def egreedy(self, features):
    	if np.random.random()>self.epsilon:
    		return np.dot(self.qOption, features).argmax()
    	else:
    		return np.random.randint(len(options))

   	def Q_learning(self, oIndex, r, k, features, nextFeatures):
   		if not oIndex:
   			return
   		delta = r + self.gamma**k * np.dot(self.qOption, nextFeatures).max() - np.dot(self.qOption[oIndex], features)
   		self.qOption[oIndex]+= self.alpha * features* delta

   	def act(self, reward, features):
   		state = self.getState(features)
   		if not self.activeOption or self.options[self.activeOption].terminate(state):
   			self.Q_learning(self.activeOption, self.r, self.k, self.startingfeature, features)
   			self.activeOption = self.egreedy(features)
   			self.k=0
   			self.r=0
   			self.startingfeature = features
   		else:
   			self.r+=reward
   			self.k+=1
   		return self.options[self.activeOption].pi(state)