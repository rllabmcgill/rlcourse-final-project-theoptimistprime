import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Environment.taxi import Taxi, getFeatures, trans
import numpy as np

cab = Taxi()
car, p, des = cab.observation_space.sample()
for i in range(100):
	print 'car', car, 'passenger', p, 'Des', des
	action = np.random.randint(6)
	print action
	obs, reward, done= cab.step(action)
	car, p, des = obs
	print reward, done