import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from options import *
from agent import *
from DynaAgent import *
import pickle
import gym
from tiles.tiling_np import GetTiles


if __name__ == '__main__':
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of output.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    env = gym.make('CartPole-v0' if len(sys.argv)<2 else sys.argv[1])

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    #outdir = '/tmp/random-agent-results'
    #env.monitor.start(outdir, force=True, seed=0)

    # This declaration must go *after* the monitor call, since the
    # monitor's seeding creates a new action_space instance with the
    # appropriate pseudorandom number generator.
    def getFeatures(state):
        x = np.clip(state[0], -3, 3)/0.2 # Limiting x to [-3,3] and 30 tiles
        v = np.clip(state[1],-100,100)/1 # v in [-100, 100] with 200 tiles
        theta = np.clip(state[2],-.5,.5)/0.01 # angleis between -30 and 30 degrees with 100 tiles
        omega = np.clip(state[3],-100,100)/1 # 200 tiles
        phi = []
        phi.extend(GetTiles(24,[x],1,100,0))
        phi.extend(GetTiles(24,[theta],1,100,2))
        phi.extend(GetTiles(24,[x,v],2,100,8))
        phi.extend(GetTiles(24,[theta,omega],2,100,11))
        phi.extend(GetTiles(48,[x,v, theta, omega], 4,100,83))
        features = np.zeros(100).astype('int8')
        features[phi] = 1
        return features

    agent = LearningAgent(env, getFeatures, 100)
     
    if os.path.isfile('w_r100.npy'):
        agent.set_weights(np.load('w_r100.npy'))

    episode_count = 100000
    reward = 0
    
    count=0
    for i in range(episode_count):
        count=2
        ob = env.reset()
        ob, reward, done, _ = env.step( agent.start(ob) )
        while done==False:
            count+=1
            action = agent.step(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if i%50==0:
                print action
                time.sleep(0.1)
                env.render()
        agent.end(ob, reward)
        if i%50==0:
            print "Steps = ",count, "Iteration = ",i
            w_r = agent.get_weights()
            np.save('w_r100', w_r)
    
    # Note there's no env.render() here. But the environment still can open window and
    # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
    # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Dump result info to disk
    #env.monitor.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    #logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    #gym.upload(outdir)
    w_r = agent.get_weights()
    np.save('w_r', w_r)
