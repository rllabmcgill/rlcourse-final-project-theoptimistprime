import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse

MAX_NUM_VARS = 20
np.random.seed(seed=10)
#rndseq = np.random.randint(0, 2048, (2048))


def GetTiles( num_tilings,           # number of tile indices to be returned in tiles       
            variables,             # array of variables
            num_variables,         # number of variables
            memory_size,           # total number of possible tiles (memory size)
            hash1=-1, hash2=-1, hash3=-1    # change these from -1 to get a different hashing
        ):        
    tiles = T.zeros(num_tilings)
    coordinates = T.zeros(MAX_NUM_VARS + 4)   # one interval number per rel dimension 
    num_coordinates = num_variables

    if hash1 == -1:
        num_coordinates += 1      # no additional hashing corrdinates
    elif hash2 == -1:
        num_coordinates += 2      # one additional hashing coordinates
        coordinates = T.set_subtensor(coordinates[num_variables+1], hash1)
    elif hash3 == -1:
        num_coordinates += 3      # two additional hashing coordinates
        coordinates = T.set_subtensor(coordinates[num_variables+1:num_variables+3], [hash1, hash2])
    else:
        num_coordinates += 4      # three additional hashing coordinates
        coordinates = T.set_subtensor(coordinates[num_variables+1:num_variables+4], [hash1,hash2,hash3])
    
    # quantize state to integers (henceforth, tile widths == num_tilings) 
    qstate = T.mul(variables,num_tilings).astype("int32")
    base = T.zeros(num_variables)
    
    def coordinateValue(qstate, base):
        return ifelse(T.ge(qstate, base), qstate-(qstate-base)%num_tilings, qstate+1+(base-qstate-1)%num_tilings - num_tilings)

    # Takes an array of integer coordinates and returns the corresponding tile after hashing 
    def hash_coordinates(val, index, prev):
        coordinate = val
        coordinate += 449*index
        coordinate %= 2048
        #prev += rndseq[coordinate]
        prev += coordinate
        coordinate = prev % memory_size

        return T.cast(coordinate, "float32")


    # print qstate
    # compute the tile numbers
    for j in range(num_tilings):
        # find coordinates of activated tile in tiling space 
        coord,_ = theano.scan(fn=coordinateValue, 
                               outputs_info = None,
                               sequences = [qstate, base])
        coordinates = T.set_subtensor(coordinates[:num_variables], coord)
        base = base + T.mul(range(num_variables),2)+1
        
        # add additional indices for tiling and hashing_set so they hash differently */
        coordinates = T.set_subtensor(coordinates[num_variables], j)
        # print coordinates[:num_tilings]
        t, _ = theano.scan(fn=hash_coordinates,
                            outputs_info=np.float32(0),
                            sequences=[coordinates, T.arange(num_coordinates)] )
        tiles = T.set_subtensor(tiles[j], t[-1])
    return tiles


def getFeatures(state):
    x = T.clip(state[0], -3, 3)/0.2 # Limiting x to [-3,3] and 30 tiles
    v = T.clip(state[1],-100,100)/1 # v in [-100, 100] with 200 tiles
    theta = T.clip(state[2],-.5,.5)/0.01 # angle is between -30 and 30 degrees with 100 tiles
    omega = T.clip(state[3],-100,100)/1 # 200 tiles
    phi = T.zeros(240)
    phi = T.set_subtensor(phi[:48], GetTiles(48,[x],1,1000,0))
    phi = T.set_subtensor(phi[48:96], GetTiles(48,[theta],1,1000,2))
    phi = T.set_subtensor(phi[96:144], GetTiles(48,[x,v],2,1000,8))
    phi = T.set_subtensor(phi[144:192], GetTiles(48,[theta,omega],2,1000,11))
    phi = T.set_subtensor(phi[192:240], GetTiles(48,[x,v, theta, omega],4,1000,83))
    return phi


if __name__ == "__main__":
    """
    state = theano.shared(np.random.random((4,)).astype("float32"))
    print state.get_value()
    phi=getFeatures(state)
    print "back"
    f=theano.function([],getFeatures(state))
    print f()
    """
    phi = GetTiles(10,[2],1,1000,0)
    f=theano.function([],phi)
    print f()
    phi = GetTiles(10,[1],1,1000,0)
    f=theano.function([],phi)
    print f()