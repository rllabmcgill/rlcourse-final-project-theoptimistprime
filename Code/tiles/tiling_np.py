import numpy as np
MAX_NUM_VARS = 20
np.random.seed(seed=10)
rndseq = np.random.randint(0, 2048, (2048))


def GetTiles( num_tilings,           # number of tile indices to be returned in tiles       
            variables,             # array of variables
            num_variables,         # number of variables
            memory_size,           # total number of possible tiles (memory size)
            hash1=-1, hash2=-1, hash3=-1    # change these from -1 to get a different hashing
        ):        
    tiles = np.zeros(num_tilings)
    qstate = np.zeros(MAX_NUM_VARS)
    base = np.zeros(MAX_NUM_VARS)
    coordinates = np.zeros(MAX_NUM_VARS + 4)   # one interval number per rel dimension 
    num_coordinates = num_variables

    if hash1 == -1:
        num_coordinates += 1      # no additional hashing corrdinates
    elif hash2 == -1:
        num_coordinates += 2      # one additional hashing coordinates
        coordinates[num_variables+1] = hash1
    elif hash3 == -1:
        num_coordinates += 3      # two additional hashing coordinates
        coordinates[num_variables+1] = hash1
        coordinates[num_variables+2] = hash2
    else:
        num_coordinates += 4      # three additional hashing coordinates
        coordinates[num_variables+1] = hash1
        coordinates[num_variables+2] = hash2
        coordinates[num_variables+3] = hash3
    
    # quantize state to integers (henceforth, tile widths == num_tilings) 
    for i in range(num_variables):
        qstate[i] = int(variables[i] * num_tilings)
        base[i] = 0
    
    # print qstate
    # compute the tile numbers
    for j in range(num_tilings):
    
        # loop over each relevant dimension
        for i in range(num_variables):
        
            # find coordinates of activated tile in tiling space 
            if qstate[i] >= base[i]:
                coordinates[i] = qstate[i] - ((qstate[i] - base[i]) % num_tilings)
            else:
                coordinates[i] = qstate[i]+1 + ((base[i] - qstate[i] - 1) % num_tilings) - num_tilings;
                        
            # compute displacement of next tiling in quantized space 
            base[i] += 1 + (2 * i);
        
        # add additional indices for tiling and hashing_set so they hash differently */
        coordinates[i+1] = j
        i+=1
        # print coordinates[:num_tilings]
        tiles[j] = hash_coordinates(coordinates, num_coordinates, memory_size);
    return tiles.astype("int32")

            
# Takes an array of integer coordinates and returns the corresponding tile after hashing 
def hash_coordinates(coordinates, num_indices, memory_size):
    sum = 0;
    for i in range(num_indices):
        # add random table offset for this dimension and wrap around
        index = coordinates[i]
        index += (449 * i)
        index %= 2048;
        while (index < 0):
            index += 2048
        sum += rndseq[int(index)]
    index = int(sum % memory_size)
    while (index < 0):
        index += memory_size
    
    return index

if __name__ == "__main__":
    """
    state = theano.shared(np.random.random((4,)).astype("float32"))
    print state.get_value()
    phi=getFeatures(state)
    print "back"
    f=theano.function([],getFeatures(state))
    print f()
    """
    phi = GetTiles(10,[10],1,1000,0)
    print phi

