import numpy as np

class Option:
    """ A Markov option o consists of a tuple:
    :math:`langle \mathcal{I}, \pi, \beta \rangle`

    where the initiation set :math:`\mathcal{I}` specifies in which states the
    option can be invoked, :math:`\pi` is a sub-policy defined for
    o and :math:`\beta` defines the probability of terminating
    the execution of :math:`\pi` in any given state.

    The options framework was originally introduced in:

    R. S. Sutton, D. Precup, and S. Singh, "Between MDPs and semi-MDPs:
    A framework for temporal abstraction in reinforcement learning,"
    Artificial Intelligence, vol. 112, no. 1-2, pp. 181-211, 1999.

    """

    def initiate(self, observation):
        """ Initiation predicate

        :param observation: the current observation
        :returns: True if this option can be taken in the current observation
        :rtype: bool

        """

    def terminate(self, observation):
        """ Termination (beta) function

        :param observation: the current observation
        :returns: True if this option must terminate in the current observation
        :rtype: bool or float defined as a proper probability function

        """

    def pi(self, observation):
        """ A deterministic greedy policy with respect to the approximate
        action-value function. Please note that we generally don't want to
        learn a policy over non-stationary options. Exploration strategies over
        primitive actions are therefore not needed in this case.

        :param observation: the current observation
        :returns: the greedy action with respect to the action-value function of
        this option
        :rtype: int

        """
class PrimitiveOption(Option):
    """ This class wraps a primitive discrete action into an option.

    A primitive action can be seen as a special kind of option whose
    policy always selects it whenever it is available, can be initiated
    everwhere and only last one step.

    """

    def __init__(self, action, id, I, tran, beta):
        self.action = action
        self.id = id
        self.I = I
        self.b = beta
        self.pi = tran
        self.pi[:,action] = 1

    def initiate(self, features):
        return True

    def terminate(self, features):
        return True

    def pi(self, features):
        return self.action

    def policy(self, features=None):
        return self.action

    def beta(self, features):
        return 1.0

    def tran(self, features, action):
        return 1 if action==self.action else 0

    def __str__(self):
        return 'Primitive option %s'%(self.id)


def makePrimitiveOptions(action_space):
    options = []
    for i in range(action_space.n):
        options.append(PrimitiveOption(i,i))
    return options


class MarkovOptions(Option):
    """
    Temoporaly extended sequence of actions
    param name: unique id to refer to an option
    param I: whether option can be intiated with 'this' feature
    type I: list of booleans
    param tran: probability distribution over all possible actions for features
    type tran: numpy float array of shape (noOfStates, allPossibleActions)
    param beta: probability of termination for 'this' featureVector
    type beta: np float array
    """
    def __init__(self, name, I, tran, beta, actionRange=None):
        self.id = str(name)
        self.I = I
        self.pi = tran
        self.b = beta
        self.actions = actionRange

    """
    initiate():
    -----------
    parameter:  feature vector
    returns:    1 if option can be initiated in with this feature set otherwise returns 0
    """
    def initiate(self, features):
        return np.dot(self.I, features)

    """
    terminate():
    ------------
    parameter:  feature vector
    returns:    1 if options terminates here, 0 means option continues
    """
    def terminate(self, features):
        return np.random.random()<=np.dot(self.b, features)

    
    """
    policy():
    ---------
    parameter:  feature vector
    returns:    a stocastically selected action
    """
    def policy(self, features):
        return np.random.choice( self.pi.shape[1], p=np.dot(features, self.pi) )

    """
    beta():
    -------
    parameter:  feature vector
    returns:    beta value corr. to features
    """
    def beta(self, features):
        return np.dot(self.b, features)

    """
    tran():
    -------
    parameter:  feature vector, action 
    returns:    probability of selecting the action for this feature set
    """
    def tran(self, features, action):
        return np.dot(features, self.pi)[action] if not self.actions or action in self.actions else 0

    def __str__(self):
        return 'Markov Option %s' %self.id
