import numpy as np
from copy import deepcopy
from options import Option


class ID(object):
    """
    ID object distinguishes different nodes. ID objects would also be used to fetch
    hashed TreeNodes in MCTS class.

    Parameters-
    -----------

    obs - is the observed state of the environment
    active_option - is the active option. It can either be None or one of the feasible options
                    at the observed state. This value determines the nature of the child nodes.
                    If the active_option is None:
                        children would represent trees formed after selecting different options
                    Else if active_option is one of the feasible options:
                        every child represents a stocastically selected action according to
                        active_opion's policy
    """

    def __init__(self, observation, option=None, action=None):
        self.obs = observation
        self.active_option = option
        self.action = action


hashtable = {} # hash table to store TreeNode objects


def getNode(parent, observation, option, action, prior_p):
    key = hash((observation, option, action))
    if key in hashtable:
        # print "Found in hashtable", observation, option
        return hashtable[key]
    else:
        # print "Creating a new node", observation, option, action
        node = TreeNode(parent, observation, option, action, prior_p)
        hashtable[key] = node
        return node


class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, observation, option, action, prior_p):
        self.ID = ID(observation, option, action)
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        # This value for u will be overwritten in the first call to update(), but is useful for
        # choosing the first action from this node.
        self._u = prior_p
        self._P = prior_p

    def expand_options(self, obs, option_priors):
        """Expand tree by creating new options as children.

        Arguments:
        option_priors -- output from policy function - a list of tuples of actions and their prior
            probability according to the policy function.

        Returns:
        None
        """
        for option, prob in option_priors:
            if option not in self._children:
                self._children[option] = getNode(self, obs, option, None, prob)
        
    def expand_actions(self, obs, action_priors):
        """Expand tree by creating new stocastically selected actions as children.

        Arguments:
        action_priors -- output from policy function - a list of tuples of actions and their prior
            probability according to the policy function.

        Returns:
        None
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = getNode(self, obs, self.ID.active_option, action, prob)

    def select(self):
        """Select action among children that gives maximum action value, Q plus bonus u(P).

        Returns:
        A tuple of (action, next_node)
        """
        return max(self._children.iteritems(), key=lambda (action, node): node.get_value())

    def update(self, leaf_value, c_puct):
        """Update node values from leaf evaluation.

        Arguments:
        leaf_value -- the value of subtree evaluation from the current player's perspective.
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.

        Returns:
        None
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += (leaf_value - self._Q) / self._n_visits
        # Update u, the prior weighted by an exploration hyperparameter c_puct and the number of
        # visits. Note that u is not normalized to be a distribution.
        if not self.is_root():
            self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)

    def update_recursive(self, leaf_value, c_puct):
        """Like a call to update(), but applied recursively for all ancestors.

        Note: it is important that this happens from the root downward so that 'parent' visit
        counts are correct.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(leaf_value, c_puct)
        self.update(leaf_value, c_puct)

    def get_value(self):
        """Calculate and -return the value for this node: a combination of leaf evaluations, Q, and
        this node's prior adjusted for its visit count, u
        """
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTSOption(object):
    """A simple implementation of Monte Carlo Tree Search using options.

    Each subtree results from execution of a different option untill its termination,
    i.e.  branching occurs only when an option ends. While an option os active, next actions 
    are selected using option's policy. With this modification to MCTS, it is adapted for 
    options more sensibly - planning stage can use option model to predict termination 
    features; can also work with actual option selection process without completely disregarding the tree.
    
    """

    def __init__(self, value_fn, policy_fn, rollout_policy_fn, options, get_features, get_states, expectedFeatures, optionRewards, lmbda=0.5, c_puct=5,
                 rollout_limit=500, playout_depth=20, n_playout=10000):
        """ Arguments:
        value_fn -- a function that takes in a state and ouputs a score in [-1, 1], i.e. the
            expected value of the end game score from the current player's perspective.
        policy_fn -- a function that takes in a state and outputs a list of (action, probability)
            tuples for the current player.
        rollout_policy_fn -- a coarse, fast version of policy_fn used in the rollout phase.
        lmbda -- controls the relative weight of the value network and fast rollout policy result
            in determining the value of a leaf node. lmbda must be in [0, 1], where 0 means use only
            the value network and 1 means use only the result from the rollout.
        c_puct -- a number in (0, inf) that controls how quickly exploration converges to the
            maximum-value policy, where a higher value means relying on the prior more, and
            should be used only in conjunction with a large value for n_playout.
        """
        self._root = None
        self._value = value_fn
        self._policy = policy_fn
        self._options = options
        self._get_features = get_features
        self._get_states = get_states
        self._rollout = rollout_policy_fn
        self._lmbda = lmbda
        self._c_puct = c_puct
        self._rollout_limit = rollout_limit
        self._L = playout_depth
        self._n_playout = n_playout
        self._active_option = None
        self._expectedFutureVector = expectedFeatures
        self._rewards = optionRewards
        
        global hashtable
        hashtable = {} #purging hashtable for every new call to MCTSplanning
 
    def _playout(self, env, leaf_depth, option):
        """Run a single playout from the root to the given depth, getting a value at the leaf and
        propagating it back through its parents. State is modified in-place, so a deepcopy must be
        provided.

        Arguments:
        env -- a deep copy of the environent.
        leaf_depth -- after this many moves, leaves are evaluated.
        option -- active_option from previous iteration of get_move()

        Returns:
        None
        """
        observation = env.observation_space.getState()
        epoch = 0
        done = env.episodeEnded()
        node = self._root

        while not done and epoch<leaf_depth:
            epoch += 1
            # If the 'option' terminated here, its time to branch to a new one
            # otherwise just use current option to select next action
            if not option or option.terminate(self._get_states(self._get_features(observation))):
                # if this is a leaf node, expand all possible options
                # print option,"terminated at", observation
                if node.is_leaf():
                    option_probs = self._policy(self._get_features(observation))
                    # print "Expanding with option probabilities", action_probs
                    node.expand_options(observation, option_probs)
                option, node = node.select()
                # print 'Option:',option, epoch, hash(node.ID)
                # sanitory check
                # we are using same stucture to store options and actions
                # if the selected option is not of type Option, print a warning statement
                if not isinstance(option, Option):
                    print "ERROR: option selected by MCTS isn't Option"
            
            # continue using previously active 'option', now we need to select an action 
            # using the option's policy
            if node.is_leaf():
                prob = option.policy(self._get_states(self._get_features(observation)))
                # print "Exapanding", option, "with action probabilities", prob, '\n'
                node.expand_actions(observation, prob)
            action, node = node.select()
            self.history.append(node)
            # print 'action:', action, epoch, hash(node.ID)

            # sanitory check
            # if action isn't an integer, print a warning statement
            if not isinstance(action, int):
                print "ERROR: action was actually an option!!!"
            
            observation, reward, done = env.step(action)
            # print observation, option, action, '\n'

        # Evaluate the leaf using a weighted combination of the value network, v, and the game's
        # winner, z, according to the rollout policy. If lmbda is equal to 0 or 1, only one of
        # these contributes and the other may be skipped. Both v and z are from the perspective
        # of the current player (+1 is good, -1 is bad).
        v = self._value(self._get_features(observation)) if self._lmbda < 1 else 0
        try:
            z = self._evaluate_rollout(env, option, self._rollout_limit) if self._lmbda > 0 else 0
        except:
            #print "WARNING: rollout reached max limit"
            z = v
        leaf_value = (1 - self._lmbda) * v + self._lmbda * z

        # Update value and visit count of nodes in this traversal.
        self.update_node_in_history(leaf_value)


    def _evaluate_rollout(self, env, option, limit=-1):
        """Use the rollout policy to play until the end of the game.
        
        Arguments:
        env -- environment after payout
        limit -- cutoff steps in rollout stage, in case of -1 an episode runs till termination
        option -- option that was last selected in playout stage, if the option hasn't terminated
                we still need to follow it.

        Returns:
        0 if the game doesn't finish under 'limit' steps;
        Environment.totalReward() otherwise. This is the accumulated return.
        
        """
        
        stoppingIndicies = [16, 97, 418, 479] # specific to taxi
        done = env.episodeEnded()
        observation = env.observation_space.getState()
        featureVector = self._get_features(observation)
        epoch=0
        reward = env.totalReward()
        optionIndex = self._options.index(option)

        if not option.terminate(self._get_states(featureVector)):
            reward+= np.dot(self._rewards[optionIndex], featureVector)
            featureVector = self._expectedFutureVector(featureVector, optionIndex)
            done = np.sum(featureVector[stoppingIndicies]) > 0.5

        while epoch!= limit and not done:
            epoch += 1
            option = self._rollout(featureVector)
            optionIndex = self._options.index(option)
            reward+= np.dot(self._rewards[optionIndex], featureVector)
            # setting feature vector to expected vector at the end of the option
            featureVector = self._expectedFutureVector(featureVector, optionIndex)
            done = np.sum(featureVector[stoppingIndicies]) > 0.5
        if not done:
            raise Exception("Ran out of simulation limit") # If no break from the loop, issue a warning.
        return reward
    
    def update_node_in_history(self, leaf_value):
        """Update node values from leaf evaluation.

        Arguments:
        leaf_value -- the value of subtree evaluation from the current player's perspective.
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.

        Returns:
        None
        """
        history = set(self.history)
        for node in history:
            # Count visit
            node._n_visits += 1
            # Update Q, a running average of values for all visits.
            node._Q += (leaf_value - node._Q) / node._n_visits
            # Update u, the prior weighted by an exploration hyperparameter c_puct and the number of
            # visits. Note that u is not normalized to be a distribution.
            if not node.is_root():
                node._u = self._c_puct * node._P * np.sqrt(node._parent._n_visits) / (1 + node._n_visits)

    def get_move(self, env):
        """Runs all playouts sequentially and returns the most visited action.

        Arguments:
        state -- the current state, including both game state and the current player.

        Returns:
        the selected action
        """

        self._root = getNode(parent=None, observation=env.observation_space.getState(), 
            option=None, action=None, prior_p=1.0)

        self.active_option = None

        for n in range(self._n_playout):
            self.history=[]
            self._playout(deepcopy(env), self._L, self._active_option)

        # chosen action is the *most visited child*, not the highest-value one
        # (they are the same as self._n_playout gets large).
        self._active_option = max(self._root._children.iteritems(), key=lambda (a, n): n._n_visits)[0]
        return self._active_option