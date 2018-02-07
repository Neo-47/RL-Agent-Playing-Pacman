# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
	"""
	  Q-Learning Agent

	  Functions you should fill in:
		- computeValueFromQValues
		- computeActionFromQValues
		- getQValue
		- getAction
		- update

	  Instance variables you have access to
		- self.epsilon (exploration prob)
		- self.alpha (learning rate)
		- self.discount (discount rate)

	  Functions you should use
		- self.getLegalActions(state)
		  which returns legal actions for a state
	"""
	def __init__(self, **args):
		"You can initialize Q-values here..."
		ReinforcementAgent.__init__(self, **args)

		"*** YOUR CODE HERE ***"

		#initialize a Q_values dictionary for each action per state to 0
		self.Q_values_for_actions = util.Counter()


	def getQValue(self, state, action):
		"""
		  Returns Q(state,action)
		  Should return 0.0 if we have never seen a state
		  or the Q node value otherwise
		"""
		"*** YOUR CODE HERE ***"

		#If we have never seen the action for that state return 0.0
		if(self.Q_values_for_actions[(state, action)] == 0):
			return 0.0

		#else return the Q-value for that action for that state
		return self.Q_values_for_actions[(state, action)]

		util.raiseNotDefined()


	def computeValueFromQValues(self, state):
		"""
		  Returns max_action Q(state,action)
		  where the max is over legal actions.  Note that if
		  there are no legal actions, which is the case at the
		  terminal state, you should return a value of 0.0.
		"""
		"*** YOUR CODE HERE ***"
		#get legal actions for that state
		actions = self.getLegalActions(state)

		#if there are no actions return 0.0
		if(not actions):
			return 0.0

		#initialize a q_values dictionary for each action for that state to 0
		q_values = util.Counter()

		#loop on possible actions for that state
		for action in actions:

			#get the Q-value for that action
			q_values[action] = self.getQValue(state, action)

		#return the maximum value from values for actions
		return max(q_values.values())

		util.raiseNotDefined()

	def computeActionFromQValues(self, state):
		"""
		  Compute the best action to take in a state.  Note that if there
		  are no legal actions, which is the case at the terminal state,
		  you should return None.
		"""
		"*** YOUR CODE HERE ***"

		#Get possible actions for that state
		actions = self.getLegalActions(state)

		#If there are no actions for that state, return 0.0
		if(len(actions) == 0):
			return None

		#initialize a q_values dictionary for each action for that state to 0
		q_values = util.Counter()

		#loop on actions
		for action in actions:

			#get the Q-value for that action
			q_values[action] = self.getQValue(state, action)

		#return the action with the maximum Q-value
		return q_values.argMax()

		util.raiseNotDefined()

	def getAction(self, state):
		"""
		  Compute the action to take in the current state.  With
		  probability self.epsilon, we should take a random action and
		  take the best policy action otherwise.  Note that if there are
		  no legal actions, which is the case at the terminal state, you
		  should choose None as the action.

		  HINT: You might want to use util.flipCoin(prob)
		  HINT: To pick randomly from a list, use random.choice(list)
		"""
		# Pick Action
		legalActions = self.getLegalActions(state)
		"*** YOUR CODE HERE ***"

		if(not legalActions):
			return None

		exploration_flag = util.flipCoin(self.epsilon)

		if(exploration_flag):
			action = random.choice(legalActions)

		else:
			action = self.computeActionFromQValues(state)
		
		return action

		util.raiseNotDefined()
	def update(self, state, action, nextState, reward):
		"""
		  The parent class calls this to observe a
		  state = action => nextState and reward transition.
		  You should do your Q-Value update here

		  NOTE: You should never call this function,
		  it will be called on your behalf
		"""
		"*** YOUR CODE HERE ***"

		#compute the current sample, the immediate reward + the discounted maximum
		#Q-value for the state the agent landed in. 
		sample = reward + self.discount * self.computeValueFromQValues(nextState)

		#update every Q-value for each action per state, based on the current sample and the stored value of Q(s, a)
		self.Q_values_for_actions[(state, action)] = (1 - self.alpha) * self.Q_values_for_actions[(state, action)] + self.alpha * sample

		#return the updated Q-values
		return self.Q_values_for_actions.copy()

		util.raiseNotDefined()

	def getPolicy(self, state):
		return self.computeActionFromQValues(state)

	def getValue(self, state):
		return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
	"Exactly the same as QLearningAgent, but with different default parameters"

	def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
		"""
		These default parameters can be changed from the pacman.py command line.
		For example, to change the exploration rate, try:
			python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

		alpha    - learning rate
		epsilon  - exploration rate
		gamma    - discount factor
		numTraining - number of training episodes, i.e. no learning after these many episodes
		"""
		args['epsilon'] = epsilon
		args['gamma'] = gamma
		args['alpha'] = alpha
		args['numTraining'] = numTraining
		self.index = 0  # This is always Pacman
		QLearningAgent.__init__(self, **args)

	def getAction(self, state):
		"""
		Simply calls the getAction method of QLearningAgent and then
		informs parent of action for Pacman.  Do not change or remove this
		method.
		"""
		action = QLearningAgent.getAction(self,state)
		self.doAction(state,action)
		return action


class ApproximateQAgent(PacmanQAgent):
	"""
	   ApproximateQLearningAgent

	   You should only have to overwrite getQValue
	   and update.  All other QLearningAgent functions
	   should work as is.
	"""
	def __init__(self, extractor='IdentityExtractor', **args):
		self.featExtractor = util.lookup(extractor, globals())()
		PacmanQAgent.__init__(self, **args)
		self.weights = util.Counter()

	def getWeights(self):
		return self.weights

	def getQValue(self, state, action):
		"""
		  Should return Q(state,action) = w * featureVector
		  where * is the dotProduct operator
		"""
		"*** YOUR CODE HERE ***"
		#get the features of current action given current state
		features = self.featExtractor.getFeatures(state, action)
		
		#perform dot product
		Q_value = self.weights * features

		return Q_value
		util.raiseNotDefined()

	def update(self, state, action, nextState, reward):
		"""
		   Should update your weights based on transition
		"""
		"*** YOUR CODE HERE ***"
		#get features of current action given current state
		features = self.featExtractor.getFeatures(state, action)
		#compute the difference = (R + gamma * max_a Q(s_prime, a_prime)) - Q(s, a)
		difference = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)

		#loop on features dictionary, perform in place operation f = f * alpha * difference
		for key in features.keys():
			features[key] = features[key] * self.alpha * difference

		#add the new f to the old w to get the new w, w = w + f
		self.weights.__radd__(features)

		#return to terminate the function
		return

		util.raiseNotDefined()

	def final(self, state):
		"Called at the end of each game."
		# call the super-class final method
		PacmanQAgent.final(self, state)

		# did we finish training?
		if self.episodesSoFar == self.numTraining:
			# you might want to print your weights here for debugging
			"*** YOUR CODE HERE ***"
			pass
