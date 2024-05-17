# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions,GameState
from pacman_utils.game import Agent
from pacman_utils import util



class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        "*** YOUR CODE HERE ***"
        # Features used to identify the state
        self.pacmanPosition = state.getPacmanPosition()
        self.ghostPositions = state.getGhostPositions()
        self.food = state.getFood()
        self.food_positions = tuple(state.getFood().asList())
        self.score = state.getScore()
        self.food_num = state.getNumFood()
        self.win = state.isWin()
        self.lose = state.isLose()

        # collect the legal actions
        self.legalAction = state.getLegalPacmanActions()
        
    
    def __hash__(self):
        """
        Returns:
            The hash value representing features
        """
        return hash((self.pacmanPosition, self.food, tuple(self.ghostPositions),
                     self.score, self.food_num, self.food_positions,self.win,self.lose))

class QLearnAgent(Agent):

    def __init__(self,
             alpha: float = 0.2,
             epsilon: float = 0.05,
             gamma: float = 0.8,
             maxAttempts: int = 30,
             numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        
        # Collect all Q-Values
        self.qValues = util.Counter()
        # Collect 
        self.actionCounts={}
        # Collect last states
        self.lastState = []
        # Collect last actions
        self.lastAction = []

        
    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1
    
    def getEpisodesSoFar(self):
        return self.episodesSoFar
    
    def getNumTraining(self):
        return self.numTraining
    
    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value
    
    def getAlpha(self) -> float:
        return self.alpha
    
    def setAlpha(self, value: float):
        self.alpha = value
    
    def getGamma(self) -> float:
        return self.gamma
    
    def getMaxAttempts(self) -> int:
        return self.maxAttempts
    
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        "*** YOUR CODE HERE ***"
        return(endState.getScore()-startState.getScore())
        
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        "*** YOUR CODE HERE ***"
        return self.qValues[(state.__hash__(),action)]
    
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        "*** YOUR CODE HERE ***"
        # Collect all Q-values that satisfy the conditions
        q_list = []
        for action in state.legalAction:
            q = self.getQValue(state,action)
            q_list.append(q)
        if len(q_list) ==0:
            return 0
        # Return the max Q-value
        return max(q_list)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        "*** YOUR CODE HERE ***"
        qMax = self.maxQValue(nextState)
        q = self.getQValue(state,action)
        #Q-learning update function
        self.qValues[(state.__hash__(),action)] = q + self.alpha*(reward + self.gamma*qMax - q)
    

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        "*** YOUR CODE HERE ***"
        key = (state.__hash__(), action)
        if key in self.actionCounts:
            self.actionCounts[key] += 1
        else:
            self.actionCounts[key] = 1
            
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        "*** YOUR CODE HERE ***"
        key = (state.__hash__(), action)
        if key in self.actionCounts:
            count = self.actionCounts[key]
        else:
            count = 0
        return(count)
    
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        "*** YOUR CODE HERE ***"
        if counts < self.maxAttempts:
            # Calculate a bonus based on how close it is to the max attempts
            exploration_bonus = (self.maxAttempts - counts) / self.maxAttempts
        else:
            # Otherwise, no bonus
            exploration_bonus = 0
        return utility + exploration_bonus

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # Legal actions of the state
        legal = state.getLegalPacmanActions()
        # Remove stop as an action
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        # Convert state into GameStateFeatures
        stateFeatures = GameStateFeatures(state)
        # Start from the second move 
        if len(self.lastState) > 0:
            last_state = self.lastState[-1]
            # Convert state into GameStateFeatures
            last_stateFeatures = GameStateFeatures(last_state)
            last_action = self.lastAction[-1]
            # Compute the reward
            reward = self.computeReward(last_state,state)
            # Learn
            self.learn(last_stateFeatures, last_action, reward, stateFeatures)
            # Update the count of learning events
            self.updateCount(last_stateFeatures, last_action)
        # Epsilon-greedy exploration
        if util.flipCoin(self.epsilon):
            action =  random.choice(legal)
        else:            
            # Collect legal actions of the state again after learning
            legal = state.getLegalPacmanActions()
            # Agent shouldn't stop or turn back while not being chased by a ghost 
            # during the first half of training
            if (self.getEpisodesSoFar() * 1.0) / self.getNumTraining() < 0.5:
                if Directions.STOP in legal:
                    legal.remove(Directions.STOP)
                # Check if the last action list is not empty
                if len(self.lastAction) > 0:
                    last_action = self.lastAction[-1]
                    # Calculate the horizontal distance between Pacman and the first ghost
                    distance0 = state.getPacmanPosition()[0] - state.getGhostPosition(1)[0]
                    # Calculate the vertical distance between Pacman and the first ghost
                    distance1 = state.getPacmanPosition()[1] - state.getGhostPosition(1)[1]
                    # Check if Pacman is more than 2 units away from the first ghost
                    if (distance0**2 + distance1**2) > 4:
                        # Check if reverse is in the list of legal actions 
                        # and there are more than one legal actions
                        if (Directions.REVERSE[last_action] in legal) and len(legal) > 1:
                            # Remove reverse from the list of legal actions
                            legal.remove(Directions.REVERSE[last_action])
            # Explore by using explore function
            utilities = []
            for action in legal:
                qValue = self.getQValue(stateFeatures, action)
                count = self.getCount(stateFeatures, action)
                utilitywithBonus = self.explorationFn(qValue,count)

                utilities.append((utilitywithBonus,action))
            # Exploitation: find the action that maximizes Q of the state
            utility , action = max(utilities, key=lambda x: x[0])
            
        # Collect states and actions
        self.lastState.append(state)
        self.lastAction.append(action)
        return action


    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        # Learning when win or loss a game
        # Convert state into GameStateFeatures
        stateFeatures = GameStateFeatures(state)
        last_state = self.lastState[-1]
        # Convert state into GameStateFeatures
        last_stateFeatures = GameStateFeatures(last_state)
        last_action = self.lastAction[-1]
        # Compute the rewaard
        reward = self.computeReward(last_state,state)
        # Learn
        self.learn(last_stateFeatures, last_action, reward, stateFeatures)
        # Update the count of learning events
        self.updateCount(last_stateFeatures, last_action)
        
        # Reset state and action lists
        self.lastState = []
        self.lastAction = []

        print(f"Game {self.getEpisodesSoFar()} just ended!")
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print ('%s\n%s' % (msg,'-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)