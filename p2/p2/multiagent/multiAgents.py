# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        score = 0

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # find closest food
        foods = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if len(foods) == 0:
            return successorGameState.getScore() # 
        minFood = min(foods)

        # find closest ghost
        minGhost = manhattanDistance(newPos, newGhostStates[0].configuration.pos)

        score = minGhost / (minFood*100)    # closer the food, the higher the number

        if action == 'Stop':    # stops pacman from being stuck
            score -= 50
        if newScaredTimes != 0: # during powerups
            score += 50

        return successorGameState.getScore() + score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        return self.value(gameState, 0, 0)[1]

    def value(self, gameState, index, depth):
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:
            return gameState.getScore(), ""

        # max
        if index == 0:
            return self.max_value(gameState, index, depth)

        # min
        else:
            return self.min_value(gameState, index, depth)

    def max_value(self, gameState, index, depth):
        legalActions = gameState.getLegalActions(index)
        maxScore = float("-inf")
        maxAction = ""

        for action in legalActions:
            newState = gameState.generateSuccessor(index, action)
            newStateIndex = index + 1
            newStateDepth = depth

            # next index is pacman
            if newStateIndex == gameState.getNumAgents():
                newStateIndex = 0
                newStateDepth += 1

            score = self.value(newState, newStateIndex, newStateDepth)[0]

            if score > maxScore:
                maxScore = score
                maxAction = action

        return maxScore, maxAction

    def min_value(self, gameState, index, depth):
        legalActions = gameState.getLegalActions(index)
        minScore = float("inf")
        minAction = ""

        for action in legalActions:
            newState = gameState.generateSuccessor(index, action)
            newStateIndex = index + 1
            newStateDepth = depth

            # next index is pacman
            if newStateIndex == gameState.getNumAgents():
                newStateIndex = 0
                newStateDepth += 1

            score = self.value(newState, newStateIndex, newStateDepth)[0]

            if score < minScore:
                minScore = score
                minAction = action

        return minScore, minAction
 

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 0, 0, float("-inf"), float("inf"))[1]

    def value(self, gameState, index, depth, alpha, beta):
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:
            return gameState.getScore(), ""

        # max
        if index == 0:
            return self.max_value(gameState, index, depth, alpha, beta)

        # min
        else:
            return self.min_value(gameState, index, depth, alpha, beta)

    def max_value(self, gameState, index, depth, alpha, beta):
        legalActions = gameState.getLegalActions(index)
        maxScore = float("-inf")
        maxAction = ""

        for action in legalActions:
            newState = gameState.generateSuccessor(index, action)
            newStateIndex = index + 1
            newStateDepth = depth

            # next index is pacman
            if newStateIndex == gameState.getNumAgents():
                newStateIndex = 0
                newStateDepth += 1

            score = self.value(newState, newStateIndex, newStateDepth, alpha, beta)[0]

            if score > maxScore:
                maxScore = score
                maxAction = action

            # pruning
            alpha = max(alpha, maxScore)
            if maxScore > beta:
                return maxScore, maxAction

        return maxScore, maxAction

    def min_value(self, gameState, index, depth, alpha, beta):
        legalActions = gameState.getLegalActions(index)
        minScore = float("inf")
        minAction = ""

        for action in legalActions:
            newState = gameState.generateSuccessor(index, action)
            newStateIndex = index + 1
            newStateDepth = depth

            # next index is pacman
            if newStateIndex == gameState.getNumAgents():
                newStateIndex = 0
                newStateDepth += 1

            score = self.value(newState, newStateIndex, newStateDepth, alpha, beta)[0]

            if score < minScore:
                minScore = score
                minAction = action

            # pruning
            beta = min(beta, minScore)
            if minScore < alpha:
                return minScore, minAction

        return minScore, minAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        return self.value(gameState, 0, 0)[1]

    def value(self, gameState, index, depth):
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:
            return gameState.getScore(), ""

        # max
        if index == 0:
            return self.max_value(gameState, index, depth)

        # min
        else:
            return self.exp_value(gameState, index, depth)

    def max_value(self, gameState, index, depth):
        legalActions = gameState.getLegalActions(index)
        maxScore = float("-inf")
        maxAction = ""

        for action in legalActions:
            newState = gameState.generateSuccessor(index, action)
            newStateIndex = index + 1
            newStateDepth = depth

            # next index is pacman
            if newStateIndex == gameState.getNumAgents():
                newStateIndex = 0
                newStateDepth += 1

            score = self.value(newState, newStateIndex, newStateDepth)[0]

            if score > maxScore:
                maxScore = score
                maxAction = action

        return maxScore, maxAction

    def exp_value(self, gameState, index, depth):
        legalActions = gameState.getLegalActions(index)
        v = 0
        vAction = ""

        for action in legalActions:
            successor = gameState.generateSuccessor(index, action)
            successorIndex = index + 1
            successorDepth = depth

            # probability
            p = 1.0 / len(legalActions)

            # next index is pacman
            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            v += p * self.value(successor, successorIndex, successorDepth)[0]

        return v, vAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # pacmanPosition = currentGameState.getPacmanPosition()
    # ghostPositions = currentGameState.getGhostPositions()
    # food = currentGameState.getFood().asList()
    # score = 0

    # # closest food
    # if len(food) > 0:
    #     closeFood = min([manhattanDistance(pacmanPosition, foodPosition) for foodPosition in food])

    # # closest ghost
    # for ghost in ghostPositions:
    #     if [manhattanDistance(pacmanPosition, ghost)] < 2:
    #         closeFood = 1000000

    # return (currentGameState.getScore()*200) + ((1.0 / closeFood) * 10) - (len(food) * 100) - (len(currentGameState.getCapsules())*10)
        # Setup information to be used as arguments in evaluation function
    pacman_position = currentGameState.getPacmanPosition()
    ghost_positions = currentGameState.getGhostPositions()

    food_list = currentGameState.getFood().asList()
    food_count = len(food_list)
    capsule_count = len(currentGameState.getCapsules())
    closest_food = 1

    game_score = currentGameState.getScore()

    # Find distances from pacman to all food
    food_distances = [manhattanDistance(pacman_position, food_position) for food_position in food_list]

    # Set value for closest food if there is still food left
    if food_count > 0:
        closest_food = min(food_distances)

    # Find distances from pacman to ghost(s)
    for ghost_position in ghost_positions:
        ghost_distance = manhattanDistance(pacman_position, ghost_position)

        # If ghost is too close to pacman, prioritize escaping instead of eating the closest food
        # by resetting the value for closest distance to food
        if ghost_distance < 2:
            closest_food = 99999

    features = [1.0 / closest_food,
                game_score,
                food_count,
                capsule_count]

    weights = [10,
               200,
               -100,
               -10]

    # Linear combination of features
    return sum([feature * weight for feature, weight in zip(features, weights)])


# Abbreviation
better = betterEvaluationFunction
