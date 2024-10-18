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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodDistance = 0
        score = successorGameState.getScore() + max(newScaredTimes)
        ghostPos = successorGameState.getGhostPositions()

        # Calculate the distance to the closest food
        for food in newFood.asList():
            foodDistNew = abs(newPos[0] - food[0]) + abs(newPos[1] - food[1])
        # updates closest food distance
            if foodDistance > 0:
                foodDistance = min(foodDistance, foodDistNew)
            else:
                foodDistance = foodDistNew

        # change score based on food distance
        if foodDistance < 2:
            score += 10
        else:
            score += 1 / foodDistance

        # check if Pacman is too close to a ghost
        for ghost in ghostPos:
            ghostDistance = abs(newPos[0] - ghost[0]) + abs(newPos[1] - ghost[1])
            # if Pacman is too close to a ghost, set a penalty
            if ghostDistance < 2:
                score = -100000

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        def minValue(state, agentIndex, depth):
            agentCount = gameState.getNumAgents()
            legalActions = state.getLegalActions(agentIndex)

            #check if there is no legal action
            if not legalActions:
                return self.evaluationFunction(state)

            #check if agent is Pacman
            if agentIndex == agentCount - 1:
                minimumValue = float('inf')  # Set to positive infinity initially
                #goes through all legal actions in this case Pacman
                for action in legalActions:
                    successor_state = state.generateSuccessor(agentIndex, action)
                    value = maxValue(successor_state, agentIndex, depth)
                    minimumValue = min(minimumValue, value)
            else:
                minimumValue = float('inf')  # Set to positive infinity initially
                #gooes through all legal actions
                for action in legalActions:
                    successor_state = state.generateSuccessor(agentIndex, action)
                    value = minValue(successor_state, agentIndex + 1, depth)
                    minimumValue = min(minimumValue, value)

            return minimumValue

        def maxValue(state, agentIndex, depth):
            agentIndex = 0
            legalActions = state.getLegalActions(agentIndex)

            #checks if there is no legal actions or if depth reached
            if not legalActions or depth == self.depth:
                return self.evaluationFunction(state)

            maximumValue = float('-inf')  # Set to negative infinity initially
            #goes through all legal actions in this case Pacman
            for action in legalActions:
                successor_state = state.generateSuccessor(agentIndex, action)
                value = minValue(successor_state, agentIndex + 1, depth + 1)
                maximumValue = max(maximumValue, value)
            return maximumValue

        #get legal action for Pacman
        actions = gameState.getLegalActions(0)

        #calculate minimum value for each action and stores it into allActions
        allActions = {}
        for action in actions:
            allActions[action] = minValue(gameState.generateSuccessor(0, action), 1, 1)

        return max(allActions, key=allActions.get)
        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """


    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphaBetaValue(gameState, 0, 0, float("-inf"), float("inf"))
        #util.raiseNotDefined()

    def alphaBetaValue(self, gameState, agentIndex, nodeDepth, alpha, beta):

        # checks if all agents have moved
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
            nodeDepth += 1

        # if search reaches a specific depth
        if nodeDepth == self.depth:
            # returns evaluation of current state
            return self.evaluationFunction(gameState)

        # if it's agent's turn, call max_value else call min_value
        if agentIndex == self.index:
            return self.maxValueAlphaBeta(gameState, agentIndex, nodeDepth, alpha, beta)
        else:
            return self.minValueAlphaBeta(gameState, agentIndex, nodeDepth, alpha, beta)

    def maxValueAlphaBeta(self, gameState, agentIndex, nodeDepth, alpha, beta):

        # #checks for terminal state either win or lose, return evaluation of current state
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        value = float("-inf")
        actionValue = None

        # goes through legal actions of current agent
        for legalActions in gameState.getLegalActions(agentIndex):
            # checks if legalAction is stop
            if legalActions != Directions.STOP:
                successor = gameState.generateSuccessor(agentIndex, legalActions)
                temp = self.alphaBetaValue(successor, agentIndex + 1, nodeDepth, alpha, beta)

                # if better move, value and actionValue is updated
                if temp > value:
                    value, actionValue = temp, legalActions

                # prune if value goes over beta
                if value > beta:
                    return value

                alpha = max(alpha, value)

        # if at the root node return best action
        if nodeDepth == 0:
            return actionValue
        else:
            return value

    def minValueAlphaBeta(self, gameState, agentIndex, nodeDepth, alpha, beta):

        #checks for terminal state either win or lose, return evaluation of current state
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        value = float("inf")
        actionValue = None

        # goes through legal actions of current agent
        for legalActions in gameState.getLegalActions(agentIndex):
            if legalActions != Directions.STOP:
                successor = gameState.generateSuccessor(agentIndex, legalActions)
                temp = self.alphaBetaValue(successor, agentIndex + 1, nodeDepth, alpha, beta)

                if temp < value:
                    value, actionValue = temp, legalActions

                # Prune if value is less than alpha
                if value < alpha:
                    return value
                # update beta
                beta = min(beta, value)

        return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """



    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(gameState,0,0)
        #util.raiseNotDefined()

    def expValue(self, gameState, agent, depth):
        expectedValue = 0

        legalActions = gameState.getLegalActions(agent)

        numberOfLegalActions = len(legalActions)

        #checks if there are any legal moves
        if numberOfLegalActions == 0:
            return expectedValue

        #calculates the probability of each legal action
        probability = 1.0 / numberOfLegalActions

        #goes through legal actions
        for action in legalActions:
            successor = gameState.generateSuccessor(agent, action)

            currentExpectedValue = self.expectimax(successor, agent + 1, depth)
            expectedValue += probability * currentExpectedValue

        return expectedValue

    def expectimax(self, gameState, agentIndex=0, depth=0):

        agentIndex %= gameState.getNumAgents()

        #checks for win or lose or depth limit
        if gameState.isWin() or gameState.isLose() or (agentIndex == 0 and depth == self.depth):
            return self.evaluationFunction(gameState)

        #checks if agent is Pacman and if current depth is less than specific max depth
        if agentIndex == 0 and depth < self.depth:
            depth += 1
            return self.expmaxValue(gameState, agentIndex, depth)
        #else ghost
        else:
            return self.expValue(gameState, agentIndex, depth)

    def expmaxValue(self, gameState, agentIndex, depth):
        expectedMaxValue = float('-inf')
        finalAction = None

        legalActions = gameState.getLegalActions(agentIndex)

        #loops through legalActions
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            currentMax = self.expectimax(successor, agentIndex + 1, depth)

            #takes the maximum value of the higher value
            if expectedMaxValue < currentMax:
                expectedMaxValue = currentMax
                #updates finalAction with current action
                finalAction = action

        if depth == 1:
            return finalAction
        else:
            return expectedMaxValue

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    x,y = currentGameState.getPacmanPosition()

    food = currentGameState.getFood()
    foodList = food.asList()

    #checks if there is any food left
    if not foodList:
        return float('inf')

    #determines distance from Pacman to each food position
    foodDistances = []
    for foodPos in foodList:
        foodDistance = abs(x - foodPos[0]) + abs(y - foodPos[1])
        foodDistances.append(foodDistance)

    #finds the minimum distance to food
    foodDis = min(foodDistances)
    foodScore = 1 / foodDis

    capsules = currentGameState.getCapsules()

    #check if there are capsules
    if len(capsules) != 0:
        capsuleDistances = []
        for capsule in capsules:
            #distance between Pacman and capsule
            capDistance = abs(x - capsule[0]) + abs(y - capsule[1])
            capsuleDistances.append(capDistance)

        capsuleDis = min(capsuleDistances)
        capsuleScore = 1 / capsuleDis
    else:
        capsuleScore = 0

    ghostStates = currentGameState.getGhostStates()

    ghostDistances = []
    for ghost in ghostStates:
        ghost_position = ghost.getPosition()
        #the distance between Pacman and current ghost
        ghostDistance = abs(x - ghost_position[0]) + abs(y - ghost_position[1])
        ghostDistances.append(ghostDistance)

    #find closest ghost
    nearestGhostDistance = min(ghostDistances)
    ghostScore = 1 / (nearestGhostDistance + 1)

    #checks if ghosts are scared, aka Pacman can eat ghosts
    scaredTime = ghostStates[0].scaredTimer
    if scaredTime > 0:
        coefficient = 2
    #ghosts are not scared
    else:
        coefficient = -1

    finalScore = currentGameState.getScore() + foodScore + capsuleScore + coefficient * ghostScore
    return finalScore

# Abbreviation
better = betterEvaluationFunction
