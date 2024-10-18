# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    visited = set()
    stack = util.Stack()
    start = problem.getStartState()
    stack.push((start, []))

    while stack.isEmpty() == False: # checks if stack is empty
        currentState, path = stack.pop()
        if problem.isGoalState(currentState):  # checks if we reached our goal state
            return path

        if currentState not in visited: #checks if currentNode was visited
            visited.add(currentState)

            successors = problem.getSuccessors(currentState)
            for (successor, action, cost) in successors:
                newPath = path + [action]
                stack.push((successor, newPath))

    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    startState = problem.getStartState()
    queue = util.Queue()
    visited = []

    queue.push((startState, [], 0))

    # checks if priorityQueue is empty
    while queue.isEmpty() == False:
        currentState, path, costs = queue.pop()

        # check if currentState was visited
        if currentState not in visited:
            visited.append(currentState)

            if problem.isGoalState(currentState):
                return path

            successors = problem.getSuccessors(currentState)
            for nextState, action, cost in successors:
                if nextState not in visited:
                    newPath = path +[action]
                    queue.push((nextState, newPath, cost))
    util.raiseNotDefined()


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    visited = set()
    priorityQueue = util.PriorityQueue()

    priorityQueue.push((problem.getStartState(), []), 0)

    # checks if priorityQueue is empty
    while priorityQueue.isEmpty() == False:
        currentState, path = priorityQueue.pop()

        # checks if currentState is at goal state
        if problem.isGoalState(currentState):
            return path

        # check if currentState was visited
        if currentState not in visited:
            visited.add(currentState)

            successors = problem.getSuccessors(currentState)
            for nextState, action, cost in successors:
                # checks if the nextState was visited
                if nextState not in visited:
                    newPath = path + [action]
                    newCost = problem.getCostOfActions(newPath)
                    priorityQueue.push((nextState, newPath), newCost)
    return []
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    startState = problem.getStartState()
    # A* using priority queue to prioritize successors with the least heuristic cost
    priorityQueue = util.PriorityQueue()
    visited = set()

    #setting totalpathCost, combinedCost, and priority for ordering to 0
    priorityQueue.push((startState, [], 0, 0), 0)

    while priorityQueue.isEmpty() == False:
        currentState, path, totalPathCost, combinedCost = priorityQueue.pop()

        # check if currentState has already been visited with a lower cost
        if currentState in visited and totalPathCost + heuristic(currentState, problem) >= combinedCost:
            continue

        # Update visited status and combined cost
        visited.add(currentState)

        if problem.isGoalState(currentState):
            return path

        successors = problem.getSuccessors(currentState)
        for nextState, action, cost in successors:
            if nextState not in visited: # checks if the nextState was visited
                newPathCost = totalPathCost + cost
                heuristicCost = heuristic(nextState, problem)
                newCombinedCost = newPathCost + heuristicCost
                newPath = path + [action]
                #pushes successor state, newPath, pathCost, and newCombinedCost into priority queue, and priorizes the ordering by the estimated combinedCost to reach the goal state.
                priorityQueue.push((nextState, newPath, newPathCost, newCombinedCost), newCombinedCost)

    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
