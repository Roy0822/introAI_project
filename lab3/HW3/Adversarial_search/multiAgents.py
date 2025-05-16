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
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        return childGameState.getScore()

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
    Your minimax agent (par1-1)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        """
        pacman = agent[0]
        All states in minimax should be GameStates, either passed in to getAction or generated via GameState.generateSuccessor
        When Pacman believes that his death is unavoidable, he will try to end the game as soon as possible because of the constant penalty for living. Sometimes, this is the wrong thing to do with random ghosts, but minimax agents always assume the worst:
        Make sure you understand why Pacman rushes the closest ghost in this case.

        """
        def getNextAgent(index, numAgents):
            #Determines the next agent's index in a round-robin fashionz.
            return (index + 1) % numAgents  # Wraps around to 0 after the last agent.

        def max_value(index, depth, gameState):
            """
            Pacman's turn to maximize the score.
            """
            bestScore = -float("inf")  # neg infinity
            bestAction = None  # No action

            # Iterate over all legal actions for Pacman.
            for action in gameState.getLegalActions(index):
                nextState = gameState.getNextState(index, action)  # Get the new state after the action.
                nextAgent = getNextAgent(index, gameState.getNumAgents())  # Find the next agent.
                # Determine the depth for the next state; only increment if it goes back to Pacman.
                nextDepth = depth if nextAgent != 0 else depth + 1
                # Recursively call Minimax for the next agent and get the associated score.
                score, _ = miniMaxFunction(nextAgent, nextDepth, nextState)

                # Update bestScore and bestAction if the current score is better.
                if score > bestScore:
                    bestScore = score
                    bestAction = action

            return bestScore, bestAction  # Return the best score and corresponding action.

        def min_value(index, depth, gameState):
            """
            Ghosts' turn to minimize the score.
            """
            bestScore = float("inf")  # Start with the highest possible value.
            bestAction = None  # No action selected yet.

            # Iterate over all legal actions for the ghost.
            for action in gameState.getLegalActions(index):
                nextState = gameState.getNextState(index, action)  # Get the new state after the action.
                nextAgent = getNextAgent(index, gameState.getNumAgents())  # Find the next agent
    
                # Determine the depth for the next state; go deeper when going back to pacman.
                nextdepth = depth if nextAgent != 0 else depth + 1
                # Recursively call Minimax for the next agent and get the associated score.
                score, _ = miniMaxFunction(nextAgent, nextdepth, nextState)

                # Update bestScore and bestAction if the current score is lower (minimizing).
                if score < bestScore:
                    bestScore = score
                    bestAction = action

            return bestScore, bestAction  # Return the best score and corresponding action.




        def miniMaxFunction(index,depth,gameState):

        # Check if game is in a final or the search has reached its depth limit.
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), None  # Return score and no action.
            # If it's a ghost's turn, minimize the score.
            if index == 0 and depth != self.depth:
                return max_value(index, depth, gameState)
            if index != 0 and depth != self.depth:
                return min_value(index, depth, gameState)

        # Initial call to Minimax with Pacman (index 0) and depth 0.
        _, bestAction = miniMaxFunction(0, 0, gameState)  # We only need the action here.
        return bestAction  # Return the best action for Pacman to take.
    
        # End your code

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (part1-2)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        


        # Begin your code
        def expectiMaxFunc(index, depth, gameState):
            # terminating condis
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            if index  == 0: #for pacman
                maxscore = -float("inf") 
                actions = gameState.getLegalActions(0)

                #find the best action recursively
                for act in actions:
                    nextstate = gameState.getNextState(index,act)
                    cur_score = expectiMaxFunc(index+1,depth,nextstate)
                    if cur_score > maxscore:
                        maxscore = cur_score
                return maxscore #get best action
            
            else:
                avg_score = 0
                actions = gameState.getLegalActions(index)
                for action in actions:
                    # Get the state resulting from the ghost's action.
                    nextState = gameState.getNextState(index, action)
                    # Determine the index of the next agent.
                    
                    nextAgent = (index + 1) % gameState.getNumAgents()
                    # Determine the depth for the recursive call.
                    nextDepth = depth + 1 if nextAgent == 0 else depth
                
                    # Recursively call expectimax_search to get the expected score.
                    cur_score = expectiMaxFunc(nextAgent, nextDepth, nextState)
                    # Accumulate the average score.
                    avg_score += cur_score / len(actions)  # Each action contributes equally to the mean.
                    
                return avg_score  # Return the expected score.

        actions = gameState.getLegalActions(0)
        maxScore = -float("inf")
        returnAction = ""

        # get maximum and decide next action
        for action in actions:
            nextState = gameState.getNextState(0, action)
            score = expectiMaxFunc(1, 0, nextState)

            if score > maxScore:
                returnAction = action
                maxScore = score

        return returnAction


        # End your code

better = scoreEvaluationFunction
