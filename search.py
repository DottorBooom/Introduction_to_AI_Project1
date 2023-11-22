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
    lists = [s, s, w, s, w, w, s, w]
    print(lists)
    return  lists

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
    ###### Test code ######
    
    #print("Start:", problem.getStartState())
    #print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    #print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    ###### Data structure ###### 
    
    # This part of the code is dedicated to declaring the data structures that will be used for DFS.
    # In particular: a list to contain all the nodes that have been visited and a stack to insert 
    # the neighboring nodes that still need to be visited.
    # Additionally, we will need some way to keep track of the path for the solution. 
    # This will be explained in more detail later in the code. 
    # For now, it's enough to know that these two structures will be the ones allowing us to navigate within the code.
    
    visited = []
    stack = util.Stack()

    ###### DFS ###### 

    # Let's review briefly how the DFS algorithm behaves:
    #       1) Take a certain starting node and expand all neighboring nodes;
    #       2) Insert neighboring nodes into the stack, a Last In First Out (LIFO) structure;
    #       3) Mark the starting node as visited in the list of visited nodes;
    #       4) Perform a pop operation from the stack, and that node becomes the 'current_node';
    #       5) Repeat from step 1.

    # With that we wrote the general outline of the DFS algorithm. Let's now focus on the details:
    #
    # 1) Nodes are tuples of the form ((x, y), "Direction", Weight), where the first element indicates 
    # the coordinates of the node, the second indicates the direction in which Pacman should move to reach it, 
    # and the third indicates the weight to reach that specific position;
    # 2) DFS should return the path from the start node to the goal node. How do we follow 
    # the path without saving it in an additional structure? Every time we insert a new 
    # discovered node into the stack, we update the second element of the tuple from a string 
    # to a list containing all the movements necessary to reach that specific node.

    # And that is the summary of the algorithm we need to implement. Let's now analyze the code line by line.

    current_node = problem.getStartState() # Start with a given start node
    for i in problem.getSuccessors(current_node): # Expan the neighbors and go throw each one of them
        i = list(i) # Change the neighbors from a Tuple to a List, that's cause the type Tuple would not allow us to 
                    # change any type of his component
        i[1] = list([i[1]])     # Change the type of the direction from string to list, so we can insert the 
                                # path to each node from the start 
        stack.push(i)    # Push every neighbors in to the stack, the last will be the next one to be analyzed
    
    visited.append(current_node[0]) # Insert only the coordinates of the start node in the visite list. 
                                    # Why? Because we only need those for undertsand if a node is already be visited
                                    # or if it's a goal state
    while not stack.isEmpty():  # Unless the stack is empty, execute. Simple.
        current_node = stack.pop() # Pop the nex node from the stack

        if problem.isGoalState(current_node[0]): # If that particular node is the goal state
            return current_node[1] #Then, we can return the path that we save throw the execution of the code

        visited.append(current_node[0]) #If the current node is not a goal state, we insert it in the visisted list
        neighbors = problem.getSuccessors(current_node[0]) # And expand its neighbors 
        for i in neighbors: # We go throw every neighbors
            i = list(i) # As already explained before in the first while, we change the type

            if i[0] not in visited: # we iterate inside the visited list and search for an already explored node
                # The next one is foundamental
                i[1] = current_node[1] + [i[1]] # If not, we set the path of the neighbour equal to the path from 
                                                # start to the current + the path to reach the current neighbor
                stack.push(i) #and than push it in to the stack

    # And that's all for the DFS. Let's move on to the next one

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    
    ###### Data structure ###### 
    
    # This part of the code is dedicated to declaring the data structures that will be used for BFS.
    # In particular: a list to contain all the nodes that have been visited and a queue to insert 
    # the neighboring nodes that still need to be visited.
    # Additionally, we will need some way to keep track of the path for the solution. 
    # This will be explained in more detail later in the code. 
    # For now, it's enough to know that these two structures will be the ones allowing us to navigate within the code.
    
    visited = []
    queue = util.Queue()

    ###### BFS ###### 

    # Let's review briefly how the BFS algorithm behaves:
    #       1) Take a certain starting node and expand all neighboring nodes;
    #       2) Insert neighboring nodes into the queue, a Firs In First Out (FIFO) structure;
    #       3) Mark the starting node as visited in the list of visited nodes;
    #       4) Perform a pop operation from the queue, and that node becomes the 'current_node';
    #       5) Repeat from step 1.

    # With that we wrote the general outline of the BFS algorithm. Let's now focus on the details:
    #
    # 1) Nodes are tuples of the form ((x, y), "Direction", Weight), where the first element indicates 
    # the coordinates of the node, the second indicates the direction in which Pacman should move to reach it, 
    # and the third indicates the weight to reach that specific position;
    # 2) BFS should return the path from the start node to the goal node. How do we follow 
    # the path without saving it in an additional structure? Every time we insert a new 
    # discovered node into the queue, we update the second element of the tuple from a string 
    # to a list containing all the movements necessary to reach that specific node.

    # And that is the summary of the algorithm we need to implement. Let's now analyze the code line by line.

    current_node = problem.getStartState() # Start with a given start node
    for i in problem.getSuccessors(current_node): # Expan the neighbors and go throw each one of them
        i = list(i) # Change the neighbors from a Tuple to a List, that's cause the type Tuple would not allow us to 
                    # change any type of his component
        i[1] = list([i[1]])     # Change the type of the direction from string to list, so we can insert the 
                                # path to each node from the start 
        visited.append(i[0])    # Insert the neighbor in to the visisted list; if I don't do so, the same node could 
                                # be explored from another node and be expanded 2 times, and we don't want that
        queue.push(i)    # Push every neighbors in to the queue, the first will be the next one to be analyzed
    
    visited.append(current_node[0]) # Insert only the coordinates of the start node in the visite list. 
                                    # Why? Because we only need those for undertsand if a node is already be visited
                                    # or if it's a goal state

    while not queue.isEmpty():  # Unless the queue is empty, execute. Simple.
        current_node = queue.pop() # Pop the nex node from the queue 

        if problem.isGoalState(current_node[0]): # If that particular node is the goal state
            return current_node[1] #Then, we can return the path that we save throw the execution of the code

        neighbors = problem.getSuccessors(current_node[0]) # If it's not the goal, we expand its node
        for i in neighbors: # We go throw every neighbors
            i = list(i) # As already explained before in the first while, we change the type

            if i[0] not in visited: # we iterate inside the visited list and search for an already explored node
                # The next one is foundamental
                i[1] = current_node[1] + [i[1]] # If not, we set the path of the neighbour equal to the path from 
                                                # start to the current + the path to reach the current neighbor
                visited.append(i[0])            # And then we insert that node in to the visited list, in this way
                                                # if the next node in the queue also have this neighbor is not gonna 
                                                # insert it in the queue again
                queue.push(i) #and than push it in to the queue

    # And that's all for the BFS. Let's move on to the next one

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    ###### Data structure ###### 
    
    # This part of the code is dedicated to declaring the data structures that will be used for UCS.
    # In particular: a list to contain all the nodes that have been visited and a priority queue to insert 
    # the neighboring nodes that still need to be visited.
    # Additionally, we will need some way to keep track of the path for the solution and the cost. 
    # This will be explained in more detail later in the code. 
    # For now, it's enough to know that these two structures will be the ones allowing us to navigate within the code.
    
    visited = []
    priorityQueue = util.PriorityQueue()

    ###### UCS ###### 

    # Let's review briefly how the  algorithm behaves:
    #       1) Take a certain starting node and expand all neighboring nodes;
    #       2) Insert neighboring nodes into the priority queue, particular type of structure that put the lowest path cost first;
    #       3) Mark the starting node as visited in the list of visited nodes;
    #       4) Perform a pop operation from the priority queue, and that node becomes the 'current_node';
    #       5) Repeat from step 1.

    # With that we wrote the general outline of the UCS algorithm. Let's now focus on the details:
    #
    # 1) Nodes are tuples of the form ((x, y), "Direction", Weight), where the first element indicates 
    # the coordinates of the node, the second indicates the direction in which Pacman should move to reach it, 
    # and the third indicates the weight to reach that specific position;
    # 2) UCS should return the path from the start node to the goal node. How do we follow 
    # the path without saving it in an additional structure? Every time we insert a new 
    # discovered node into the priority queue, we update the second element of the tuple from a string 
    # to a list containing all the movements necessary to reach that specific node.
    # 3) We need to keep track of the cost for reach each node. So, like we did with the path, we update the cost
    # before inserting them inside the priority queue. The priority queue will automatically rearange itself to put
    # first the lowest cost path
    # 4) Unlike BFS, if we find an already visited node while exploring some neighbor I don't skip it. Instead, I 
    # confront if the cost to reach that node is less than going throw this neighbor. If so, I remove that node from
    # the visited list, update the path and then insert it again in the priority queue.

    # And that is the summary of the algorithm we need to implement. Let's now analyze the code line by line.

    current_node = problem.getStartState() # Start with a given start node
    visited.append(current_node)    # Insert only the coordinates of the start node in the visite list. 
                                    # Why? Because we only need those for undertsand if a node is already be visited
                                    # or if it's a goal state

    for i in problem.getSuccessors(current_node): # Expan the neighbors and go throw each one of them
        i = list(i) # Change the neighbors from a Tuple to a List, that's cause the type Tuple would not allow us to 
                    # change any type of his component
        i[1] = list([i[1]])    # Change the type of the direction from string to list, so we can insert the 
                                # path to each node from the start 
        visited.append(i) # Insert the visited neighbour in the visited list
        priorityQueue.push(i,i[2])   # Push every neighbors in to the priority queue, the lowest cost one will be the next to be analyzed.

    while not priorityQueue.isEmpty():  # Unless the priority queue is empty, execute. Simple.
        current_node = priorityQueue.pop() # Pop the nex node from the priority queue 

        if problem.isGoalState(current_node[0]): # If that particular node is the goal state
            return current_node[1] #Then, we can return the path that we save throw the execution of the code

        neighbors = problem.getSuccessors(current_node[0]) # If it's not the goal, we expand its node        
        for i in neighbors: # If the node is not a goal state we go throw every neighbors
            i = list(i) # As already explained before the while, we change the type
            if i[0] not in [x[0] for x in visited]: # we iterate inside the visited node and search for an already explored node
                # The next one is foundamental
                i[1] = current_node[1] + [i[1]] # If not, we set the path of the neighbour equal to the path from 
                                                # start to the current + the path to reach the current neighbor
                cost = current_node[2] + i[2] # We save the cost from current node + the cost to reach that neighbor
                i[2] = cost # Update the neighbor cost
                visited.append(i)   # And then we insert that node in to the visited list, in this way
                                    # if the next node in the queue also have this neighbor is not gonna 
                                    # insert it in the queue again
                priorityQueue.push(i, cost) # and than push it in with the updated cost
            else: # Otherwise, if the neighbor is in the vissited list
                for j in visited: # We go throw each node in the visited list
                    if i[0] == j[0] and j != problem.getStartState(): # Using the coordinates, if we find that particular node
                                                                      # And it's not the starting node
                        if j[2] > current_node[2] + i[2]:   # If the cost of that node is major of the current node + the cost to
                                                            # to reach the neighbor
                            visited.remove(j)   # Remove that particular node from the visited list
                            cost = current_node[2] + i[2]   # Update its cost with a new (and less) cost
                            j[2] = cost
                            j[1] = current_node[1] + [i[1]] # Update its new found path
                            priorityQueue.push(j,cost) # And then reinstert it in the priority queue

    # And that's all for the UCS. Let's move on to the next one

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    
    visited = []
    priorityQueue = util.PriorityQueue()
    h = 0

    current_node = problem.getStartState() 
    for i in problem.getSuccessors(current_node): 
        i = list(i) 
        i[1] = list(i[1].split(" "))    
        priorityQueue.push(i,i[2]+h)   
                                        
    visited.append(current_node[0])

    while not priorityQueue.isEmpty(): 
        current_node = priorityQueue.pop() 
        visited.append(current_node[0])
        neighbors = problem.getSuccessors(current_node[0]) 

        if problem.isGoalState(current_node[0]):
            #print(current_node[1])
            return current_node[1]

        for i in neighbors:
            i = list(i) 
            direction = i[1] 
            f = current_node[2] + i[2] + h
            print(visited)
            print(i)
            if i[0] not in visited: 
                i[1] = current_node[1] + [direction]
                i[2] = current_node[2]
                priorityQueue.push(i, f)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
