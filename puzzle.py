from __future__ import division
from __future__ import print_function

import sys
import math
import time
import queue as Q
import resource

##Global variable for the goal state


#### SKELETON CODE ####
## The Class that Represents the Puzzle
class PuzzleState(object):

    """
        The PuzzleState stores a board configuration and implements
        movement instructions to generate valid children.
    """
    def __init__(self, config, n, parent=None, action="Initial", cost=0, depth=0):
        """
        :param config->List : Represents the n*n board, for e.g. [0,1,2,3,4,5,6,7,8] represents the goal state.
        :param n->int : Size of the board
        :param parent->PuzzleState
        :param action->string
        :param cost->int
        :param depth ->int
        """
        if n*n != len(config) or n < 2:
            raise Exception("The length of config is not correct!")
        if set(config) != set(range(n*n)):
            raise Exception("Config contains invalid/duplicate entries : ", config)

        self.n        = n
        self.cost     = cost
        self.parent   = parent
        self.action   = action
        self.config   = config
        self.children = []
        self.depth = depth

        # Get the index and (row, col) of empty block
        self.blank_index = self.config.index(0)

    def __lt__(self, state2):
        mycost = calculate_total_cost(self)
        cost = calculate_total_cost(state2)
        if mycost < cost: # if both are same then just say I am lower??
            return True
        elif mycost == cost:
            return self.depth < state2.depth
        return False;
    
    def display(self):
        """ Display this Puzzle state as a n*n board """
        for i in range(self.n):
            print(self.config[3*i : 3*(i+1)])

    def move_up(self):
        """ 
        Moves the blank tile one row up.
        :return a PuzzleState with the new configuration
        """
        
        index = self.blank_index
        length = self.n
        """need to find if moving left is possible, check row"""
        if index // length == 0:
            return None
            
        """able to move, swap positions now"""
        clone = list(self.config)
        clone[index - length] = self.config[index]
        clone[index] = self.config[index - length]
        
        up_child = PuzzleState(
        config=clone,
        n=self.n,
        parent=self,
        action="Up",
        cost=self.cost + 1,
        depth = self.depth + 1
        )
        
        return up_child
      
    def move_down(self):
        """
        Moves the blank tile one row down.
        :return a PuzzleState with the new configuration
        """
        clone = list(self.config)
        index = self.blank_index
        length = self.n
        """need to find if moving left is possible, check row"""
        if index // length == length - 1:
            return None
            
        """able to move, swap positions now"""
        clone[index + length] = self.config[index]
        clone[index] = self.config[index + length]
        
        down_child = PuzzleState(
        config=clone,
        n=self.n,
        parent=self,
        action="Down",
        cost=self.cost + 1,
        depth = self.depth + 1
        )
        
        return down_child
      
    def move_left(self):
        """
        Moves the blank tile one column to the left.
        :return a PuzzleState with the new configuration
        """
        clone = list(self.config)
        index = self.blank_index
        length = self.n
        """need to find if moving left is possible, check column"""
        if index % length == 0:
            return None
            
        """able to move, swap positions now"""
        clone[index - 1] = self.config[index]
        clone[index] = self.config[index - 1]
        
        left_child = PuzzleState(
        config=clone,
        n=self.n,
        parent=self,
        action="Left",
        cost=self.cost + 1,
        depth = self.depth + 1
        )
        
        return left_child

    def move_right(self):
        """
        Moves the blank tile one column to the right.
        :return a PuzzleState with the new configuration
        """
        clone = list(self.config)
        index = self.blank_index
        length = self.n
        """need to find if moving right is possible, check column"""
        if index % length == length - 1:
            return None
            
        """able to move, swap positions now"""
        clone[index + 1] = self.config[index]
        clone[index] = self.config[index + 1]
        
        right_child = PuzzleState(
        config=clone,
        n=self.n,
        parent=self,
        action="Right",
        cost=self.cost + 1,
        depth = self.depth + 1
        )
        
        return right_child
        
      
    def expand(self) :
        """ Generate the child nodes of this node """
        
        # Node has already been expanded
        if len(self.children) != 0:
            return self.children
        
        # Add child nodes in order of UDLR
        children = [
            self.move_up(),
            self.move_down(),
            self.move_left(),
            self.move_right()]

        # Compose self.children of all non-None children states
        self.children = [state for state in children if state is not None]
        return self.children

# Function that Writes to output.txt

### Students need to change the method to have the corresponding parameters
def writeOutput(path, depth, max_depth, nodes_expanded, run_time, ram_use, filename="output.txt"):
    with open(filename, "w") as file:
    ### Student Code Goes here
        file.write(f"path_to_goal: {path}\n")
        file.write(f"cost_of_path: {len(path)}\n")
        file.write(f"nodes_expanded: {nodes_expanded}\n")
        file.write(f"search_depth: {depth}\n")
        file.write(f"max_search_depth: {max_depth}\n")
        file.write(f"running_time: {run_time}\n")
        file.write(f"max_ram_usage: {ram_use}\n")

def bfs_search(initial_state):
    """BFS search"""
    ### STUDENT CODE GOES HERE ###
    bfs_frontier = Q.Queue()
    visited = set()
    in_bfs_frontier = set()
    max_search_depth = 0
    nodes_expanded = 0
    time_start = time.process_time()
    start_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    """case : expand and find goal state, with frontier and visited included"""
    """1) establish the frontier, visited,max_depth, and expanded nodes"""
    bfs_frontier.put(initial_state)
    in_bfs_frontier.add(tuple(initial_state.config))
    
    
    """2) go through bfs loop while the frontier is not empty"""
    while not bfs_frontier.empty():
        ###Take state out of the queue
        state = bfs_frontier.get()
#        print(f"printing {state.action} {state.config}")
        state_config = tuple(state.config)
        in_bfs_frontier.discard(state_config)
        visited.add(state_config)

        if test_goal(state):
            ##find the path
#            print("final")
            path = []
            child_state = state
            while child_state.parent is not None:
                path.append(child_state.action)
                child_state = child_state.parent
            ##going from child to root, need other way, use reverse method on list
            path.reverse()
            time_end = time.process_time()
            time_run = time_end - time_start
            ram_usage = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - start_ram) / (2**20)
#            print(f"final path is {path}")
            ###insert write output and return
#            print(f"nodes expanded {nodes_expanded}")
            writeOutput(path, len(path), max_search_depth, nodes_expanded, time_run, ram_usage)
            return

        ###not goal state, now expand
        nodes_expanded+=1
        for child in state.expand():
            child_config = tuple(child.config)
            ###check to see not in visited or already in the frontier
            if (child_config not in visited) and (child_config not in in_bfs_frontier):
                bfs_frontier.put(child)
                in_bfs_frontier.add(child_config)
                if child.depth > max_search_depth:
                    max_search_depth = child.depth
#        print("last " + str(bfs_frontier.empty()))
            

def dfs_search(initial_state):
    """DFS search"""
    ### STUDENT CODE GOES HERE ###
        
    """case : expand and find goal state, with frontier and visited included"""
    """1) establish the stack, visited,max_depth, and expanded nodes"""
    dfs_frontier = []
    in_dfs_frontier = set()
    visited = set()
    max_search_depth = 0
    nodes_expanded = 0
    time_start = time.process_time()
    start_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    dfs_frontier.append(initial_state)
    in_dfs_frontier.add(tuple(initial_state.config))
    
    
    while dfs_frontier :
        ###Take state out of the queue
        state = dfs_frontier.pop()
        state_config = tuple(state.config)
        in_dfs_frontier.discard(state_config)
        ###avoid putting duplicates in the visited set
        if state_config in visited:
            continue
        visited.add(state_config)
        
        if test_goal(state):
            ##find the path
            path = []
            child_state = state
            while child_state.parent is not None:
                path.append(child_state.action)
                child_state = child_state.parent
            ##going from child to root, need other way, use reverse method on list
            path.reverse()
            time_end = time.process_time()
            time_run = time_end - time_start
            ram_usage = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - start_ram) / (2**20)
#            print(f"final path is {path}")
            ###insert write output and return
#            print(f"nodes expanded {nodes_expanded}")
            writeOutput(path, len(path), max_search_depth, nodes_expanded, time_run, ram_usage)
            return

        ###not goal state, now expand
        nodes_expanded+=1
        ### UDLR needs to get added in reverse order bc stack will pop last child first
        ### do not want that : instead want to push R,L,D,U so stack will pop U first
        ### use reversed function for iterator
        for child in reversed(state.expand()):
            child_config = tuple(child.config)
            ###check to see not in visited or already in the stack
            if (child_config not in visited) and (child_config not in in_dfs_frontier):
                dfs_frontier.append(child)
                in_dfs_frontier.add(child_config)
                if child.depth > max_search_depth:
                    max_search_depth = child.depth
#        print("last " + str(bfs_frontier.empty()))

def A_star_search(initial_state):
    """A * search"""
    ### STUDENT CODE GOES HERE ###
    a_frontier = Q.PriorityQueue()
    visited = set()
    in_a_frontier = {}
    max_search_depth = 0
    nodes_expanded = 0
    time_start = time.process_time()
    start_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    skipped = 0
    
    a_frontier.put((calculate_total_cost(initial_state),initial_state))
    in_a_frontier[tuple(initial_state.config)] = calculate_total_cost(initial_state)
    
    while not a_frontier.empty():
        p, state = a_frontier.get()
        # print(f"processing {state.config} -- {calculate_total_cost(state)}")
        state_config = tuple(state.config)
        if in_a_frontier.get(state_config) is None:
#            print(f"Node previously processed, ignore - {state.config}")
            skipped+=1
            continue
        if in_a_frontier.get(state_config) != p:
#            print(f"222 Node previously processed, ignore - {state.config}")
            skipped+=1
            continue
            
        del in_a_frontier[state_config]
        visited.add(state_config)
        depth = state.depth
        if depth > max_search_depth:
            max_search_depth = depth
        # print(f"Queue {a_frontier.qsize()}: {a_frontier.queue}")

        if test_goal(state):
            ##find the path
#            print(f" ===== >>> final {skipped} {len(visited)} ")
            path = []
            child_state = state
            while child_state.parent is not None:
                path.append(child_state.action)
                child_state = child_state.parent
            ##going from child to root, need other way, use reverse method on list
            path.reverse()
            time_end = time.process_time()
            time_run = time_end - time_start
            ram_usage = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - start_ram) / (2**20)
#            print(f"final path is {path}")
            ###insert write output and return
#            print(f"nodes expanded {nodes_expanded}")
            writeOutput(path, len(path), max_search_depth, nodes_expanded, time_run, ram_usage)
            return

        ###not goal state, now expand
        nodes_expanded+=1
        for child in state.expand():
            child_config = tuple(child.config)
            best_cost = in_a_frontier.get(child_config)
            ###check to see not in visited or already in the frontier
            if (child_config not in visited) and (best_cost is None):
                a_frontier.put((calculate_total_cost(child),child))
                in_a_frontier[child_config] = calculate_total_cost(child)
            elif (child_config not in visited) and (best_cost is not None):
                # check if child already exists in pq and if has higher priority there
                # if yes then add this one
                new_cost = calculate_total_cost(child)
                if best_cost > new_cost:
#                    print(f"Updated cost for {child_config} from {best_cost} to {new_cost}")
                    # same child config but has lower cost, so replace
                    a_frontier.put((new_cost, child))
                    in_a_frontier[child_config] = new_cost
                    


def calculate_total_cost(state):
    """calculate the total estimated cost of a state"""
    ### STUDENT CODE GOES HERE ###
    ### formula is g(n) + h(n) = f(n), where :
    ### g(n) : cost to node n and h(n) : cost from n to goal -- total manhattan dist
    g_n = state.depth
    h_n = 0
    for index,value in enumerate(state.config):
        if value != 0:
        ###possible use enumerate here??
            h_n += calculate_manhattan_dist(index, value, state.n)
    return g_n + h_n

def calculate_manhattan_dist(idx, value, n):
    """calculate the manhattan distance of a tile"""
    ### STUDENT CODE GOES HERE ###
    ### take idx and ideal index // n --> get row distance
    ### do same thing using % and get col distance, then add both distances
    row_dist = abs((idx//n) - (value//n))
    col_dist = abs((idx % n) - (value % n))
    return row_dist + col_dist

def test_goal(puzzle_state):
    """test the state is the goal state or not"""
    ### STUDENT CODE GOES HERE ###
#    print(puzzle_state.config == GOAL_STATE)
    GOAL_STATE = [0,1,2,3,4,5,6,7,8]
    return puzzle_state.config == GOAL_STATE

# Main Function that reads in Input and Runs corresponding Algorithm

def main():
    search_mode = sys.argv[1].lower()
    begin_state = sys.argv[2].split(",")
    begin_state = list(map(int, begin_state))
    board_size  = int(math.sqrt(len(begin_state)))
    hard_state  = PuzzleState(begin_state, board_size)
    start_time  = time.time()
    
    if   search_mode == "bfs": bfs_search(hard_state)
    elif search_mode == "dfs": dfs_search(hard_state)
    elif search_mode == "ast": A_star_search(hard_state)
    else:
        print("Enter valid command arguments !")
        
    end_time = time.time()
    print("Program completed in %.3f second(s)"%(end_time-start_time))

if __name__ == '__main__':
    main()
    

#def main2():
#    # Example usage:
#    # python puzzle.py 1,2,3,4,5,6,7,0,8
#    search_mode = sys.argv[1].lower()
#    print(search_mode)
#    begin_state = sys.argv[2].split(",")
#    begin_state = list(map(int, begin_state))
#    board_size  = int(math.sqrt(len(begin_state)))
#    hard_state  = PuzzleState(begin_state, board_size)
#    print("Initial:")

#    print(calculate_manhattan_dist(1, 8, board_size))

#    print("After expand")
#    children = hard_state.expand()
#    for idx, child in enumerate(children):
#        print(f"Child {idx} via action {child.action}:")
#        child.display()
    
#    print("After move down")
#    s1 = s1.move_down()
#    s1.display()
    
            


