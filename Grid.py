'''Grid.py
Create the Grid World MDP used in many of the Berkeley lectures.

Also run a test consisting of one episode of random exploration
in this world, receiving the rewards along the way.

S. Tanimoto, May 14, 2016.

Megh Vakharia
meghv, 1220592

'''

ACTIONS = ['North', 'South', 'East', 'West', 'End']

INITIAL_STATE = (0,0)

# Operators are the actual state-space search operators as used
# in classical algorithms such as A* search.
import math

class Operator:

  def __init__(self, name, precond, state_transf):
    self.name = name
    self.precond = precond
    self.state_transf = state_transf

  def is_applicable(self, s):
    return self.precond(s)

  def apply(self, s):
    return self.state_transf(s)

NorthOp = Operator("Move North if Possible",\
                   lambda s: can_move(s, 0,1),\
                   lambda s: move(s, 0,1))

SouthOp = Operator("Move South if Possible",\
                   lambda s: can_move(s, 0,-1),\
                   lambda s: move(s, 0,-1))

WestOp  = Operator("Move West if Possible",\
                   lambda s: can_move(s, -1, 0),\
                   lambda s: move(s, -1, 0))

EastOp  = Operator("Move East if Possible",\
                   lambda s: can_move(s, 1, 0),\
                   lambda s: move(s, 1, 0))

EndOp  = Operator("Go to the DEAD state",\
                   lambda s: s==(3,2) or s==(3,1),\
                   lambda s: "DEAD")

OPERATORS = [NorthOp, SouthOp, WestOp, EastOp, EndOp]

# The following dictionary maps each action (except the End action)
# to the three operators that might be randomly chosen to perform it.
# In this MDP, the first gets probability P_normal, and the other two
# each get probability P_noise.

# Possible resulting actions for each attempted action
ActionOps = {'North': [NorthOp, WestOp, EastOp],
             'South': [SouthOp, EastOp, WestOp],
             'East':  [EastOp, SouthOp, NorthOp],
             'West':  [WestOp, NorthOp, SouthOp]}

# Here's the helper function for defining operator preconditions:
# Updated as per suggestion from Tyler Williamson
def can_move(s, dx, dy):
    if s=="DEAD" or s in [(3,2), (3,1)]: return False
    x, y = s
    # If the change in movement results in invalid move,
    # return False
    if x+dx < 0 or x+dx > 3: return False
    if y+dy < 0 or y+dy > 2: return False
    if x+dx==1 and y+dy==1: return False # Can't move into the rock.
    return True

# Here's the corresponding helper function for defining operator
# state transition functions:
def move(s, dx, dy):
    x, y = s
    return (x+dx, y+dy)

P_normal = 0.8   # As used in the example by Dan Klein and Pieter Abbeel.
P_noise  = 0.1

def T(s, a, sp):
    '''Compute the transition probability for going from state s to
       state sp after taking action a.  This could have been implemented
       using a big dictionary, but this looks more easily generalizable
       to larger grid worlds.'''
    if s=="DEAD": return 0
    if sp=="DEAD":
      if s==(3,2) or s==(3,1): return 1 # 100% probability of moving to DEAD state when in pit / gem
    sx, sy = s
    spx, spy = sp
    # If moving North from current state
    if sx==spx and sy == spy-1:
        if a=="North": return P_normal # 80% chance of moving north
        if a=="East" or a=="West": return P_noise # 10% chance of moving east / west
    # If moving South from current state
    if sx==spx and sy == spy+1:
        if a=="South": return P_normal
        if a=="East" or a=="West": return P_noise
    # If moving East
    if sx==spx-1 and sy == spy:
        if a=="East": return P_normal
        if a=="North" or a=="South": return P_noise
    # If moving West
    if sx==spx+1 and sy == spy:
        if a=="West": return P_normal
        if a=="North" or a=="South": return P_noise
    if s==sp:  # This means precondition was not satisfied, in most problem formulations.#
        # Go through the 3 relevant operators in order of highest-probability first, and
        # total up the probabilities of those whose preconditions were not satisfied.
        ops = ActionOps[a]
        prob = 0.0
        if not ops[0].is_applicable(s): prob += P_normal
        if not ops[1].is_applicable(s): prob += P_noise
        if not ops[2].is_applicable(s): prob += P_noise
        return prob
    return 0.0 # Default case is probability 0.

def R(s, a, sp):
    '''Return the reward associated with transitioning from s to sp via action a.'''
    if s=='DEAD': return 0
    if s==(3,2): return 1.0  # the Gem
    if s==(3,1): return -1.0 # the Pit
    return -0.01   # cost of living.

import MDP

GRID_WIDTH = 20 # General width of each square in grid with '-'
GRID_HEIGHT = 7 # General height of each square in grid with '|'

# Drawing Functions
def build_horizontal_rule(cols):
    '''Creates an ASCII horizontal line to be used
       in constructing a grid. Each line is 22 characters
       long, starting and ending with "+".'''
    row_str = "+"
    for c in range(0, cols):
        row_str += "-"*GRID_WIDTH + "+"
    row_str += "\n"
    return row_str

def build_vertical_lines(cols):
    '''Creates an ASCII vertical line to be used
      in constructing a grid.'''
    row_str = "|"
    for c in range(0, cols):
        row_str +=" "*GRID_WIDTH + "|"
    row_str += "\n"
    return row_str

def draw_grid(rows, cols):
    '''Creates an empty ASCII grid.'''
    hr = build_horizontal_rule(cols)
    vr = build_vertical_lines(cols)
    grid = hr
    for r in range(0, rows):
        for i in range(0,GRID_HEIGHT):
            grid += vr
        grid += hr
    return grid

def middle_value_in_grid(value):
  '''Creates a padded row for a grid box to
  center the value.'''
  grid_str = ""
  next_num = str(round(value, 3))
  num_width = len(next_num)
  left_padding = math.ceil((GRID_WIDTH-num_width) / 2)
  right_padding = math.floor((GRID_WIDTH-num_width) / 2)
  grid_str += " "*left_padding + next_num + " "*right_padding
  return grid_str

def draw_grid_with_V_values(V, rows, cols):
    '''Draws a grid using inputted V values with
       states corresponding to cartesian coordinates.'''
    hr = build_horizontal_rule(cols)
    grid_str = hr
    for r in range(rows - 1, -1, -1):
        for height in range(0, GRID_HEIGHT):
            grid_str +="|"
            for c in range(0, cols):
                if height == 3 and (c,r) in V:
                    grid_str += middle_value_in_grid(V[(c,r)])
                else:
                    grid_str += " "*GRID_WIDTH
                grid_str += "|"
            grid_str += "\n"
        grid_str += hr
    print(grid_str)

def draw_grid_with_Q_values(Q, rows, cols):
    '''Draws a grid with Q-values in each box corresponding to the following
       actions: north, south, east, west, end.'''
    hr = build_horizontal_rule(cols)
    grid_str = hr
    for r in range(rows - 1, -1, -1):
        for height in range(0, GRID_HEIGHT):
            grid_str +="|"
            for c in range(0, cols):
                sa = {sa[1]:v for sa, v in Q.items() if sa[0] == (c,r)}
                if height == 0 and "North" in sa.keys():
                  grid_str += middle_value_in_grid(sa["North"])
                elif height == 6 and "South" in sa.keys():
                  grid_str += middle_value_in_grid(sa["South"])
                elif height == 3 and "End" in sa.keys():
                  grid_str += middle_value_in_grid(sa["End"])
                elif height == 3:
                  east_width, west_width = 0,0
                  if "West" in sa.keys():
                    west_num = str(round(sa["West"],3))
                    west_width = len(west_num)
                    grid_str+=west_num
                  if "East" in sa.keys():
                    east_num = str(round(sa["East"],3))
                    east_width = len(east_num)
                    grid_str+= ' '*(GRID_WIDTH-west_width-east_width) + east_num
                  else:
                    grid_str += ' '*(GRID_WIDTH-west_width-east_width)
                else:
                    grid_str += " "*GRID_WIDTH
                grid_str += "|"
            grid_str += "\n"
        grid_str += hr
    print(grid_str)

def extractPolicy(policy):
  '''Extracts the optimal policy from an MDP and
     draws a grid with optimal V-values for each state.'''
  draw_grid_with_V_values(policy, 3, 4)

def test():
    '''Create the MDP, then run an episode of random actions for 10 steps.'''
    global grid_MDP
    grid_MDP = MDP.MDP()
    grid_MDP.register_start_state((0,0))
    grid_MDP.register_actions(ACTIONS)
    grid_MDP.register_operators(OPERATORS)
    grid_MDP.register_transition_function(T)
    grid_MDP.register_reward_function(R)
    grid_MDP.random_episode(100)
    grid_MDP.generateAllStates()
    grid_MDP.ValueIterations(0.9, 100)
    draw_grid_with_V_values(grid_MDP.V, 3, 4)
    grid_MDP.QLearning(0.8, 50, 0.8)
    draw_grid_with_Q_values(grid_MDP.QValues, 3, 4)
    grid_MDP.extractPolicy()
    extractPolicy(grid_MDP.optPolicy)
    return grid_MDP

#test()
                  
