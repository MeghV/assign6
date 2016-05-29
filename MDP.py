'''MDP.py
S. Tanimoto, May 2016.

Megh Vakharia
meghv, 1220592

Provides representations for Markov Decision Processes, plus
functionality for running the transitions.

The transition function should be a function of three arguments:
T(s, a, sp), where s and sp are states and a is an action.
The reward function should also be a function of the three same
arguments.  However, its return value is not a probability but
a numeric reward value -- any real number.

operators:  state-space search objects consisting of a precondition
 and deterministic state-transformation function.
 We assume these are in the "QUIET" format used in earlier assignments.

actions:  objects (for us just Python strings) that are 
 stochastically mapped into operators at runtime according 
 to the Transition function.
'''
import random
import itertools
import operator

REPORTING = True

class MDP:
    def __init__(self):
        self.known_states = set() # unordered set of known states
        self.succ = {} # hash of adjacency lists by state.

    # Sets the start state of the game.
    # start_state = coordinate pair
    def register_start_state(self, start_state):
        self.start_state = start_state
        self.known_states.add(start_state)

    # Stores all available actions.
    def register_actions(self, action_list):
        self.actions = action_list

    # Stores the operators that make moves,
    def register_operators(self, op_list):
        self.ops = op_list

    # Registers function to use to determine transition 
    # probabilities. 
    # transition_function = def T(state, action, s')
    def register_transition_function(self, transition_function):
        self.T = transition_function

    # Registers function to use to determine rewards
    # for movement to a state.
    # reward_function = R(s, a, s')
    def register_reward_function(self, reward_function):
        self.R = reward_function


    def state_neighbors(self, state):
        '''Return a list of the successors of state.  First check
           in the hash self.succ for these.  If there is no list for
           this state, then construct and save it.
           And then return the neighbors.'''
        neighbors = self.succ.get(state, False)
        if neighbors==False:
            neighbors = [op.apply(state) for op in self.ops if op.is_applicable(state)]
            self.succ[state]=neighbors
            self.known_states.update(neighbors)
        return neighbors

    def generateAllStates(self):
        '''Produces a set of all of the states in any MDP
           and stores states in member variable known_states.
           Uses iterative depth-first search.'''
        self.known_states = set()

        if REPORTING: print("Generating all states with BFS")
        OPEN_STATES = [self.start_state]

        while OPEN_STATES != []:
            S = [OPEN_STATES[0]]
            del OPEN_STATES[0]
            self.known_states.update(S)
            if REPORTING:
                print("Next state: " + str(S))
            L = []
            neighbors = [op.apply(S[0]) for op in self.ops if op.is_applicable(S[0])]
            print("State neighbors: " + str(neighbors))
            for sp in neighbors:
                if sp not in self.known_states:
                    L.append(sp)
            # Removes duplicates
            for sp in L:
                if sp in OPEN_STATES:
                    if REPORTING: print(str(sp) + " already known.")
                    OPEN_STATES.remove(sp)
            OPEN_STATES = OPEN_STATES + L
            if REPORTING: print("To Explore: " + str(OPEN_STATES) + "\n")
        if REPORTING: print("Found states: " + str(self.known_states) + "\n")

    def extractPolicy(self):
        '''Extracts optimal policy for MDP and places state:value
           mappings into optPolicy. This must be run after Q-Values 
           have already been obtained to some extent.'''
        self.optPolicy = {s:0 for s in self.known_states}
        for s in self.optPolicy:
            s_q_values = {sa: v for sa,v in self.QValues.items() if sa[0] == s}
            # If the Q-value hasn't already been obtained, its value is set to 0
            if len(s_q_values) > 0:
                self.optPolicy[s] = max([v for sa, v in s_q_values.items()])
            else:
                self.optPolicy[s] = 0
        if REPORTING: print("\n================================================\nExtracted optimal policy:\n" + \
                            str(self.optPolicy) + "\n================================================\n")


    def QLearning(self, discount, nEpisodes, epsilon):
        '''Initiates Q-Learning for nEpisodes starting from the start state, 
           running each episode until a "DEAD" state is reached. Over many episodes, 
           the Q-value updates will degrade according to 1/N, with N representing 
           the number of times an action has been taken from a state. Q-Values will 
           be stored in member variable QValues.'''
        self.QValues = {(s,a):0 for (s, a) in itertools.product(self.known_states, self.actions)}
        
        # Filter out invalid moves
        for k in [sa for sa in self.QValues.keys() if not self.can_move(sa[0],sa[1])]:
            del self.QValues[k]

        # Holds N(s,a) to be used in alpha
        self.N = {k:v for k,v in self.QValues.items()}

        # Returns (state, action) tuples for a given state
        next_actions = lambda s: [(state_action) for state_action in self.QValues.keys() if state_action[0] == s]

        episode = 0
        while episode < nEpisodes:
            if REPORTING: print("\n========================\n" +"Episode: " + str(episode+1) + "\n========================\n")

            # Begin with start state and go until 'DEAD' state
            self.current_state = self.start_state
            while self.current_state != 'DEAD':
                # Deepcopy current state
                s = (self.current_state[0], self.current_state[1])
                # Looks at possible actions and adds the following probabilites to actions:
                #   * 1 - alpha to optimal action (highest Q-value)
                #   * alpha to randomly choose an action
                possible_actions = next_actions(self.current_state)
                action_values = {action: self.QValues[action] for action in possible_actions}
                if len(action_values) > 1:
                    for k,v in action_values.items():
                        action_values[k] = epsilon / (len(action_values) - 1)
                    action_values[max(action_values.keys(), key=lambda k:action_values[k])] = 1 - epsilon
                else:
                    action_values[max(action_values.keys(), key=lambda k:action_values[k])] = 1

                # Chooses an action using probability of exploration vs. exploitation (epsilon)
                threshold = 0.0
                rnd = random.uniform(0.0, 1.0)
                r = 0
                for action, prob in action_values.items():
                    threshold += prob
                    if threshold>rnd:
                        chosen_a = action[1]
                        break
                
                if REPORTING:
                    print("\nCurrent State: " + str(s) + "\n========================\n")
                    print("Action Probabilities: " + str(action_values))
                    print("Chosen Action: " + str(chosen_a) + " (N=" + str(self.N[(s, chosen_a)]) + ")")

                # Take action and take note of distribution (running average)
                # and reward
                r, sp = self.take_action(chosen_a)
                self.N[(s, chosen_a)] += 1
                
                # Calculate sample (s, a, r, s') of next state using
                # reward and discounted optimal Q-value:
                # sample = R(s,a,s') + discount * max(Q(s',a'))
                if sp == 'DEAD':
                    sample = r
                else:
                    sample = r + discount * max(value for state_action, value in self.QValues.items() if state_action[0] == sp)
                # Update running Q-value averages by using new sample of data
                alpha = 1/self.N[(s, chosen_a)]
                self.QValues[(s, chosen_a)] = (1 - alpha)*self.QValues[(s, chosen_a)] \
                                                                    + (alpha)*sample
                if REPORTING: 
                    print("Updated Value for (" + str(s) + ", " + str(chosen_a) + "): " +  str(self.QValues[(s, chosen_a)]))
                # Keep running for nEpisodes until "DEAD" state reached
            episode+=1
        if REPORTING: print("Q-Values:\n" + str({sa:v for sa, v in self.QValues.items()}))

    def ValueIterations(self, discount, niterations):
        '''Performs up to niterations of Bellman updates to get an
           approximation of V*, which are stored in member variable V.'''
        # Initial value for every state = 0
        self.V = {}
        for s in self.known_states:
            self.V[s] = 0

        k = 1
        while k <= niterations:
            # starting with step = 0, value = 0
            # work way up to step = niterations + 1
            values_at_k = {s: 0 for s in self.known_states}
            for s, s_value in self.V.items():
                if REPORTING: print("\n========================\n"+"Current State: " \
                                    + str(s) + "\n========================\n")
                sp_states = self.state_neighbors(s)
                
                # Set to hold discounted sum of rewards
                # for each possible action
                action_discounted_rewards = set()

                # For each possible action, calculate the
                # probability and reward for state -> a ->s'
                for action in self.actions:
                    if self.can_move(s, action):
                        if REPORTING: print("Action: " + str(action) + "\n")
                        action_reward = 0
                        for sp in sp_states:
                            probability = self.T(s, action, sp)
                            state_reward = self.R(s, action, sp)
                            print("State: " +str(sp))
                            print("Known: " + str(self.known_states))
                            print("V: " + str(self.V))
                            discounted_reward = discount * self.V[sp]
                            if REPORTING:
                                print("Probability: " + str(probability))
                                print("Reward: " + str(state_reward))
                                print("State:" + str(sp))
                                print("Discounted Reward: " + str(discounted_reward) + "\n")
                            # Add to sum of discounted rewards for given action
                            action_reward += probability * (state_reward + discounted_reward)
                        action_discounted_rewards.add(action_reward)
                        if REPORTING: print("Total Discounted Sum for Action: " + str(action_reward) + "\n========================\n")
                
                # Set the value of k + 1 state to expected discounted sum of rewards
                # as per Bellman equation
                update_number = max(action_discounted_rewards) if len(action_discounted_rewards) > 0 else 0
                values_at_k[s] = update_number
                if REPORTING: print("Updated Value for " + str(s) + ": " + str(update_number) + "\n")
            self.V = values_at_k
            k+=1
        if REPORTING:
            for s, s_value in self.V.items():
                print("Values:\n"+str(s) + ": " + str(s_value) + "\n")


    def can_move(self, s, action):
        '''Takes a state and action and returns whether the 
           action is valid based on the state.'''
        if action=="End":
            if s not in [(3,2), (3,1)]: return False
            else: return True
        if s=="DEAD" or s in [(3,2), (3,1)]: return False
        if action=="North": dx, dy = 0, 1
        if action=="South": dx, dy = 0, -1
        if action=="West":  dx, dy = -1, 0
        if action=="East":  dx, dy = 1, 0
        x, y = s
        # If the change in movement results in invalid move,
        # return False
        if x+dx < 0 or x+dx > 3: return False
        if y+dy < 0 or y+dy > 2: return False
        if x+dx==1 and y+dy==1: return False # Can't move into the rock.
        return True

    # Randomly moves agent nsteps from starting step
    # choosing a random action at each state.
    # nsteps = number of steps to explore
    def random_episode(self, nsteps):
        self.current_state = self.start_state
        self.current_reward = 0.0
        for i in range(nsteps):
            self.take_action(random.choice(self.actions))
            if self.current_state == 'DEAD':
                print('Terminating at DEAD state.')
                break
        if REPORTING: print("Done with "+str(i)+" of random exploration.")

    def take_action(self, a):
        s = self.current_state
        neighbors = self.state_neighbors(s)
        threshold = 0.0
        rnd = random.uniform(0.0, 1.0)
        r = 0
        for sp in neighbors:
            threshold += self.T(s, a, sp)
            if threshold>rnd:
                r = self.R(s, a, sp)
                break
        self.current_state = sp
        if REPORTING: print("After action "+a+", moving to state "+str(sp)+\
                            "; reward is "+str(r))
        return r, sp

