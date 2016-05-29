grid_MDP.QValues = {(s,a):0 for (s, a) in itertools.product(grid_MDP.known_states, grid_MDP.actions)}

starting_actions = [(state_action) for state_action in grid_MDP.QValues.keys if state_action[0] == grid_MDP.start_state]

(state_action) for state_action in grid_MDP.QValues.keys if state_action[0] == grid_MDP.start_state

max(value for state_action, value in grid_MDP.QValues.items() if state_action[0] == (0,0))

 mdp.QValues = {(s,a):0 for (s, a) in itertools.product(mdp.known_states, mdp.actions)}

remove = [sa for sa in mdp.QValues.keys() if not mdp.can_move(sa[0],sa[1])]

next_actions = lambda s: [(state_action) for state_action in mdp.QValues.keys() if state_action[0] == s]
starting_actions = next_actions((2,1))
action_values = {action: mdp.QValues[action] for action in starting_actions}
for k,v in action_values.items():
	action_values[k] = epsilon / (len(action_values) - 1)

action_values[max(action_values.keys(), key=lambda k:action_values[k])] = 1 - epsilon