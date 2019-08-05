import numpy as np
import matplotlib.pyplot as plt

from UAH007_02_01_Dynamic_programming import Grid, create_grid, print_value_policy, print_policy, random_policy

def random_windy(a, windy=0.5, possible_actions=('U', 'D', 'L', 'R')):
    p = np.random.random()

    if p > windy:
        return a
    else:
        actions = list(possible_actions)
        
        if a  in actions:
            actions.remove(a)
            
        return np.random.choice(actions)


def play_game(grid, policy, windy=0, gamma=0.9):

    # Restart the game to start at a random position
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    s = grid.current_state()
    states_and_rewards = [(s, 0)]
    while not grid.game_over():
        a = policy[s]
        a = random_windy(a, windy, grid.actions[s])
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s, r))

    # Calculation of the returns, the value of the terminal state is 0 by definition
    G = 0
    states_and_returns = []
    first = True

    for s, r in reversed(states_and_rewards):
        # It must ignore the first state and the last G since it does not correspond
        # to any movement
        if first:
            first = False
        else:
            states_and_returns.append((s, G))
        G = r + gamma*G

    # The states are rearranged
    states_and_returns.reverse()

    return states_and_returns


def init_v_returns():
    V = {}
    returns = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            returns[s] = []
        else:
            V[s] = 0

    return V, returns


def play_game_es(grid, policy, gamma=0.9):

    # Restart the game to start at a random position
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    s = grid.current_state()

    # The first action is random
    a = np.random.choice(Grid.possible_actions)

    # A triplet of actions and rewards states is created.
    states_actions_rewards = [(s, a, 0)]
    seen_states = set()
    seen_states.add(grid.current_state())
    num_steps = 0
    while True:
        r = grid.move(a)
        num_steps += 1
        s = grid.current_state()

        if s in seen_states:
            # In order not to end up in an infinite episode this penalty is
            # added for falling on the wall.
            reward = -10. / num_steps
            states_actions_rewards.append((s, None, reward))
            break
        elif grid.game_over():
            states_actions_rewards.append((s, None, r))
            break
        else:
            a = policy[s]
            states_actions_rewards.append((s, a, r))
        seen_states.add(s)

    # Calculation of the returns, the value of the terminal state is 0 by definition
    G = 0
    states_actions_returns = []
    first = True
    for s, a, r in reversed(states_actions_rewards):
        if first:
            first = False
        else:
            states_actions_returns.append((s, a, G))
        G = r + gamma*G

    states_actions_returns.reverse()

    return states_actions_returns


def max_dict(d):
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val


def init_q_returns(grid):
    Q = {}
    returns = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            Q[s] = {}
            for a in Grid.possible_actions:
                Q[s][a] = 0
                returns[(s, a)] = []
        else:
            pass

    return Q, returns


def play_game_no_es(grid, policy,  windy=0.1,gamma=0.9):
    # The game starts in a known state
    s = (2, 0)
    grid.set_state(s)
    a = random_windy(policy[s], windy, grid.actions[s])

    # Now each triplet is s(t), a(t), r(t)
    # r(t) is obtained from the action a(t-1) of s(t-1) and arrive at s(t)
    states_actions_rewards = [(s, a, 0)]
    while True:
        r = grid.move(a)
        s = grid.current_state()
        if grid.game_over():
            states_actions_rewards.append((s, None, r))
            break
        else:
            a = random_windy(policy[s], windy, grid.actions[s])
            states_actions_rewards.append((s, a, r))

    # Calculation of the returns, the value of the terminal state is 0 by definition
    G = 0
    states_actions_returns = []
    first = True
    for s, a, r in reversed(states_actions_rewards):
        # It must ignore the first state and the last G since it does not correspond
        # to any movement
        if first:
            first = False
        else:
            states_actions_returns.append((s, a, G))
        G = r + gamma*G

    # The states are rearranged
    states_actions_returns.reverse()

    return states_actions_returns
