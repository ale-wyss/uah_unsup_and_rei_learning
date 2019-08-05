
# coding: utf-8

import numpy as np


class Grid:
    def __init__(self, width, height, start):
        self.width = width
        self.height = height
        self.i = start[0]
        self.j = start[1]

    # Class properties
    possible_actions = ('U', 'D', 'L', 'R')

    def set(self, rewards, actions):
        # rewards is a dictionary: (i, j): r (row, col): rewards
        # actions is a dictionary: (i, j): A (row, col): list of possible actions

        self.rewards = rewards
        self.actions = actions

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]

    def get_actions(self):
        return self.actions[(self.i, self.j)]

    def current_state(self):
        return (self.i, self.j)

    def is_terminal(self, s):
        return s not in self.actions

    def move(self, action):
        # Check if the move is legal valid
        if action in self.actions[(self.i, self.j)]:
            if action == 'U':
                self.i -= 1
            elif action == 'D':
                self.i += 1
            elif action == 'R':
                self.j += 1
            elif action == 'L':
                self.j -= 1

        # Returns the reward
        return self.rewards.get((self.i, self.j), 0)

    def undo_move(self, action):
        # Undo the movement
        if action == 'U':
            self.i += 1
        elif action == 'D':
            self.i -= 1
        elif action == 'R':
            self.j -= 1
        elif action == 'L':
            self.j += 1

        # Throw an exception in the cell in invalid.
        assert(self.current_state() in self.all_states())

    def game_over(self):
        # Returns true when the game is over and false otherwise.
        return (self.i, self.j) not in self.actions

    def all_states(self):
        # Returns all states
        return set(self.actions.keys()) | set(self.rewards.keys())


def create_grid(step_cost=0):
    # Create a new board with the reward and possible actions in the cells.
    # Default each movement does not have a cost.
    #
    # The board is as follows
    #   S initial point
    #   X not allowed cell
    #   . allowed cell
    #
    # Number indicate the rewards of states
    #
    #
    # .  .  .  1
    # .  x  . -1
    # s  .  .  .

    grid = Grid(3, 4, (2, 0))

    rewards = {
        (0, 0): step_cost,
        (0, 1): step_cost,
        (0, 2): step_cost,
        (0, 3): 1,
        (1, 0): step_cost,
        (1, 2): step_cost,
        (1, 3): -1,
        (2, 0): step_cost,
        (2, 1): step_cost,
        (2, 2): step_cost,
        (2, 3): step_cost}

    actions = {
        (0, 0): ('D', 'R'),
        (0, 1): ('L', 'R'),
        (0, 2): ('L', 'D', 'R'),
        (1, 0): ('U', 'D'),
        (1, 2): ('U', 'D', 'R'),
        (2, 0): ('U', 'R'),
        (2, 1): ('L', 'R'),
        (2, 2): ('L', 'R', 'U'),
        (2, 3): ('L', 'U')
    }

    grid.set(rewards, actions)

    return grid


def print_values(value, grid):
    for i in range(grid.width):
        print("---------------------------")
        for j in range(grid.height):
            v = value.get((i, j), 0)
            if v >= 0:
                print(" %.2f|" % v, end="")
            else:
                print("%.2f|" % v, end="")
        print("")


def print_policy(policy, grid):
    for i in range(grid.width):
        print("---------------------------")
        for j in range(grid.height):
            action = policy.get((i, j), ' ')

            if action == 'U':
                print("  ↑  |", end="")
            elif action == 'D':
                print("  ↓  |", end="")
            elif action == 'R':
                print("  →  |", end="")
            elif action == 'L':
                print("  ←  |", end="")
            else:
                print("     |", end="")
        print("")


def print_value_policy(V, policy, grid):
    print("Value function")
    print_values(V, grid)
    print()
    print("Policy")
    print_policy(policy, grid)


def init_states(grid, value=None):
    states = grid.all_states()
    V = {}

    for s in states:
        if s in grid.actions:
            if value is None:
                V[s] = np.random.random()
            else:
                V[s] = value
        else:
            V[s] = 0

    return V, states


def policy_evaluation(grid, policy, V, states, gamma=1, max_iter=100, threshold=1e-3):
    num_iter = 0

    while num_iter < max_iter:
        num_iter += 1
        biggest_change = 0
        for s in states:
            old_v = V[s]

            if s in policy:
                action = policy[s]
                grid.set_state(s)
                r = grid.move(action)
                V[s] = r + gamma * V[grid.current_state()]
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))

        if biggest_change < threshold:
            break

    if num_iter >= max_iter:
        print("The maximum number of iterations has been reached")


def optimal_policy(grid, policy, gamma=1, max_iter=100, threshold=1e-3):

    # Initialize the funciton value and states
    V, states = init_states(grid)

    # Iterate over policies until convergence or maximum iterations
    num_iter = 0

    while num_iter < max_iter:
        num_iter += 1

        # Policy evaluation
        policy_evaluation(grid, policy, V, states, gamma, max_iter, threshold)

        # Improvement the policy
        is_policy_converged = True
        for s in states:
            if s in policy:
                old_a = policy[s]
                new_a = None
                best_value = float('-inf')

                # Iterate over actions until the best is obtained
                for a in Grid.possible_actions:
                    grid.set_state(s)
                    r = grid.move(a)
                    v = r + gamma * V[grid.current_state()]
                    if v > best_value:
                        best_value = v
                        new_a = a
                policy[s] = new_a

                if new_a != old_a:
                    is_policy_converged = False

        if is_policy_converged:
            break

    if num_iter >= max_iter:
        print("The maximum number of iterations has been reached")

    return V


def random_policy(grid):
    policy = {}

    for s in grid.actions.keys():
        policy[s] = np.random.choice(grid.actions[s])

    return policy


def policy_evaluation_windy(grid, policy, V, states, windy=0.5, gamma=1, max_iter=100, threshold=1e-3):
    num_iter = 0

    while num_iter < max_iter:
        num_iter += 1
        biggest_change = 0
        for s in states:
            old_v = V[s]
            new_v = 0

            if s in policy:
                for action in grid.actions[s]:
                    if action == policy[s]:
                        p = windy
                    else:
                        p = (1-windy)/(len(grid.actions[s])-1)
                    grid.set_state(s)
                    r = grid.move(action)
                    new_v += p * (r + gamma * V[grid.current_state()])

                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))

        if biggest_change < threshold:
            break

    if num_iter >= max_iter:
        print("The maximum number of iterations has been reached")


def optimal_policy_windy(grid, policy, windy=0.5, gamma=1, max_iter=100, threshold=1e-3):

    # Initialize the funciton value and states
    V, states = init_states(grid)

    # Iterate over policies until convergence or maximum iterations
    num_iter = 0

    while num_iter < max_iter:
        num_iter += 1

        # Policy evaluation
        policy_evaluation_windy(grid, policy, V, states,
                                windy, gamma, max_iter, threshold)

        # Improvement the policy
        is_policy_converged = True
        for s in states:
            if s in policy:
                old_a = policy[s]
                new_a = None
                best_value = float('-inf')

                # Iterate over actions until the best is obtained
                for action in Grid.possible_actions:
                    v = 0
                    for w_action in grid.actions[s]:
                        if action == w_action:
                            p = windy
                        else:
                            p = (1-windy)/(len(grid.actions[s])-1)
                        grid.set_state(s)
                        r = grid.move(action)
                        v += p * (r + gamma * V[grid.current_state()])
                    if v > best_value:
                        best_value = v
                        new_a = action
                policy[s] = new_a

                if new_a != old_a:
                    is_policy_converged = False

        if is_policy_converged:
            break

    if num_iter >= max_iter:
        print("The maximum number of iterations has been reached")

    return V


def optimal_value(grid, policy, gamma=0.9, max_iter=100, threshold=1e-3):
    V, states = init_states(grid)

    num_iter = 0
    while num_iter < max_iter:
        num_iter += 1
        biggest_change = 0

        for s in states:
            old_v = V[s]

            # La función de valor solo tiene valor si no es final
            if s in policy:
                new_v = float('-inf')
                for a in grid.actions[s]:
                    grid.set_state(s)
                    r = grid.move(a)
                    v = r + gamma * V[grid.current_state()]
                    if v > new_v:
                        new_v = v
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))

        if biggest_change < threshold:
            break

    if num_iter >= max_iter:
        print("The maximum number of iterations has been reached")

    # Obtener la politica para la función de valor
    for s in policy.keys():
        best_a = None
        best_value = float('-inf')
        # Itera sobre todas las posibles acciones
        for a in Grid.possible_actions:
            grid.set_state(s)
            r = grid.move(a)
            v = r + gamma * V[grid.current_state()]
            if v > best_value:
                best_value = v
                best_a = a

        policy[s] = best_a

    return V
