import numpy as np
import utils
import random
import math
import copy


class Agent:

    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        self.reset()

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    def discretize_state(self, state):
        snake_head_x, snake_head_y, snake_body, food_x, food_y = state
        # Determine adjoining walls
        def adjoining_wall(coord, min_val, max_val):
            if int(coord) <= int(min_val):
                return 1
            elif int(coord) >= (max_val):
                return 2
            else:
                return 0

        adjoining_wall_x = adjoining_wall(snake_head_x, utils.WALL_SIZE, utils.DISPLAY_SIZE-utils.WALL_SIZE)
        adjoining_wall_y = adjoining_wall(snake_head_y, utils.WALL_SIZE, utils.DISPLAY_SIZE-utils.WALL_SIZE)

        # Determine food direction
        def direction_delta(head, food):
            if int(food) > int(head):
                return 2
            elif int(food) < int(head):
                return 1
            else:
                return 0

        food_dir_x = direction_delta(snake_head_x, food_x)
        food_dir_y = direction_delta(snake_head_y, food_y)

        adjacents = [
            (snake_head_x, snake_head_y - utils.GRID_SIZE),  # Up
            (snake_head_x, snake_head_y + utils.GRID_SIZE),  # Down
            (snake_head_x - utils.GRID_SIZE, snake_head_y),  # Left
            (snake_head_x + utils.GRID_SIZE, snake_head_y)   # Right
        ]
        adjoining_body = [1 if (x, y) in snake_body else 0 for x, y in adjacents]
        return (adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, *adjoining_body)

    def calculate_reward(self, curr_points, last_points, dead):
        """
        :param curr_points: float, the current points
        :param last_points: float, the last points
        :param dead: bool, if the snake is dead
        :return: float, the reward value
        """
        reward = -0.1
        if curr_points - last_points > 0:
            reward = 1
        elif dead:
            reward = -1
        return reward

    def calculate_q_value(self, last_move_state, last_move_action, curr_state, dead, curr_points):
        """
        :param last_move_state: list, the last move state
        :param last_move_action: int, the last move action 
        :param curr_state: list, the current state 
        :param dead: bool, if the snake is dead
        :param curr_points: float, the current points
        :return: float, the updated Q value

        Calculate the updated Q value using the Q learning formula
        Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a))
        """
        last_move = self.discretize_state(last_move_state)
        reward = self.calculate_reward(curr_points, self.points, dead)
        curr_discrete_state = self.discretize_state(curr_state)

        q_values = [self.Q[curr_discrete_state][action] for action in self.actions]
        max_q = max(q_values)
        alpha = self.C / (self.C + self.N[last_move][last_move_action])

        last_move_q_val = self.Q[last_move][last_move_action]
        delta = alpha * (reward + self.gamma * max_q - last_move_q_val)
        return last_move_q_val + delta
    
    def exploration_function(self, u, n):
        """
        :param u: float, Q value
        :param n: int, N value
        :return: float, the exploration function value
        """
        if n < self.Ne and self._train:
            return 1.0
        else:
            return u

    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately
        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)
        '''
        state_copy = copy.deepcopy(state)

        curr_discrete_state = self.discretize_state(state)
        # Get the last move discrete state if the current move is not the first move
        if self.s is not None:
            last_move_state = self.discretize_state(self.s)

        if dead:
            # Update the Q value if it is not the first move 
            if self.s is not None:
                self.Q[last_move_state][self.a] = self.calculate_q_value(self.s, self.a, state, dead, points)
            self.reset()
            return None

        # Update the Q value of the last move if it is not the first move and the agent is in training mode
        if not (self.s is None or self.a is None) and self._train:
            self.Q[last_move_state][self.a] = self.calculate_q_value(self.s, self.a, state, dead, points)

        # Find the action with the highest Q value
        q_values = [self.exploration_function(self.Q[curr_discrete_state][action], self.N[curr_discrete_state][action]) for action in self.actions]
        q_values = q_values[::-1]
        action_index = len(q_values) - np.argmax(q_values) -1
        action = self.actions[action_index]

        # Update the previous exploitation count, state, action & points
        self.N[curr_discrete_state][action] += 1
        self.s = state_copy
        self.a = action
        self.points = points
        return action