"""
Base environment class
"""

# For get random choice
import random

# For work with gym environment
import gym

# Import data
from ..data import Maps, Actions, States, RenderModes, RenderObjectTypes
# Import render functions
from ..utils import render_functions


class Base(gym.Env):
    """ Base frozen lake environment functional """

    def __init__(self, map_name, action_set_name, state_set_name, max_episode_size):
        # type: (str, str, str, int) -> None
        """
        :param map_name: Str, name of map (on which map we play)
        :param action_set_name: Str, map of action set (with which actions we play)
        :param state_set_name: Str, name of state set (
        :param max_episode_size: Int, max episode size for skip collisions
        """

        # Inheritance base gym class
        super(Base, self).__init__()

        # Save max episode size
        self.max_episode_size = max_episode_size

        # Initialize text map, action and state sets
        self.text_map = Maps.get(map_name)
        self.action_set = Actions.get(action_set_name)
        self.state_set = States.get(state_set_name)

        # Calculate map params (map size in any variants)
        self.__calculate_map_params()
        # Indexing map states
        self.__index_map()
        # Initialize environment spaces
        self.__initialize_spaces()
        # Generate transition matrix
        self.__generate_transition_matrix()

    def __calculate_map_params(self):
        """ Initialize map params (e.g sizes) """
        self.amount_rows = len(self.text_map)
        self.amount_columns = len(self.text_map[0])
        self.shape = (self.amount_rows, self.amount_columns)

    def __initialize_spaces(self):
        """ Initialize action and observation spaces """
        self.action_space = gym.spaces.Discrete(len(self.action_set))
        self.observation_space = gym.spaces.Discrete(len(self.index_map))

    def __index_map(self):
        """ Indexing map states (in any cases) """
        # Index map - dict[state index, state position]
        # Unindex map - dict[state position, state index]
        # Index value map -> dict[state index, state value (description)]
        self.index_map, self.unindex_map, self.index_value_map = {}, {}, {}
        # Stat state indexing
        state_index = 0
        # Passing for map rows
        for row_index in range(self.amount_rows):
            # Passing for map columns
            for column_index in range(self.amount_columns):
                # Format state position
                position = (row_index, column_index)
                # Index state in any cases
                self.index_map[state_index] = position
                self.unindex_map[position] = state_index
                self.index_value_map[state_index] = self.text_map[row_index][column_index]
                # Increment state index
                state_index += 1

    def _get_next_state(self, current_state_index, direction):
        # type: (int, tuple) -> int
        """
        Get next state by current state and move direction by chosen action
        :param current_state_index: Int, start point
        :param direction: Tuple, move direction by chosen action
        :return Int, next state index
        """

        # Get state value (desctiption)
        state_value = self.index_value_map[current_state_index]
        # If state is transit
        if self.state_set.is_transit_state(state_value):
            # Get current state position
            current_state_position = self.index_map[current_state_index]
            # Move by direction
            new_state_position = (
                min(self.amount_rows - 1, max(0, current_state_position[0] + direction[0])),
                min(self.amount_columns - 1, max(0, current_state_position[1] + direction[1])),
            )
            # Return next state index
            return self.unindex_map[new_state_position]
        # Else if state is terminal
        elif self.state_set.is_end_state(state_value):
            # Do nothing and return current state index
            return current_state_index

    def _get_state_reward(self, current_state_index, next_state_index):
        """ For get state reward by current state and next state """
        raise NotImplementedError('Method _get_state_reward not implemented.')

    def _is_round_end(self, state_index):
        # type: (int) -> bool
        """
        Check is round end by state index
        (if state is terminal then round end else continue round
        """
        state_value = self.index_value_map[state_index]
        if self.state_set.is_end_state(state_value):
            return True
        else:
            return False

    @staticmethod
    def _get_state_probability(directions):
        """ Get direction probability (work when one action include many directions """
        return 1 / len(directions)

    def __generate_transition_matrix(self):
        """
        Generate transition matrix for work with on base methods
        Transition matrix -
            dict[state index] =
                dict[action index] =
                    list of action description (probability, next state, reward, terminal state flag)
        """

        # Initialize transition matrix container
        self.transition_matrix = {
            state_index: {
                action_index: []
                for action_index in range(self.action_space.n)
            }
            for state_index in self.index_map
        }
        # Pass for states
        for state_index in self.index_map:
            # Pass for actions
            for action in self.action_set:
                # Pass for directions
                for direction in action.directions:
                    # Get next state index by current state index and direction
                    new_state_index = self._get_next_state(state_index, direction)
                    # Update transition matrix information
                    self.transition_matrix[state_index][action.index].append((
                        self._get_state_probability(action.directions),
                        new_state_index,
                        self._get_state_reward(state_index, new_state_index),
                        self._is_round_end(new_state_index)
                    ))

    def step(self, action):
        # type: (int) -> (int, float, bool, None)
        """
        Play one time
        :param action: Int, chosen action for play
        :return: (int, float, bool, None) - current state, reward, flag end of episode and empty step description
        """

        # Increment episode size
        self.current_episode_size += 1

        # Get transfer states (from current by chosen action)
        transfer_states = self.transition_matrix[self.current_state_index][action]
        # Get move probabilities
        probabilities = [probability for probability, _, _, _ in transfer_states]
        # Chose random next state (by probabilties)
        transfer_state_index = random.choices(range(len(probabilities)), probabilities)[0]

        # Get step description
        _, new_state_index, reward, done = transfer_states[transfer_state_index]
        # If episode size equal max episode size then set end of episode
        if self.current_episode_size == self.max_episode_size:
            done = True

        # Update current state index
        self.current_state_index = new_state_index

        # Return step result
        return new_state_index, reward, done, _

    def reset(self, start_state_index=0):
        # type: (int) -> int
        """
        Reset environment for start new episode
        :param start_state_index: Int, from which state need start episode
        :return: Int, start state index
        """
        self.current_state_index = start_state_index
        self.current_episode_size = 0
        return self.current_state_index

    def __render_environment_state(self):
        """ Render environment board (current state)"""

        # Get current state (for to mark it)
        current_state_position = self.index_map[self.current_state_index]
        # Render border
        environment_display = render_functions.get_border(self.amount_rows, self.amount_columns)
        # Render header
        environment_display += render_functions.get_header(self.amount_rows, self.amount_columns)
        # Render boarder
        environment_display += render_functions.get_border(self.amount_rows, self.amount_columns)
        # Render rows
        for row_index in range(self.amount_rows):
            environment_display += render_functions.get_row(
                row_index,
                self.text_map[row_index],
                self.amount_rows,
                self.amount_columns,
                current_state_position[1] if current_state_position[0] == row_index else None
            )
        # Render boadred
        environment_display += render_functions.get_border(self.amount_rows, self.amount_columns)
        # Return environment board
        return environment_display

    def render(self, mode=RenderModes.HUMAN, object_type=RenderObjectTypes.ENVIRONMENT):
        # type: (str, str) -> Optional[str, None]
        """
        Render environment objects (board, actions, states)
        :param mode: Str, "human" or "ascii" mode
        :param object_type: Str, "environment", "actions" or "states" types
        :return: Str or None,
                 if input mode is "human" then print current info into console
                 if inout mode is "ascii" then return current info as str
        """

        # If type is "environment"
        if object_type == RenderObjectTypes.ENVIRONMENT:
            # Render environment board
            object_str = self.__render_environment_state()
        # Elif type is "actions"
        elif object_type == RenderObjectTypes.ACTIONS:
            # Render environment actions
            object_str = self.action_set.__str__()
        # Elif type is "states"
        elif object_type == RenderObjectTypes.STATES:
            # Render environment states
            object_str = self.state_set.__str__()
        # Else raise exception
        else:
            raise TypeError(f"Unknown render object type {object_type}.")

        # If mode is "human"
        if mode == RenderModes.HUMAN:
            # Print current state into console
            print(object_str)
        # Elif mode is "ascii"
        elif mode == RenderModes.ASCII:
            # Return current state as str
            return object_str
        # Else raise exception
        else:
            raise TypeError(f"Unknown render mode {mode}.")

    def seed(self, seed=None):
        # type: (int) -> None
        """ Set env generator seed"""
        super().seed(seed)
        random.seed(seed)
