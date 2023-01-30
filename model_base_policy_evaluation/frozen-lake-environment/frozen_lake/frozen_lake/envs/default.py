"""
Default environment class
"""

# For inheritance base environment class
from .base import Base

# Import map, action, state names and environment initialize params
from ..data import MapNames, ActionSetNames, StateSetNames, EnvironmentParams


class Default(Base):
    """ Default environment """
    def __init__(
            self,
            map_name=MapNames.SMALL,
            action_set_name=ActionSetNames.DEFAULT,
            max_episode_size=EnvironmentParams.MAX_EPISODE_SIZE
    ):
        super(Default, self).__init__(map_name, action_set_name, StateSetNames.DEFAULT, max_episode_size)

    def _get_state_reward(self, current_state_index, next_state_index):
        # type: (int, int) -> int
        """
        Get state reward logic
        (in current realization
        if player coming into goal state first time then he receive 1 point
        else in any other case he receive 0 points)
        """
        next_state_value = self.index_value_map[next_state_index]
        if self.state_set.is_goal_state(next_state_value):
            return 0 if current_state_index == next_state_index else 1
        else:
            return 0
