"""
States data
"""


class BaseStates(object):
    """ Names for define one state """
    # Frozen surface
    FROZEN = 'F'
    # Hole
    HOLE = 'H'
    # Goal
    GOAL = 'G'


class BaseStatesSet(object):
    """ Base set of states description """

    def __init__(self, transit_states, end_states):
        # type: (list, list) -> None
        """
        :param transit_states: List, list of transit states
        :param end_states: List, list of terminal states
        """
        self.transit_states = transit_states
        self.end_states = end_states

    def is_transit_state(self, state):
        # type: (str) -> bool
        """ For check is state transit """
        return state in self.transit_states

    def is_end_state(self, state):
        # type: (str) -> bool
        """ For check is state terminal """
        return state in self.end_states

    @staticmethod
    def is_goal_state(state):
        # type: (str) -> bool
        """ For check is state goal """
        return state == BaseStates.GOAL

    @staticmethod
    def is_hole_state(state):
        # type: (str) -> bool
        """ For check is state hole """
        return state == BaseStates.HOLE

    def __iter__(self):
        """ Passing for state sets """
        for key, value in self.__dict__.items():
            yield key, value

    def __str__(self):
        """ Get state set string representation """
        return "States: " + ', '.join([f"{key}={value}" for key, value in self.__iter__()])


class StateSetNames(object):
    """ Base set of states description """
    DEFAULT = 'default'
    FALL = 'fall'


class States(object):
    """ Main state set class aggregator """

    @staticmethod
    def get(states_set_name):
        # type: (str) -> BaseStatesSet
        """
        Return state set by defined name
        :param states_set_name: Str, name from StateSetNames
        :return: BaseStatesSet, return state set if name exist else raise exception
        """
        if states_set_name == StateSetNames.DEFAULT:
            return States.DefaultStatesSet()
        elif states_set_name == StateSetNames.FALL:
            return States.FallStatesSet()
        else:
            raise TypeError(f"Unknown states set name \"{states_set_name}\".")

    class DefaultStatesSet(BaseStatesSet):
        """ Casual actions """
        def __init__(self):
            """
            Frozen surface - transit
            Hole and Goal - terminal
            """
            super().__init__(
                transit_states=[BaseStates.FROZEN],
                end_states=[BaseStates.HOLE, BaseStates.GOAL]
            )

    class FallStatesSet(BaseStatesSet):
        def __init__(self):
            """
            Frozen surface and Hole - transit
            Goal - terminal
            """
            super().__init__(
                transit_states=[BaseStates.FROZEN, BaseStates.HOLE],
                end_states=[BaseStates.GOAL]
            )
