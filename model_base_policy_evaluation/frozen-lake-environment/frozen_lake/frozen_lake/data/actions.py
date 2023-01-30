"""
Action data
"""


class BaseAction(object):
    """ Base one action description """

    def __init__(self, index, name, directions):
        """
        :param index: Int, action index
        :param name: Str, action name
        :param directions: Tuple, (x, y) move direction
        """
        self.index = index
        self.name = name
        self.directions = directions

    def __str__(self):
        """ For get action string representation """
        return f"{self.index}={self.name}"


class BaseActionSet(object):
    """ Base set of actions description """

    def __iter__(self):
        """ Passing for actions """
        for value in self.__dict__.values():
            yield value

    def __len__(self):
        """ Get action set size """
        return len(self.__dict__)

    def __str__(self):
        """ Get action set string representation """
        return f"Actions: {', '.join([action.__str__() for action in self.__iter__()])}"


class ActionSetNames(object):
    """ Names for define action set packs """
    DEFAULT = 'default'
    SLIPPERY = 'slippery'


class Actions(object):
    """ Main action set class aggregator """

    @staticmethod
    def get(actions_set_name):
        # type: (str) -> BaseActionSet
        """
        Return action set by defined name
        :param actions_set_name: Str, name from ActionSetNames
        :return: BaseActionSet, return action set if name exist else raise exception
        """
        if actions_set_name == ActionSetNames.DEFAULT:
            return Actions.DefaultActionSet()
        elif actions_set_name == ActionSetNames.SLIPPERY:
            return Actions.SlipperyActionSet()
        else:
            raise TypeError(f"Unknown actions set name \"{actions_set_name}\".")

    class DefaultActionSet(BaseActionSet):
        """ Casual actions """
        def __init__(self):
            # Left move left
            self.left = BaseAction(index=0, name='left', directions=[(0, -1)])
            # Down move down
            self.down = BaseAction(index=1, name='down', directions=[(1, 0)])
            # Right move right
            self.right = BaseAction(index=2, name='right', directions=[(0, 1)])
            # Up move up
            self.up = BaseAction(index=3, name='up', directions=[(-1, 0)])

    class SlipperyActionSet(BaseActionSet):
        """ Slippery actions  """
        def __init__(self):
            # Left move left, up and down
            self.left = BaseAction(index=0, name='left', directions=[(0, -1), (1, 0), (-1, 0)])
            # Down move down, left and right
            self.down = BaseAction(index=1, name='down', directions=[(1, 0), (0, -1), (0, 1)])
            # Right move right, up and down
            self.right = BaseAction(index=2, name='right', directions=[(0, 1), (1, 0), (-1, 0)])
            # Up move up, left and right
            self.up = BaseAction(index=3, name='up', directions=[(-1, 0), (0, -1), (0, 1)])
