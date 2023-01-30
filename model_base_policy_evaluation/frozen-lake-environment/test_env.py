"""
Frozen lake environment test
"""

# For load env
import gym


####################################################################################################
#################################### frozen lake default test ######################################
####################################################################################################

def test_init_default_env():
    """ Test init default env """
    gym.make('frozen_lake:default-v0')


def test_step_default_env():
    """ Test step default env """
    env = gym.make('frozen_lake:default-v0')
    env.reset()
    env.step(0)


####################################################################################################
##################################### frozen lake fall test ########################################
####################################################################################################

def test_init_fall_env():
    """ Test init fall env """
    gym.make('frozen_lake:fall-v0')


def test_step_fall_env():
    """ Test step fall env """
    env = gym.make('frozen_lake:fall-v0')
    env.reset()
    env.step(0)


####################################################################################################
##################################### frozen lake other test #######################################
####################################################################################################

def test_init_maps():
    """ Test init maps """
    gym.make('frozen_lake:default-v0', map_name='small')
    gym.make('frozen_lake:default-v0', map_name='medium')
    gym.make('frozen_lake:default-v0', map_name='large')
    gym.make('frozen_lake:default-v0', map_name='huge')
    gym.make('frozen_lake:default-v0', map_name='colossal')


def test_render():
    """ Test render """
    env = gym.make('frozen_lake:default-v0')
    env.reset()
    env.render(mode='ascii', object_type='environment')
    env.render(mode='ascii', object_type='actions')
    env.render(mode='ascii', object_type='states')


####################################################################################################
##################################### frozen lake seed test ########################################
####################################################################################################

def test_init_random_seed_env():
    """ Test init random seed env """
    env = gym.make('frozen_lake:default-v0')
    env.seed(None)


def test_init_fix_seed_env():
    """ Test init fix seed env """
    env = gym.make('frozen_lake:default-v0')
    env.seed(1)
