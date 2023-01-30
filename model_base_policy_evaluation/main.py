import gym
from policy_eval import PolicyEvaluator
import numpy as np

if __name__ == "__main__":
    env = gym.make('frozen_lake:default-v0', map_name='small', action_set_name='slippery')
    env.reset()
    new_state_index, reward, done, _ = env.step(2)

    policy = np.arange(0, env.observation_space.n * env.action_space.n).reshape(
        (env.observation_space.n, env.action_space.n))
    policy = policy / np.sum(policy, axis=1).reshape((-1, 1))

    evaluator = PolicyEvaluator(env=env)
    v1 = evaluator.direct_evaluation(policy)
    v2 = evaluator.iterative_evaluation(policy, th=0.001)
    print(v1)

    print('_________________')

    error = np.abs(v1-v2).sum()
    print(v2)
    print(error)


