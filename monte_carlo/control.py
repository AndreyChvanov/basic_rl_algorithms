import time
import numpy as np
import gym
from policy_iteration import PolicyIteration
import random


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


class MonteCarloControl:
    def __init__(self, env, max_episode_len=10000):
        self.env = env
        self.count_action = self.env.action_space.n
        self.count_states = self.env.observation_space.n
        self.max_episode_len = max_episode_len

    def _get_episode(self, policy, s0=None, a0=None):
        done = False
        episode_history = []
        if s0 is not None:
            cur_state = self.env.reset(start_state_index=s0)
            new_state_index, reward, done, _ = self.env.step(a0)
            episode_history.append((s0, a0, reward))
            cur_state = new_state_index
        else:
            cur_state = self.env.reset()
        while not done:
            action = np.random.choice(np.arange(self.count_action), p=policy[cur_state])
            new_state_index, reward, done, _ = self.env.step(action)
            episode_history.append((cur_state, action, reward))
            cur_state = new_state_index
        return episode_history

    def create_policy(self, Q, eps=0.0):
        policy = np.ones_like(Q, dtype=float)
        policy = policy * eps / self.count_action
        opt_action = np.argmax(Q, axis=1)
        policy[np.arange(Q.shape[0]), opt_action] += (1 - eps)
        return policy

    def exploring_starts(self, nrof_episodes=1, gamma=0.1):
        policy = np.ones((self.env.observation_space.n, self.env.action_space.n)) / self.env.action_space.n
        Q = np.zeros_like(policy)
        returns = {(s, a): [] for s in range(self.count_states)
                   for a in range(self.count_action)}
        for e in range(nrof_episodes):
            s0, a0 = np.random.randint(self.count_states), np.random.randint(self.count_action)
            episode = self._get_episode(policy, s0=s0, a0=a0)
            G = 0
            states_actions = [(i[0], i[1]) for i in episode][::-1]
            for t, (s, a, r) in enumerate(episode[::-1]):
                G = gamma * G + r
                if (s, a) not in states_actions[t + 1:]:
                    returns[(s, a)].append(G)
                    Q[s, a] = sum(returns[(s, a)]) / len(returns[(s, a)])
            policy = self.create_policy(Q)
        return policy

    def policy_control(self, nrof_episodes=1, eps=0.1, gamma=0.1):
        Q = np.zeros((self.count_states, self.count_action))
        policy = self.create_policy(Q, eps)
        returns = {(s, a): [] for s in range(self.count_states)
                   for a in range(self.count_action)}
        for e in range(nrof_episodes):
            episode = self._get_episode(policy, s0=None, a0=None)
            G = 0
            states_actions = [(i[0], i[1]) for i in episode][::-1]
            for t, (s, a, r) in enumerate(episode[::-1]):
                G = gamma * G + r
                if (s, a) not in states_actions[t + 1:]:
                    returns[(s, a)].append(G)
                    Q[s, a] = sum(returns[(s, a)]) / len(returns[(s, a)])
            policy = self.create_policy(Q, eps)
        return policy


def show_policy_game_board(policy, env_shape):
    p = policy.argmax(axis=1)
    return p.reshape(env_shape)



if __name__ == '__main__':
    env = gym.make('frozen_lake:fall-v0', map_name='small')
    mc_eval = MonteCarloControl(env=env)
    gamma = 0.5
    policy = mc_eval.exploring_starts(gamma=gamma, nrof_episodes=10000)
    print('MonteCarlo exploring_start ')
    print(show_policy_game_board(policy, env.shape))
    env = gym.make('frozen_lake:fall-v0', map_name='small')
    mc_eval = MonteCarloControl(env=env)
    gamma = 0.5
    policy = mc_eval.exploring_starts(gamma=gamma, nrof_episodes=1000)
    print('MonteCarlo exploring_start ')
    print(show_policy_game_board(policy, env.shape))
    # set_seed(0)
    # env = gym.make('frozen_lake:fall-v0', map_name='small')
    # policy = mc_eval.policy_control(gamma=gamma, nrof_episodes=1, eps=0.1)
    # print('MonteCarlo policy control')
    # print(show_policy_game_board(policy, env.shape))
    #
    # alg = PolicyIteration(env, gamma=gamma, eval_policy_th=0.0001)
    # policy_1, V, i = alg.value_iteration(th=0.00001)
    # print('Value iteration')
    # print(show_policy_game_board(policy_1, env.shape))
