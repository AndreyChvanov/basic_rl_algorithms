import numpy as np
import random
import gym
import pandas as pd
import matplotlib.pyplot as plt
import os


def eps_greedy_alg(env, amount_arms, eps, q_init_value, count_iter):
    Q = np.zeros(amount_arms) + q_init_value
    N = np.zeros(amount_arms)
    env.reset()
    i = 0
    best_arm = env.get_best_arm()
    print(best_arm)
    rewards = []
    balances = []
    accuracy = []
    while True:
        a = np.random.randint(0, amount_arms) if np.random.rand() < eps else np.argmax(Q)
        balance, reward, done, _ = env.step(a)
        N[a] += 1
        Q[a] = Q[a] + 1 / N[a] * (reward - Q[a])
        i += 1
        rewards.append(reward)
        balances.append(balance)
        accuracy.append(int(a == best_arm))
        if done or i > count_iter - 1:
            break
    env.close()
    print(np.mean(rewards), np.std(rewards))
    return rewards, balances, accuracy, N


class Experiment:
    def __init__(self, env_type, eps, q_init_value, count_iter):
        self.env_type = env_type
        self.eps = eps
        self.q_init_values = q_init_value
        self.count_iter = count_iter

    def run(self, env):
        rewards, balance, accuracy, arm_n = eps_greedy_alg(env, amount_arms=10, eps=self.eps,
                                                           q_init_value=self.q_init_values, count_iter=self.count_iter)
        rewards = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
        accuracy = np.cumsum(accuracy) / np.arange(1, len(accuracy) + 1)
        return rewards, balance, accuracy, arm_n

    @staticmethod
    def save_experiment(path2save, runs_rewards, runs_balances, runs_accuracy, count_runs):
        if not os.path.isdir(path2save):
            os.makedirs(path2save)

        runs_rewards.to_csv(os.path.join(path2save, 'reward.csv'), index=False)
        runs_balances.to_csv(os.path.join(path2save, 'balance.csv'), index=False)
        runs_accuracy.to_csv(os.path.join(path2save, 'acc.csv'), index=False)

        plt.figure(figsize=(15, 10))
        runs_accuracy.mean(axis=1).plot(title=f'Кумулятивная точность, кол-во запусков: {count_runs}')
        plt.savefig(os.path.join(path2save, 'acc.png'))
        plt.figure(figsize=(15, 10))
        runs_rewards.mean(axis=1).plot(title=f'Кумулятивная средняя награда, кол-во запусков: {count_runs}')
        plt.savefig(os.path.join(path2save, 'mean_reward.png'))
        plt.figure(figsize=(15, 10))
        runs_balances.mean(axis=1).plot(title=f'Баланс, кол-во запусков: {count_runs}')
        plt.savefig(os.path.join(path2save, 'mean_balance.png'))

    def __call__(self, exp_name, count_runs, save=True):
        runs_rewards = pd.DataFrame()
        runs_balances = pd.DataFrame()
        runs_accuracy = pd.DataFrame()
        arm_bar = pd.DataFrame()
        # env = gym.make(self.env_type, seed=0)
        for i in range(count_runs):
            env = gym.make(self.env_type, seed=i)
            np.random.seed(0)
            random.seed(0)
            rewards, balances, accuracy, arm_n = self.run(env)
            runs_rewards.loc[:len(rewards)-1, i] = rewards
            runs_balances.loc[:len(balances)-1, i] = balances
            runs_accuracy.loc[:len(accuracy)-1, i] = accuracy
            arm_bar.loc[:, i] = arm_n
        if save:
            self.save_experiment(path2save=os.path.join('experiments', f'{exp_name}_{count_runs}_{self.count_iter}'),
                                 runs_rewards=runs_rewards,
                                 runs_accuracy=runs_accuracy,
                                 runs_balances=runs_balances,
                                 count_runs=count_runs)
        return runs_rewards.mean(axis=1), runs_accuracy.mean(axis=1), runs_balances.mean(axis=1)
