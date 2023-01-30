from e_greedy import Experiment
import gym
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    env_type = 'env-v1'
    count_runs = 1
    count_iter = 1000
    exp = Experiment(env_type=f'bandits:{env_type}', eps=1, q_init_value=0, count_iter=count_iter)
    no_opt_runs_rewards, no_opt_runs_accuracy, no_opt_runs_balances = exp(env_type, count_runs=count_runs, save=True)

    exp = Experiment(env_type=f'bandits:{env_type}', eps=1, q_init_value=50, count_iter=count_iter)

    opt_runs_rewards, opt_runs_accuracy, opt_runs_balances = exp(env_type + 'optimistic', count_runs=count_runs,
                                                                 save=True)
    plt.show()

    plt.plot(no_opt_runs_rewards, label='no optimistic')
    plt.plot(opt_runs_rewards, label='optimistic')
    plt.title(f'Кумулятивная средняя награда, кол-во запусков: {count_runs}')
    plt.legend()
    plt.savefig('reward.png')
    plt.show()

    plt.plot(no_opt_runs_accuracy, label='no optimistic')
    plt.plot(opt_runs_accuracy, label='optimistic')
    plt.title(f'Кумулятивная точность, кол-во запусков: {count_runs}')
    plt.legend()
    plt.savefig('acc.png')
    plt.show()

    plt.plot(no_opt_runs_balances, label='no optimistic')
    plt.plot(opt_runs_balances, label='optimistic')
    plt.title(f'Баланс, кол-во запусков: {count_runs}')
    plt.legend()
    plt.savefig('balance.png')

