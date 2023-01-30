import time
import numpy as np
import gym


class DirectEvaluator:
    def __init__(self, env):
        self.env = env
        self.count_action = len(self.env.action_set)
        self.gamma = 0.1
        self.count_states = len(self.env.transition_matrix)

    def get_mdp(self):
        P = np.zeros((self.count_states, self.count_action, self.count_states))
        R = np.zeros((self.count_states, self.count_action, self.count_states))
        for state in self.env.transition_matrix.keys():
            for action_ind, cond_probs in self.env.transition_matrix[state].items():
                for prob in cond_probs:
                    next_state_prob, next_state, reward = prob[0], prob[1], prob[2]
                    P[state, action_ind, next_state] += next_state_prob
                    R[state, action_ind, next_state] += reward
        return P, R

    def __call__(self, policy):
        p, r = self.get_mdp()
        p_pi = np.zeros((self.count_states, self.count_states))
        for s in range(self.count_states):
            p_pi[s] = policy[s] @ p[s]

        r_sa = np.sum(p * r, axis=-1)
        r_pi = np.zeros(self.count_states)
        for s in range(self.count_states):
            r_pi[s] = policy[s] @ r_sa[s]
        v = np.linalg.inv(np.eye(p_pi.shape[0]) - self.gamma * p_pi) @ r_pi
        return v


class MonteCarloEvaluator:
    def __init__(self, env, max_episode_len=1000):
        self.env = env
        self.count_action = len(self.env.action_set)
        self.count_states = len(self.env.transition_matrix)
        self.max_episode_len = max_episode_len

    def _get_episode(self, policy):
        done = False
        cur_state = self.env.reset()
        episode_history = []
        i = 0
        while not done:
            action = np.random.choice(np.arange(4), p=policy[cur_state])
            new_state_index, reward, done, _ = self.env.step(action)
            episode_history.append((cur_state, action, reward))
            cur_state = new_state_index
            i += 1
            if i > self.max_episode_len:
                break
        return episode_history

    def every_visit_evaluation(self, policy, gamma=0.1):
        V = np.zeros(self.count_states)
        returns = {s: [] for s in range(self.count_states)}
        for i in range(1500):
            episode = self._get_episode(policy)
            G = 0
            for s, a, r in episode[::-1]:
                G = gamma * G + r
                returns[s].append(G)
        for s in range(self.count_states):
            if returns[s]:
                V[s] = np.mean(returns[s])

        return V

    def first_visit_evaluation(self, policy, gamma=0.1):
        V = np.zeros(self.count_states)
        returns = {s: [] for s in range(self.count_states)}
        for i in range(1500):
            episode = self._get_episode(policy)
            G = 0
            states = [i[0] for i in episode][::-1]
            for t, (s, a, r) in enumerate(episode[::-1]):
                G = gamma * G + r
                if s not in states[t + 1:]:
                    returns[s].append(G)
        for s in range(self.count_states):
            if returns[s]:
                V[s] = np.mean(returns[s])

        return V


def get_bias_variance(count_runs, env, policy, max_episode_len, eval_type):
    d_eval = DirectEvaluator(env=env)
    v_direct = d_eval(policy)
    eval_module = MonteCarloEvaluator(env=env, max_episode_len=max_episode_len)
    v_exps = []
    for i in range(count_runs):
        if eval_type == 'first_visit':
            v = eval_module.first_visit_evaluation(policy=policy)
        else:
            v = eval_module.every_visit_evaluation(policy=policy)
        v_exps.append(v)
    v_exps = np.vstack(v_exps)
    bias = np.linalg.norm(v_direct - v_exps, axis=1, ord=2).mean()
    variance = np.linalg.norm(np.mean(np.power(v_exps, 2), axis=0) - np.power(np.mean(v_exps, axis=0), 2), ord=2)
    return bias, variance



if __name__ == '__main__':
    np.random.seed(0)
    env = gym.make('frozen_lake:fall-v0', map_name='small')
    policy = np.zeros((env.observation_space.n, env.action_space.n)) + 1 / env.action_space.n

    d_eval = DirectEvaluator(env=env)
    v_direct = d_eval(policy)
    print(v_direct)

    mc_eval = MonteCarloEvaluator(env=env)

    v = mc_eval.first_visit_evaluation(policy=policy)
    print('first visit')
    print(v)

    t = time.time()
    v = mc_eval.every_visit_evaluation(policy=policy)
    print('every visit')
    print(v)
    print(time.time() - t)