import numpy as np


class PolicyEvaluator:
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
                    P[state, action_ind, next_state] = next_state_prob
                    R[state, action_ind, next_state] = reward
        return P, R

    def direct_evaluation(self, policy):
        policy = policy / np.sum(policy, axis=1).reshape((-1, 1))
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

    def iterative_evaluation(self, policy, th=0.001):
        V = np.zeros(self.count_states)
        p, r = self.get_mdp()
        while True:
            delta = 0
            prev_v = V.copy()
            for s in range(self.count_states):
                v = prev_v[s]
                v2_ = np.sum(policy[s] @ (p[s] * (r[s] + self.gamma * prev_v)))
                V[s] = v2_
                delta = max(delta, abs(v - V[s]))
            if delta < th:
                break
        return V
