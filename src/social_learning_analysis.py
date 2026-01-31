import numpy as np
import random
import pickle
import json
import sys
from collections import defaultdict
from scipy.stats import entropy
from scipy.stats import beta as beta_dist
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import argparse
from pathlib import Path
from utils import get_run_id, setup_run_directory, write_run_yaml, write_config_yaml, finalize_run_yaml



class Bandit: 
    def __init__(self, payoffs: List[float]):
        self.payoffs = payoffs

    def pull(self, action: int) -> int:
        if random.random() < self.payoffs[action]:
            return 1
        return 0

    @property
    def K(self):
        return len(self.payoffs)

class GreedySampler():
    def __init___(self, alphas: np.ndarray, betas: np.ndarray, bandit: Bandit, T: int):
        self._alphas = alphas
        self._betas = betas
        self._bandit = bandit 
        self._T == t
        self._action = np.zeros(T, dtype=int)
        self._rewards = np.zeros(T, dtype=int)
        self.q_values = self._alphas / (self._alphas + self._betas)

    def _sample_action(self) -> int: 
        return np.argmax(self.q_values)

    def run(self):
        for t in range(self._T):
            action = self._sample_action()
            reward = self.bandit.pull(action)
            self._actions[t] = action
            self._rewards[t] = reward

    @property
    def actions(self) -> np.ndarray:
        return self._actions

    @property
    def rewards(self) -> np.ndarray:
        return self._rewards

class ThompsonSampler():
    def __init__(self, alphas: np.ndarray, betas: np.ndarray, bandit: Bandit, T: int):
        self._alphas = alphas
        self._betas = betas
        self._bandit = bandit
        self._t_alphas = np.zeros(T, bandit.K)
        self._t_betas = np.zeros(T, bandit.K)
        self._T = T
        self._actions = np.zeros(T, dtype=int)
        self._rewards = np.zeros(T, dtype=int)

    def _sample_action(self) -> int:
        return np.argmax(np.random.beta(self._alphas, self._betas))

    def run(self):
        for t in range(self._T):
            action = self._sample_action()
            reward = self._bandit.pull(action)
            self._alphas[action] += reward
            self._betas[action] += 1 - reward
            self._t_alphas[t, action] = self._alphas[action]
            self._t_betas[t, action] = self._betas[action]
            self._actions[t] = action
            self._rewards[t] = reward

    @property
    def alphas(self) -> np.ndarray:
        return self._alphas

    @property
    def betas(self) -> np.ndarray:
        return self._betas

    @property
    def bandit(self) -> Bandit:
        return self._bandit

    @property
    def T(self) -> int:
        return self._T

    @property
    def t_alphas(self): -> np.ndarray:
        return self._t_alphas

    @property
    def t_betas(self) -> np.ndarray:
        return self._t_betas
    
    @property
    def actions(self) -> np.ndarray:
        return self._actions

    @property
    def rewards(self) -> np.ndarray:
        return self._rewards

class Agent():
    def __init__(self, agent_type, actions, rewards):
        self.agent_type = agent_type
        self.rewards = rewards
        self.cumulative_rewards = np.cumsum(rewards)
        self.entropy = compute_entropy(actions)

def compute_entropy(actions):
    counts = np.bincount(actions, minlength=12)
    return entropy(actions)

def append_to_csv(agent, T_E, T_D, T_P, filename):

    new_data = pd.DataFrame({
        'agent_type': np.repeat(agent.agent_type, len(actions)),
        'T_E': np.repeat(T_E, len(actions)),
        'T_D': np.repeat(T_D, len(actions)),
        'T_P': np.repeat(T_P, len(actions)),
        'rewards': agent.rewards,
        'cumulative_rewards': agent.cumulative_rewards,
        'entropy': agent.entropy
    })  
    file_exists = os.path.isfile(filename)
    new_data.to_csv(filename, mode='a', header=not file_exists, index=False)


K=12
T_E = 128 # prior experience
T_D = 20 # number of demonstration steps 
T_P = [1,2,4,8,16,32] # number of play steps 
sorted_payoffs = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.75, 0.8]
prior_alphas = np.ones(K)
prior_betas = np.ones(K)

for i in range(10):
    payoffs = random.sample(sorted_payoffs, K)
    random.shuffle()
    bandit = Bandit(payoffs)

    demonstrator = ThompsonSampler(prior_alphas, prior_betas, bandit, T_E + T_D + T_P_max)
    demonstrator.run()

    for t_d in range(T_D):
        for t_p in T_P:
            # individual learner is same as demonstrators first t_d + t_p steps
            individual_learner_agent = Agent(agent_type='individual learner', actions=demonstrator.actions[t_d:t_d+t_p], rewards=demonstrator.rewards[t_d:t_d+t_p])
            append_to_csv(agent=individual_learner_agent, T_E=T_E, T_D=t_d, T_P=t_p, filename='social_learning_analysis.csv')

            # naive thompson social learner uses betas resulting from demonstration window as priors
            naive_prior_alphas = demonstrator.t_alphas[T_E + t_d - 1, :] - demonstrator.t_alphas[T_E - 1 , :] + prior_alphas
            naive_prior_betas = demonstrator.t_betas[T_E + t_d - 1, :] - demonstrator.t_betas[T_E - 1, :] + prior_betas
            naive_thompson_social_learner = ThompsonSampler(naive_prior_alphas, naive_prior_betas, bandit, t_p)
            naive_thompson_social_learner.run()
            naive_thompson_social_learner_agent = Agent(agent_type='naive thompson social learner', actions=naive_thompson_social_learner.actions, rewards=naive_thompson_social_learner.rewards)
            append_to_csv(agent=naive_thompson_social_learner_agent, T_E=T_E, T_D=t_d, T_P=t_p, filename='social_learning_analysis.csv')

            # greedy naive social learner 
            naive_greedy_social_learner = GreedySampler(naive_prior_alphas, naive_prior_betas, bandit, t_p)
            naive_greedy_social_learner.run()
            naive_greedy_social_learner_agent = Agent(agent_type='naive thompson social learner', actions=naive_greedy_social_learner.actions, rewards=naive_greedy_social_learner.rewards)


            # oracle thompson social learner 

            # oracle greedy social learner 





