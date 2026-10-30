import random
import numpy as np
from collections import defaultdict


class SARSAAgent:
    """
    SARSA agent implementation.

    TODO: Implement the following methods:
    1. learn() - implement the SARSA update rule
    2. Optional: Modify the exploration strategy
    3. Optional: Add any additional methods you think are useful

    The SARSA update rule is:
    Q(s,a) = Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]

    Where:
    - α (alpha) is the learning rate
    - γ (gamma) is the discount factor
    - r is the reward
    - s is the current state
    - a is the current action
    - s' is the next state
    - a' is the next action (chosen using the same policy as during action selection)

    Note: SARSA is an on-policy algorithm, meaning it learns the value of the policy
    it's currently following, unlike Q-learning which learns the optimal policy.
    """

    def __init__(self, learning_rate=0.1, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize SARSA agent.

        Args:
            learning_rate (float): Learning rate (α)
            discount_factor (float): Discount factor (γ)
            epsilon (float): Initial exploration rate
            epsilon_decay (float): Epsilon decay rate
            epsilon_min (float): Minimum exploration rate
        """
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: maps state-action pairs to Q-values
        # defaultdict automatically creates a new array for unseen states
        self.q_table = defaultdict(lambda: np.zeros(4))  # 4 actions: up, right, down, left

    def get_action(self, state, valid_actions):
        """
        Choose action using epsilon-greedy policy.

        Args:
            state: Current state
            valid_actions: List of valid actions from current state

        Returns:
            int: Selected action
        """
        if random.uniform(0, 1) < self.epsilon:
            # Explore: randomly select from valid actions
            return random.choice(valid_actions)
        else:
            # Exploit: select best action from Q-table
            q_values = self.q_table[state]
            # Filter Q-values for valid actions
            valid_q_values = {action: q_values[action] for action in valid_actions}
            return max(valid_q_values.keys(), key=lambda x: valid_q_values[x])

    def learn(self, state, action, reward, next_state, next_action):
        """
        Update Q-table using SARSA update rule.

        TODO: Implement this method!

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Action to be taken in next state
        """
        # Current estimate of Q(s,a)
        current_q = self.q_table[state][action]

        # Value of the action actually chosen next (on-policy), the key difference
        # from Q-learning. next_action is None at terminal states, whose value is 0.
        next_q = 0 if next_action is None else self.q_table[next_state][next_action]

        # Q(s,a) = Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]
        new_q = current_q + self.lr * (reward + self.gamma * next_q - current_q)

        # Update Q-table
        self.q_table[state][action] = new_q

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def set_epsilon(self, epsilon):
        """Set exploration rate"""
        self.epsilon = max(epsilon, self.epsilon_min)

    def save_q_table(self, filename):
        """Save Q-table to file"""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load_q_table(self, filename):
        """Load Q-table from file"""
        import pickle
        with open(filename, 'rb') as f:
            q_dict = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(4))
        for key, value in q_dict.items():
            self.q_table[key] = value