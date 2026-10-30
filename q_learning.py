import random
import numpy as np
from collections import defaultdict


class QLearningAgent:
    """
    Q-Learning 智能体。

    更新规则：
    Q(s,a) = Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]

    其中：
    - α 是学习率，γ 是折扣因子，r 是奖励
    - s、a 是当前状态和动作，s' 是下一状态
    - max_a' Q(s',a') 是下一状态的最大 Q 值
    """

    def __init__(self, learning_rate=0.1, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        初始化 Q-Learning 智能体。

        参数:
            learning_rate (float): 学习率 α
            discount_factor (float): 折扣因子 γ
            epsilon (float): 初始探索率
            epsilon_decay (float): ε 的衰减系数
            epsilon_min (float): 最小探索率
        """
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q 表：记录每个状态下各动作的 Q 值
        # defaultdict 会为没见过的状态自动创建全零数组
        self.q_table = defaultdict(lambda: np.zeros(4))  # 4 个动作：上、右、下、左

    def get_action(self, state, valid_actions):
        """
        按 ε-贪心策略选择动作。

        参数:
            state: 当前状态
            valid_actions: 当前状态下的合法动作列表

        返回:
            int: 选中的动作
        """
        if random.uniform(0, 1) < self.epsilon:
            # 探索：在合法动作中随机选择
            return random.choice(valid_actions)
        else:
            # 利用：选择 Q 表中价值最高的动作
            q_values = self.q_table[state]
            # 只考虑合法动作的 Q 值
            valid_q_values = {action: q_values[action] for action in valid_actions}
            return max(valid_q_values.keys(), key=lambda x: valid_q_values[x])

    def learn(self, state, action, reward, next_state, next_valid_actions):
        """
        按 Q-Learning 规则更新 Q 表。

        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            next_valid_actions: 下一状态的合法动作列表
        """
        # 当前的 Q(s,a) 估计值
        current_q = self.q_table[state][action]

        # 取 s' 的最大 Q 值（off-policy：不管下一步实际执行什么动作）。
        # train.py 在终止状态也会传入动作列表，但终止状态从不被更新，
        # Q 值始终为 0，所以此时目标值正好退化为 r。
        if next_valid_actions:
            max_next_q = max(self.q_table[next_state][a] for a in next_valid_actions)
        else:
            max_next_q = 0

        # Q(s,a) = Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)

        # 写回 Q 表
        self.q_table[state][action] = new_q

        # 衰减 ε（注意：每步衰减一次，而不是每回合）
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def set_epsilon(self, epsilon):
        """设置探索率"""
        self.epsilon = max(epsilon, self.epsilon_min)

    def save_q_table(self, filename):
        """把 Q 表保存到文件"""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load_q_table(self, filename):
        """从文件加载 Q 表"""
        import pickle
        with open(filename, 'rb') as f:
            q_dict = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(4))
        for key, value in q_dict.items():
            self.q_table[key] = value