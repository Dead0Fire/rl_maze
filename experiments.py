"""
Extra experiments for the report: python experiments.py
Writes figures to figures/ and prints the tables used in REPORT.md.

Every result is averaged over SEEDS independent runs, and every run is seeded,
so the numbers are reproducible. environment.py and train.py are not modified.
"""
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from environment import SimpleMaze
from q_learning import QLearningAgent
from sarsa import SARSAAgent

SEEDS = 30
MAX_STEPS = 100  # same limit as train.py
FIG_DIR = Path(__file__).parent / 'figures'
COLORS = {'Q-Learning': '#2a78d6', 'SARSA': '#eb6834', 'Expected SARSA': '#1baf7a'}
ARROWS = {0: (0, 0.3), 1: (0.3, 0), 2: (0, -0.3), 3: (-0.3, 0)}  # up, right, down, left

plt.rcParams.update({
    'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#e1e0d9', 'grid.linewidth': 0.6,
    'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#52514e', 'axes.axisbelow': True,
    'xtick.color': '#52514e', 'ytick.color': '#52514e', 'lines.linewidth': 2,
    'legend.frameon': False, 'figure.dpi': 150,
})


class ExpectedSARSAAgent(QLearningAgent):
    """Target is the expected Q(s',a') under the epsilon-greedy policy."""

    def learn(self, state, action, reward, next_state, next_valid_actions):
        q = self.q_table[next_state]
        # Same tie-breaking as get_action: the first maximal valid action is greedy
        greedy = max(next_valid_actions, key=lambda a: q[a])
        expected = self.epsilon * np.mean([q[a] for a in next_valid_actions]) \
            + (1 - self.epsilon) * q[greedy]
        target = reward + self.gamma * expected
        self.q_table[state][action] += self.lr * (target - self.q_table[state][action])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class SoftmaxMixin:
    """Boltzmann exploration: P(a) is proportional to exp(Q(s,a) / tau)."""
    tau = 5.0

    def get_action(self, state, valid_actions):
        q = self.q_table[state][valid_actions]
        weights = np.exp((q - q.max()) / self.tau)  # subtract max to avoid overflow
        return random.choices(valid_actions, weights=weights)[0]


class SoftmaxQLearning(SoftmaxMixin, QLearningAgent):
    pass


class SoftmaxSARSA(SoftmaxMixin, SARSAAgent):
    pass


def train_once(agent, seed, episodes):
    """Same loop as train.py, returning per-episode (reward, steps, success)."""
    random.seed(seed)
    env = SimpleMaze()
    on_policy = isinstance(agent, SARSAAgent)
    log = []
    for _ in range(episodes):
        state = env.reset()
        total, steps, done = 0, 0, False
        action = agent.get_action(state, env.get_valid_actions(state))
        while not done and steps < MAX_STEPS:
            next_state, reward, done = env.step(action)
            next_valid = env.get_valid_actions(next_state)
            if on_policy:
                next_action = agent.get_action(next_state, next_valid) if not done else None
                agent.learn(state, action, reward, next_state, next_action)
            else:
                agent.learn(state, action, reward, next_state, next_valid)
                keep_going = not done and steps + 1 < MAX_STEPS
                next_action = agent.get_action(next_state, next_valid) if keep_going else None
            state, action = next_state, next_action
            total += reward
            steps += 1
        log.append((total, steps, state == env.target_pos))
    return np.array(log, dtype=float)


def greedy_route(agent):
    """Follow the greedy policy once (the environment is deterministic)."""
    env = SimpleMaze()
    state, done, steps = env.reset(), False, 0
    visited = {state}
    while not done and steps < 50:
        q = agent.q_table[state]
        state, _, done = env.step(max(env.get_valid_actions(state), key=lambda a: q[a]))
        visited.add(state)
        steps += 1
    if state != env.target_pos:
        return 'fail'
    if steps > 7:
        return 'detour'
    return 'top' if (0, 1) in visited else 'left'


def run_many(make_agent, episodes):
    """Train SEEDS agents. Returns logs with shape (seeds, episodes, 3) and the agents."""
    agents = [make_agent() for _ in range(SEEDS)]
    logs = np.array([train_once(agent, seed, episodes) for seed, agent in enumerate(agents)])
    return logs, agents


def row(name, logs, agents):
    routes = Counter(greedy_route(agent) for agent in agents)
    last = logs[:, -100:]
    seed_rewards = last[..., 0].mean(1)
    ci = 1.96 * seed_rewards.std() / np.sqrt(SEEDS)
    total_failures = (1 - logs[..., 2]).sum(1).mean()
    print(f'{name:26s} reward={seed_rewards.mean():6.1f}+-{ci:4.1f}  train_success={last[..., 2].mean():6.1%}  '
          f'steps={last[..., 1].mean():5.1f}  total_failures={total_failures:6.1f}  routes: top={routes["top"]:2d} left={routes["left"]:2d} '
          f'detour={routes["detour"]:2d} fail={routes["fail"]:2d}')
    return routes


def smooth_band(values, window=20):
    """Moving average per seed, then mean and 95% CI across seeds."""
    kernel = np.ones(window) / window
    curves = np.array([np.convolve(v, kernel, mode='valid') for v in values])
    return curves.mean(0), 1.96 * curves.std(0) / np.sqrt(len(curves))


def plot_curve(ax, values, label, color, window=20):
    mean, ci = smooth_band(values, window)
    x = np.arange(window - 1, window - 1 + len(mean))
    ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.2, linewidth=0)
    ax.plot(x, mean, color=color, label=label)


def default_params():
    """Experiment 1: the default parameters from the assignment."""
    print('\n== 1. Default parameters, 500 episodes ==')
    results = {
        'Q-Learning': run_many(QLearningAgent, 500),
        'SARSA': run_many(SARSAAgent, 500),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, (logs, agents) in results.items():
        row(name, logs, agents)
        plot_curve(axes[0], logs[..., 0], name, COLORS[name])
        plot_curve(axes[1], logs[..., 1], name, COLORS[name])
        floors = np.argmax(np.cumsum(logs[..., 1], axis=1) >= 919, axis=1)
        print(f'  {name}: epsilon reaches 0.01 at episode {floors.mean():.0f} '
              f'(range {floors.min()}-{floors.max()})')
    start_values = [results['Q-Learning'][1][i].q_table[(0, 0)].max() for i in range(SEEDS)]
    print(f'  Q-Learning max Q(start, a) = {np.mean(start_values):.2f}, theory (7-step path) = '
          f'{sum(-0.95**k for k in range(6)) + 100 * 0.95**6:.2f}')
    axes[0].set(xlabel='Episode', ylabel='Reward per episode')
    axes[1].set(xlabel='Episode', ylabel='Steps per episode')
    axes[0].legend()
    fig.suptitle(f'Learning curves, default parameters (mean of {SEEDS} seeds, 95% CI, 20-episode average)')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig1_default_curves.png')


def epsilon_sweep():
    """Experiment 2: constant exploration exposes the on-policy/off-policy difference."""
    print('\n== 2. Constant epsilon, 1000 episodes ==')
    algos = {'Q-Learning': QLearningAgent, 'SARSA': SARSAAgent, 'Expected SARSA': ExpectedSARSAAgent}
    epsilons = [0.05, 0.1, 0.2, 0.3, 0.4]
    rewards = {name: [] for name in algos}
    detours = {name: [] for name in algos}
    kept = {}
    for eps in epsilons:
        for name, cls in algos.items():
            logs, agents = run_many(lambda: cls(epsilon=eps, epsilon_decay=1.0, epsilon_min=eps), 1000)
            routes = row(f'eps={eps} {name}', logs, agents)
            rewards[name].append(logs[:, -100:, 0].mean())
            detours[name].append(routes['detour'] / SEEDS * 100)
            if eps == 0.3:
                kept[name] = agents
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for name in algos:
        axes[0].plot(epsilons, rewards[name], marker='o', color=COLORS[name], label=name)
        axes[1].plot(epsilons, detours[name], marker='o', color=COLORS[name], label=name)
    axes[0].set(xlabel='Constant epsilon', ylabel='Reward per episode (last 100, while exploring)')
    axes[1].set(xlabel='Constant epsilon', ylabel='Seeds whose greedy policy detours (%)')
    axes[0].legend()
    fig.suptitle(f'Effect of exploration on behaviour (mean of {SEEDS} seeds)')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_epsilon_sweep.png')
    plot_policies(kept)


def plot_policies(results):
    """Majority-vote greedy action per cell across seeds, at epsilon = 0.3."""
    env = SimpleMaze()
    fig, axes = plt.subplots(1, len(results), figsize=(10, 3.6))
    for ax, (name, agents) in zip(axes, results.items()):
        ax.set(xlim=(0, env.size), ylim=(0, env.size), aspect='equal', xticks=[], yticks=[])
        ax.grid(False)
        ax.set_title(name)
        for r in range(env.size):
            for c in range(env.size):
                pos = (r, c)
                x, y = c, env.size - r - 1
                is_end = pos in env.traps or pos == env.target_pos
                color = '#e34948' if pos in env.traps else '#eda100' if pos == env.target_pos else 'white'
                ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor='#c3c2b7'))
                if is_end:
                    ax.text(x + 0.5, y + 0.5, 'T' if pos in env.traps else 'G', ha='center', va='center',
                            color='white' if pos in env.traps else '#0b0b0b', weight='bold')
                    continue
                valid = env.get_valid_actions(pos)
                votes = Counter(max(valid, key=lambda a: agent.q_table[pos][a]) for agent in agents)
                action, count = votes.most_common(1)[0]
                dx, dy = ARROWS[action]
                ax.arrow(x + 0.5 - dx / 2, y + 0.5 - dy / 2, dx, dy, head_width=0.16, head_length=0.12,
                         length_includes_head=True, color='#0b0b0b', alpha=0.35 + 0.65 * count / SEEDS)
    fig.suptitle(f'Greedy policy at constant epsilon = 0.3 (majority of {SEEDS} seeds; faint arrow = less agreement)')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_policy_eps03.png')


def softmax_sweep():
    """Experiment 3: Boltzmann exploration instead of epsilon-greedy."""
    print('\n== 3. Softmax exploration, constant tau, 1000 episodes ==')
    algos = {'Q-Learning': SoftmaxQLearning, 'SARSA': SoftmaxSARSA}
    taus = [1, 2, 5, 10, 20]
    steps = {name: [] for name in algos}
    failures = {name: [] for name in algos}

    baseline = {}
    for name, cls in {'Q-Learning': QLearningAgent, 'SARSA': SARSAAgent}.items():
        logs, agents = run_many(cls, 1000)
        row(f'default epsilon {name}', logs, agents)
        baseline[name] = (1 - logs[..., 2]).sum(1).mean()

    def build(cls, tau):
        agent = cls()
        agent.tau = tau
        return agent

    for tau in taus:
        for name, cls in algos.items():
            logs, agents = run_many(lambda: build(cls, tau), 1000)
            row(f'tau={tau} {name}', logs, agents)
            steps[name].append(logs[:, -100:, 1].mean())
            failures[name].append((1 - logs[..., 2]).sum(1).mean())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for name in algos:
        axes[0].plot(taus, failures[name], marker='o', color=COLORS[name], label=f'{name}, softmax')
        axes[0].axhline(baseline[name], color=COLORS[name], linestyle='--', linewidth=1.2,
                        label=f'{name}, default epsilon-greedy')
        axes[1].plot(taus, steps[name], marker='o', color=COLORS[name], label=name)
    axes[0].set(xlabel='Temperature tau', ylabel='Failed episodes out of 1000', xscale='log')
    axes[1].set(xlabel='Temperature tau', ylabel='Steps per episode (last 100)', xscale='log')
    for ax in axes:
        ax.set_xticks(taus, taus)
    axes[0].legend()
    fig.suptitle(f'Softmax exploration (mean of {SEEDS} seeds)')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig4_softmax.png')


if __name__ == '__main__':
    FIG_DIR.mkdir(exist_ok=True)
    default_params()
    epsilon_sweep()
    softmax_sweep()
