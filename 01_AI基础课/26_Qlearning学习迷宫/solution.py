from pathlib import Path
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
CASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
COURSE_ROOT = CASE_DIR.parents[1]
DATA_DIR = CASE_DIR / "data"
OUT_DIR = CASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                     "figure.figsize": (8, 4.5), "figure.dpi": 110})


# %%
# ## 1. 环境状态、动作与随机转移
import json
import time
import numpy as np
started = time.perf_counter()
def finish(metrics):
    metrics["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
def barplot(labels, values, title, filename="comparison.png"):
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename)
    plt.close(fig)

board = np.array([[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 1, 0], [1, 1, 0, 0, 0], [0, 0, 0, 1, 0]])
start = (0, 0)
goal = (4, 4)
moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def transition(state, action, rng, grid=board, slip=0.1):
    if rng.random() < slip:
        action = int(rng.integers(4))
    dr, dc = moves[action]
    candidate = (state[0] + dr, state[1] + dc)
    valid = 0 <= candidate[0] < grid.shape[0] and 0 <= candidate[1] < grid.shape[1] and (grid[candidate] == 0)
    nxt = candidate if valid else state
    done = nxt == goal
    return (nxt, 10.0 if done else -0.1 if valid else -0.3, done)

def state_id(state):
    return state[0] * board.shape[1] + state[1]

# %%
# ## 2. Bellman 更新与探索
def learn(episodes=1600, seed=26, gamma=0.95):
    rng = np.random.default_rng(seed)
    Q = np.zeros((board.size, 4))
    returns = []
    successes = []
    for episode in range(episodes):
        state = start
        total = 0.0
        done = False
        epsilon = max(0.05, 0.9 * (1 - episode / episodes))
        for step in range(70):
            sid = state_id(state)
            action = int(rng.integers(4)) if rng.random() < epsilon else int(Q[sid].argmax())
            nxt, reward, done = transition(state, action, rng)
            target = reward if done else reward + gamma * Q[state_id(nxt)].max()
            Q[sid, action] += 0.15 * (target - Q[sid, action])
            state = nxt
            total += reward
            if done:
                break
        returns.append(total)
        successes.append(done)
    return (Q, np.array(returns), np.array(successes))
Q, returns, successes = learn()
assert np.isfinite(Q).all()

# %%
# ## 3. 冻结策略后独立评价
def evaluate(Q, random_policy=False, seed=1026, episodes=180, grid=board):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(episodes):
        state = start
        total = 0.0
        done = False
        for step in range(70):
            action = int(rng.integers(4)) if random_policy else int(Q[state_id(state)].argmax())
            state, reward, done = transition(state, action, rng, grid)
            total += reward
            if done:
                break
        rows.append((int(done), step + 1, total))
    return np.array(rows)
learned = evaluate(Q)
random_result = evaluate(Q, random_policy=True)
assert learned.shape == (180, 3)

# %%
# ## 4. 学习曲线与策略输出
fig, ax = plt.subplots()
ax.plot(np.convolve(returns, np.ones(80) / 80, mode='valid'))
ax.set_xlabel('训练回合')
ax.set_ylabel('80 回合平均回报')
ax.set_title('Q-learning 学习曲线')
fig.tight_layout()
fig.savefig(OUT_DIR / 'learning.png')
plt.close(fig)
np.save(OUT_DIR / 'q_table.npy', Q)
finish({'learned_success_rate': float(learned[:, 0].mean()), 'random_success_rate': float(random_result[:, 0].mean()), 'learned_mean_steps': float(learned[:, 1].mean()), 'learned_mean_return': float(learned[:, 2].mean()), 'training_episodes': len(returns), 'evaluation_episodes': len(learned)})