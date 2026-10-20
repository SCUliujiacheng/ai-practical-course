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
# ## 1. 地图、状态与合法动作
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

import heapq
from collections import deque
grid = np.array([[0, 0, 0, 1, 0, 0], [1, 1, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0]], dtype=int)
start = (0, 0)
goal = (4, 5)

def neighbors(node, board):
    r, c = node
    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nr, nc = (r + dr, c + dc)
        if 0 <= nr < board.shape[0] and 0 <= nc < board.shape[1] and (board[nr, nc] == 0):
            yield (nr, nc)

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
assert manhattan(start, goal) == 9
np.savetxt(DATA_DIR / 'map.csv', grid, fmt='%d', delimiter=',')

# %%
# ## 2. 优先队列与 g+h
def astar(board, source, target, use_heuristic=True):
    for p in [source, target]:
        if not (0 <= p[0] < board.shape[0] and 0 <= p[1] < board.shape[1]) or board[p] != 0:
            raise ValueError('起终点必须可通行')
    h = (lambda p: manhattan(p, target)) if use_heuristic else lambda p: 0
    queue = [(h(source), 0, source)]
    g = {source: 0}
    parent = {}
    expanded = 0
    while queue:
        _, distance, node = heapq.heappop(queue)
        if distance != g[node]:
            continue
        expanded += 1
        if node == target:
            path = [node]
            while node != source:
                node = parent[node]
                path.append(node)
            return (list(reversed(path)), expanded)
        for nxt in neighbors(node, board):
            tentative = distance + 1
            if tentative < g.get(nxt, float('inf')):
                g[nxt] = tentative
                parent[nxt] = node
                heapq.heappush(queue, (tentative + h(nxt), tentative, nxt))
    return ([], expanded)
path, expanded = astar(grid, start, goal)

# %%
# ## 3. 独立验证每一步与最短长度
def validate_path(board, path, source, target):
    return bool(path and path[0] == source and (path[-1] == target) and all((board[p] == 0 for p in path)) and all((b in set(neighbors(a, board)) for a, b in zip(path, path[1:]))))
reference, reference_expanded = astar(grid, start, goal, use_heuristic=False)
assert validate_path(grid, path, start, goal)
assert len(path) == len(reference)
blocked = np.array([[0, 1], [1, 0]])
assert astar(blocked, (0, 0), (1, 1))[0] == []

# %%
# ## 4. 输出路线和搜索工作量
fig, ax = plt.subplots()
ax.imshow(grid, cmap='Greys', vmin=0, vmax=1)
ax.plot([p[1] for p in path], [p[0] for p in path], '-o', color='tab:red')
ax.set_title('A* 最短可行路线')
ax.text(start[1], start[0], 'S', color='blue')
ax.text(goal[1], goal[0], 'G', color='blue')
fig.tight_layout()
fig.savefig(OUT_DIR / 'route.png')
plt.close(fig)
(OUT_DIR / 'path.json').write_text(json.dumps(path), encoding='utf-8')
finish({'steps': len(path) - 1, 'astar_expanded': expanded, 'dijkstra_expanded': reference_expanded, 'path_valid': True, 'unreachable_checked': True})