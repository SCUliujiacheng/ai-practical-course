"""制作 A01—A10、A21—A25；完整算法写进每个案例，不在共享库中隐藏。"""
from textwrap import dedent
from course_builder import write_case


BASE = '''import json
import itertools
import math
import numpy as np
import pandas as pd

def save_results(metrics, recommendation):
    """只负责保存实际计算结果；决策逻辑在下文。"""
    def convert(value):
        return value.item() if hasattr(value, "item") else str(value)
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=convert), encoding="utf-8")
    (OUT_DIR / "决策建议.md").write_text(
        "# 本次运行的决策建议\\n\\n" + recommendation +
        "\\n\\n数据为本例明确给定的教学场景；结论仅对模型、目标与已检查条件成立。\\n",
        encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=convert))
    print(recommendation)
'''


def sec(text, code):
    return text, dedent(code).strip() + '\n'


def first(text, code):
    return text, BASE + '\n' + dedent(code).strip() + '\n'


def exercise(title, prompt, answer):
    return dict(title=title, prompt=prompt,
                starter='# TODO：先写预期行为，再实现。答案包含可运行检查。\n',
                answer=dedent(answer).strip() + '\n')


# 每题给出真实输入与可运行检查；None 明确表示尚未完成，不能把顺序运行当成完成练习。
# 三元组依次是输入、待完成内容、完成后的检查。algorithm 仍在案例本身。
SCAFFOLDS = {
1: [
('trial_values = [[2,3],[3,4],[1,5],[2,3]]', '用 pareto_mask 返回布尔数组', 'assert attempt.tolist() == [True,False,True,True]'),
('trial_limits = dict(limits, cost=950)', '先用 feasible_mask 筛选，再返回保留的 plan 名称列表', 'assert attempt == ["均衡","节省"]'),
('trial_computers = np.array([[5000,1.4],[6500,1.2],[7000,1.8]])', '返回这三台电脑的 Pareto 标记数组', 'assert attempt.tolist() == [True,True,False]')],
2: [
('trial_choices = [[3], [0,3]]', '逐个调用 valid_choice，返回两个布尔值组成的列表；预算和工时上限均为100', 'assert attempt == [False,True]'),
('trial_budget = 20', '调用 optimize_tasks，返回 (状态, 选择的任务索引列表)', 'trial_state, trial_picks = attempt\nassert trial_picks is not None\nassert valid_choice(trial_picks,tasks,trial_budget,hours,dependencies)\nassert sum(tasks[i]["value"] for i in trial_picks) >= value'),
('trial_units = [{"cost":c,"hours":h,"value":v} for c,h,v in [(2,2,3),(2,2,4),(3,3,6)]]', '预算与工时均为4，依赖为[(2,0)]，返回 optimize_tasks 的结果', 'trial_state, trial_picks = attempt\nassert trial_state == "OPTIMAL" and set(trial_picks) == {0,1}')],
3: [
('trial_assignments = [[0,0,2,3], [0,1,2,3]]', '用 check_assignment 分别检查，返回布尔列表', 'assert attempt == [False,True]'),
('trial_allowed = allowed.copy()\ntrial_allowed[0,1] = False', '用 sum_assignment 返回新分配；每个列表位置表示一名成员', 'assert check_assignment(attempt,durations,trial_allowed)\nassert sum(durations[i,j] for i,j in enumerate(attempt)) == enumerate_assignment(durations,trial_allowed)[0]'),
('trial_machines = np.array([[1,3],[2,8]])\ntrial_allowed = np.ones((2,2),bool)', '分别最小化总工时和最大工时，返回 (总工时最优值, 最大工时最优值)', 'assert attempt == (5,3)')],
4: [
('trial_lengths = {"X":2,"Y":3}\ntrial_edges = [("X","Y")]', '返回 critical_times 的 (总工期, 最早开始字典, 最晚开始字典)', 'trial_span,trial_es,trial_ls=attempt\nassert trial_span == 5 and trial_es == {"X":0,"Y":2} and trial_es == trial_ls'),
('trial_workers = 3', '返回 resource_schedule 的 (状态, 开始时间字典)', 'trial_state,trial_plan=attempt\nassert check_schedule(trial_plan,lengths,edges,trial_workers)\nassert max(trial_plan[j]+lengths[j] for j in lengths) >= lower_bound'),
('trial_edges = [("data","train"),("train","data")]', '用 try/except 调用 critical_times；若正确捕获 ValueError，将 attempt 设置为 True', 'assert attempt is True')],
5: [
('trial_pairs = [((0,1,3),(1,2,4)), ((0,0,2),(1,2,4))]', '用 jobs[0]、jobs[2] 和 conflict 检查两对预约，返回布尔列表', 'assert attempt == [True,False]'),
('trial_jobs = [dict(j,starts=[0,1]) for j in impossible]', '调用 solve_booking 并返回 (状态, 预约方案)', 'trial_state,trial_plan=attempt\nassert check_booking(trial_plan,trial_jobs)'),
('trial_jobs = [dict(name=n,duration=2,rooms=[0],starts=[0,2],members={n}) for n in ["A","B"]]', '同一相机视作房间0，返回 solve_booking 的结果', 'trial_state,trial_plan=attempt\nassert check_booking(trial_plan,trial_jobs)\nassert sorted(c[1] for c in trial_plan.values()) == [0,2]')],
6: [
('trial_path = ["S","A","G"]', '调用 path_totals，返回 (总时间, 总耗电)', 'assert attempt == (2,10) and attempt[1] > battery'),
('trial_capacity = 10', '返回 battery_route(roads,"S","G",容量) 的完整四元组', 'trial_path,trial_time,trial_energy,_=attempt\nassert trial_time == 2 and trial_energy <= trial_capacity\nassert path_totals(trial_path,roads) == (trial_time,trial_energy)'),
('trial_capacity = 4', '把耗电解释为可累加风险，返回 battery_route 的完整结果', 'trial_path,trial_time,trial_risk,_=attempt\nassert trial_path is not None and trial_risk <= trial_capacity\nassert path_totals(trial_path,roads) == (trial_time,trial_risk)')],
7: [
('trial_paths = [[(0,0),(0,1)],[(0,1),(0,0)]]', '调用 first_conflict 返回冲突记录', 'assert attempt[0] == "e" and attempt[3] == 1'),
('trial_orders = list(itertools.permutations(range(3)))', '对每种顺序调用 prioritized；返回6项路径方案组成的列表，无解项保留 None', 'assert len(attempt) == 6\nassert all(p is None or validate_paths(p) for p in attempt)'),
('trial_bad = [[(0,0),(0,1)],[(1,2),(0,2),(0,1)]]\ntrial_good = [[(0,0),(0,1)],[(1,2),(1,1),(1,0)]]', '分别调用 first_conflict，返回 (坏方案冲突, 好方案冲突)', 'assert attempt[0][0] == "v" and attempt[1] is None')],
8: [
('trial_graph = initial.copy()\nobserve(trial_graph,1,set())', '检查 old_path 中已不存在的边，返回边的列表', 'assert attempt == [("A","G")]'),
('trial_policies = ["fixed","repair_blocked","rolling"]', '暂时把 events 设为空，用 try/finally 恢复原值，返回三个 execute 结果的列表', 'assert all(r["success"] and r["time"] == 2 and r["changes"] == 0 for r in attempt)'),
('import copy\ntrial_log = copy.deepcopy(results[2])\ntrial_log["trace"][0]["duration"] += 3', '分别验证篡改日志和原日志，返回两个布尔值的列表', 'assert attempt == [False,True]')],
9: [
('trial_start = initial_state', '捕获直接开门的 ValueError，再执行取钥匙与开门；返回 (是否拒绝直接开门, 最终状态)', 'assert attempt[0] is True and "door-open" in attempt[1]'),
('trial_start = set(initial_state)-{"package-hall"}', '调用 plan_actions，返回 (计划或None, 搜索扩展数)', 'assert attempt[0] is None'),
('trial_actions = [action("calibrate",["ready"],["calibrated"],[]),\n                 action("experiment",["calibrated"],["measured"],[]),\n                 action("report",["measured"],["reported"],[])]', '从 {ready} 到 {reported} 调用 plan_actions，返回计划列表', 'assert attempt == ["calibrate","experiment","report"]')],
10: [
('trial_roster = roster.copy()', '返回 (每时段人数数组, 每人班次数数组)，注意两个 sum 的 axis', 'trial_coverage,trial_loads=attempt\nassert trial_coverage.tolist() == demand and trial_loads.max() <= 2'),
('trial_preferences = np.zeros_like(preferences)', '返回 schedule(availability,新偏好,skilled,demand) 的结果', 'trial_state,trial_roster=attempt\nassert check_roster(trial_roster,availability,skilled,demand)\nassert np.ptp(trial_roster.sum(axis=1)) == 1'),
('trial_availability = np.ones((6,3),int)\ntrial_preferences = np.zeros_like(trial_availability)', '合格者为{0,1}，需求[2,2,2]；返回 schedule 的结果，并思考人人一次为何不可行', 'trial_state,trial_roster=attempt\nassert check_roster(trial_roster,trial_availability,{0,1},[2,2,2])\nassert trial_roster[:2].sum() >= 3')],
21: [
('trial_profiles = [([1,0],[1,0]), ([0,1],[0,1])]', '用 check_equilibrium 分别检查，返回布尔列表', 'assert attempt == [True,False]'),
('trial_A=A.copy();trial_B=B.copy()\ntrial_A[0,:]-=.5;trial_B[:,0]-=.5', '调用 pure_equilibria 返回新收益矩阵的纯均衡列表', 'assert attempt == [(0,0)]'),
('trial_coin=np.array([[1.,-1.],[-1.,1.]])', '用 Nashpy 搜索后返回一个均衡 (行玩家概率数组, 列玩家概率数组)', 'assert check_equilibrium(trial_coin,-trial_coin,*attempt)\nassert np.allclose(attempt[0],[.5,.5]) and np.allclose(attempt[1],[.5,.5])')],
22: [
('trial_contributions=[1,0,1,0]', '调用 payoffs 返回每人收益数组', 'assert np.allclose(attempt,[.8,1.8,.8,1.8]) and np.isclose(sum(attempt),5.2)'),
('trial_attendance=np.ones((20,4),bool)', '分别模拟全员 rotation、全员 cooperate，返回 (轮值记录, 全员合作记录)', 'trial_rotation,trial_fixed=attempt\nassert all(r["shortage"] == 0 for r in trial_rotation)\nassert sum(r["contributed"] for r in trial_rotation) == 40\nassert sum(r["contributed"] for r in trial_fixed) == 80'),
('trial_attendance=np.ones((6,4),bool)', '比较全员 defect 和全员 conditional；返回两个实验的总缺岗数', 'assert attempt == (12,0)')],
23: [
('trial_full=[0]*players\ntrial_switched=trial_full.copy();trial_switched[0]=1', '返回 (全走短路是否均衡, 玩家0换路是否降低个人代价)', 'assert attempt == (False,True)'),
('trial_initial=[1]*players', '调用 best_response，返回 (状态,最终路线列表,改进历史)', 'trial_state,trial_plan,trial_history=attempt\nassert trial_state == "EQUILIBRIUM" and equilibrium(trial_plan)\nassert all(b["potential"] < a["potential"] for a,b in zip(trial_history,trial_history[1:]))'),
('trial_short=[0]*6;trial_long=[1]*6', '返回两个极端路线方案的社会总耗时，并解释全员同时改道的循环', 'assert attempt == (36,30)')],
24: [
('trial_bids=[3,10,7]', '调用 second_price 返回 (赢家索引,支付价)', 'assert attempt == (1,7)'),
('trial_bids=values.copy();trial_bids[3]=12', '先求赢家与支付，再用真实 values 求效用；返回该赢家的效用', 'assert attempt == -7'),
('trial_bids=[5,5,1]', '调用 second_price 并解释当前平局规则，返回 (赢家,支付)', 'assert attempt == (0,5)')],
25: [
('trial_allocations=list(schemes.values())', '返回每种方案的分配额总和组成的列表', 'assert len(attempt) == len(trial_allocations) and np.allclose(attempt,12)'),
('trial_worth=dict(worth);trial_worth[frozenset([0,1])]=10.', '调用 shapley 返回新的三人分配数组', 'assert np.isclose(sum(attempt),12)\nassert attempt[0] > allocation[0] and attempt[1] > allocation[1] and attempt[2] < allocation[2]'),
('trial_worth={s:(6. if {0,1}<=s else 0.) for s in all_coalitions(3)}', '调用 shapley 并解释第三位参与者为何所得为0', 'assert np.allclose(attempt,[3,3,0])')]
}


def scaffold(number, index):
    inputs, instruction, checks = SCAFFOLDS[number][index]
    return (inputs + '\n# TODO：' + instruction + '\nattempt = None\n'
            'if attempt is None:\n    print("尚未完成：请替换 attempt=None，再运行检查。")\nelse:\n'
            + '\n'.join('    '+line for line in checks.splitlines())
            + '\n    print("本题自动检查通过；继续用自己的话解释结果。")\n')


COMMON = dict(track='A', prereqs='已完成 AI 基础课；会使用列表、字典、函数和 NumPy。涉及的新结构在本例分步解释。',
              decision='运行后打开 outputs/决策建议.md；建议中的数值来自本次求解，不是预写的提升幅度。',
              data_note='原创教学模拟场景，规模与全部参数直接列在 solution.py；运行同时保存输入或结果 CSV。无真实个人记录，不需要下载数据。')


def make(number, slug, title, intro, sections, exercises, sources, pitfalls, extra_files=None):
    for index, item in enumerate(exercises):
        item['starter'] = scaffold(number, index)
    return write_case(number=number, slug=slug, title=title, intro=intro,
                      sections=sections, exercises=exercises, sources=sources,
                      pitfalls=pitfalls, extra_files=extra_files, **COMMON)


def build_a01():
    make(1, '多目标选择团队计划', '三份团队计划，应该选哪一份？',
         '你负责推荐一份团队计划。输入是工期、成本、成员最大负担；输出必须说明可行性和取舍，不能用一个未经解释的总分代替判断。', [
        first('## 1. 把文字要求变成表格\n每行是一份方案，每列是一个越小越好的指标。硬限制先过滤；权重只能比较满足硬限制的方案。`axis=1` 表示沿一行的各列计算。', '''
            plans = pd.DataFrame({"plan": ["快速", "均衡", "节省"],
                                  "days": [7, 9, 11], "cost": [1200, 900, 700],
                                  "max_hours": [36, 27, 24]})
            limits = {"days": 12, "cost": 1250, "max_hours": 40}
            columns = list(limits)
            def feasible_mask(frame, caps):
                return np.array([all(row[key] <= cap for key, cap in caps.items())
                                 for _, row in frame.iterrows()])
            candidates = plans.loc[feasible_mask(plans, limits)].copy()
            assert not candidates.empty
            plans.to_csv(DATA_DIR / "计划.csv", index=False, encoding="utf-8-sig")
            print(candidates)
        '''),
        sec('## 2. 先检查支配关系\n若方案 i 在所有指标上都不比 j 差，且至少一项更好，就说 i 支配 j。双层循环刻意保留，方便逐对手算；Pareto 集不等于唯一答案。', '''
            def pareto_mask(values):
                values = np.asarray(values, dtype=float)
                return np.array([not any(np.all(values[j] <= values[i]) and
                                         np.any(values[j] < values[i])
                                         for j in range(len(values)) if j != i)
                                 for i in range(len(values))])
            candidates["pareto"] = pareto_mask(candidates[columns].to_numpy())
            assert pareto_mask([[1, 2], [2, 3]]).tolist() == [True, False]
            print(candidates[["plan", "pareto"]])
        '''),
        sec('## 3. 用明确权重比较，再做敏感性分析\n单位不同不能直接相加。用本表最小最大值归一化，再计算权重和。这个缩放依赖候选集，新增方案时应重新检查。权重由决策者给定，不是数据自动发现的真理。', '''
            values = candidates[columns].to_numpy(float)
            span = np.ptp(values, axis=0)
            normalized = (values - values.min(axis=0)) / np.where(span == 0, 1, span)
            def choose(weights):
                weights = np.asarray(weights, dtype=float)
                assert len(weights) == len(columns) and np.all(weights >= 0)
                assert np.isclose(weights.sum(), 1)
                scores = normalized @ weights
                return int(np.argmin(scores)), scores
            selected, scores = choose([0.5, 0.3, 0.2])
            candidates["score"] = scores
            sensitivity = []
            for urgency in np.linspace(0, 1, 21):
                idx, _ = choose([urgency, 0.6 * (1 - urgency), 0.4 * (1 - urgency)])
                sensitivity.append({"urgency_weight": urgency,
                                    "chosen": candidates.iloc[idx]["plan"]})
            sensitivity = pd.DataFrame(sensitivity)
            assert feasible_mask(candidates, limits).all()
        '''),
        sec('## 4. 输出可复核的建议\n保存原始指标、分数和随权重变化的选择。图上没有“绝对最好”的位置；建议同时交代当前权重与何时重新评估。', '''
            candidates.to_csv(OUT_DIR / "方案对照.csv", index=False, encoding="utf-8-sig")
            sensitivity.to_csv(OUT_DIR / "权重敏感性.csv", index=False, encoding="utf-8-sig")
            ax = candidates.set_index("plan")["score"].plot.bar(color=["#467a9e"] * len(candidates))
            ax.set_ylabel("归一化加权代价（越小越好）")
            ax.set_title("权重：工期0.5、成本0.3、负担0.2")
            plt.tight_layout(); plt.savefig(OUT_DIR / "方案比较.png"); plt.close()
            chosen = candidates.iloc[selected]
            save_results({"chosen": chosen["plan"], "score": float(chosen["score"]),
                          "feasible_count": len(candidates),
                          "pareto_count": int(candidates["pareto"].sum())},
                         f"在工期/成本/负担权重为0.5/0.3/0.2时，推荐{chosen['plan']}方案，"
                         f"工期{chosen['days']}天、成本{chosen['cost']}、最大负担{chosen['max_hours']}小时。"
                         f"当前有{int(candidates['pareto'].sum())}个非支配方案；权重或预算变化时重新比较。")
        ''')], [
        exercise('补全：谁被支配？', '计算四个二维候选的 Pareto 标记，并解释相同候选为什么不会互相严格支配。', '''
            answer = pareto_mask([[2, 3], [3, 4], [1, 5], [2, 3]])
            assert answer.tolist() == [True, False, True, True]
            print(answer)
        '''),
        exercise('修改：预算突然减少', '预算变为950，其他硬限制不变。不要仅在原始总分里加罚分；先重新筛选可行方案。', '''
            changed = dict(limits, cost=950)
            kept = plans.loc[feasible_mask(plans, changed), "plan"].tolist()
            assert kept == ["均衡", "节省"]
            assert plans.loc[feasible_mask(plans, changed), "cost"].max() <= 950
            print(kept)
        '''),
        exercise('迁移：选电脑', '三台电脑价格、重量都越小越好：[[5000,1.4],[6500,1.2],[7000,1.8]]。找出非支配候选，提出尚缺哪些性能信息。', '''
            computers = np.array([[5000, 1.4], [6500, 1.2], [7000, 1.8]])
            mask = pareto_mask(computers)
            assert mask.tolist() == [True, True, False]
            print("候选", computers[mask], "；还需明确性能、续航等硬要求。")
        ''')], [('pymoo：有约束双目标问题与Pareto解', 'https://pymoo.org/getting_started/part_1.html')],
         '加权和依赖单位、归一化与权重；支配判定只对已列入的指标有效。不能把主观价值评分解释为客观事实。')


def build_a02():
    make(2, '有限预算选择任务', '时间与预算有限，应完成哪些任务？',
         '给定10项任务的预算、工时和教学价值评分，选择一组任务。先理解单预算背包，再把工时和依赖加入模型；输出是任务组合及其合法性证明式检查。', [
        first('## 1. 明确输入与评分来源\n用列表保存任务，字典字段让每个数字有名字。价值是本例假设评分。依赖 `(a,b)` 表示选择 a 必须同时选择 b。', '''
            from ortools.sat.python import cp_model
            tasks = [{"name": f"任务{i}", "cost": c, "hours": h, "value": v}
                     for i, (c, h, v) in enumerate([(3,2,6),(4,3,7),(2,2,4),(5,4,10),(3,1,5),
                                                   (6,5,11),(2,1,3),(4,2,8),(3,3,5),(5,3,9)])]
            budget, hours = 16, 12
            dependencies = [(3, 0), (5, 2), (7, 1)]
            pd.DataFrame(tasks).to_csv(DATA_DIR / "候选任务.csv", index=False, encoding="utf-8-sig")
            def valid_choice(chosen, jobs, money, time_limit, deps):
                if len(set(chosen)) != len(chosen): return False
                chosen = set(chosen)
                return (all(0 <= i < len(jobs) for i in chosen)
                        and sum(jobs[i]["cost"] for i in chosen) <= money
                        and sum(jobs[i]["hours"] for i in chosen) <= time_limit
                        and all(a not in chosen or b in chosen for a, b in deps))
            assert not valid_choice([0,0],tasks,100,100,dependencies)
        '''),
        sec('## 2. 单预算背包：理解动态规划\n`dp[b]` 表示预算最多 b 时的最高价值。预算倒序更新，防止同一个任务被重复选取。这一步暂时没有工时和依赖，不能把结果误用到完整任务。', '''
            def knapsack_value(jobs, cap):
                dp = [0] * (cap + 1)
                for job in jobs:
                    for b in range(cap, job["cost"] - 1, -1):
                        dp[b] = max(dp[b], dp[b-job["cost"]] + job["value"])
                return dp[-1]
            unconstrained_value = knapsack_value(tasks, budget)
            assert knapsack_value([{"cost": 2, "value": 3}], 4) == 3
            print("只有预算时的价值上界：", unconstrained_value)
        '''),
        sec('## 3. CP-SAT 同时处理多个约束\n0/1变量表示是否选中。`x[a] <= x[b]` 精确表达依赖。检查器根据原始任务重新累加，不以求解器返回状态代替需求检查。', '''
            def optimize_tasks(jobs, money, time_limit, deps, require_all=False):
                model = cp_model.CpModel()
                x = [model.new_bool_var(f"pick_{i}") for i in range(len(jobs))]
                model.add(sum(j["cost"] * x[i] for i,j in enumerate(jobs)) <= money)
                model.add(sum(j["hours"] * x[i] for i,j in enumerate(jobs)) <= time_limit)
                for a,b in deps: model.add(x[a] <= x[b])
                if require_all:
                    for variable in x: model.add(variable == 1)
                model.maximize(sum(j["value"] * x[i] for i,j in enumerate(jobs)))
                solver = cp_model.CpSolver()
                solver.parameters.max_time_in_seconds = 5
                solver.parameters.num_search_workers = 1
                status = solver.solve(model)
                chosen = ([i for i,var in enumerate(x) if solver.value(var)]
                          if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None)
                return solver.status_name(status), chosen
            status, chosen = optimize_tasks(tasks, budget, hours, dependencies)
            assert chosen is not None, f"未获得方案：{status}。UNKNOWN表示尚未找到或证明，不能解释为无解。"
            assert valid_choice(chosen, tasks, budget, hours, dependencies)
            # 10个任务只有1024种组合，用独立穷举核对小实例。
            feasible = [list(i for i in range(len(tasks)) if bits[i])
                        for bits in itertools.product([0,1], repeat=len(tasks))
                        if valid_choice([i for i,b in enumerate(bits) if b], tasks, budget, hours, dependencies)]
            value = sum(tasks[i]["value"] for i in chosen)
            exact = max(sum(tasks[i]["value"] for i in c) for c in feasible)
            assert value == exact and value <= unconstrained_value
            impossible, _ = optimize_tasks(tasks, 0, 0, dependencies, require_all=True)
            assert impossible == "INFEASIBLE"
        '''),
        sec('## 4. 与简单选择法比较并保存计划\n简单基线按价值/预算排序，在满足当前约束时才加入。它可能错过需要一起选择的依赖组合。图展示本例真实计算，不宣称求解器在所有问题上都更快。', '''
            greedy = []
            for i in sorted(range(len(tasks)), key=lambda i: tasks[i]["value"]/tasks[i]["cost"], reverse=True):
                if valid_choice(greedy+[i], tasks, budget, hours, dependencies): greedy.append(i)
            greedy_value = sum(tasks[i]["value"] for i in greedy)
            pd.DataFrame([tasks[i] for i in chosen]).to_csv(OUT_DIR / "任务计划.csv", index=False, encoding="utf-8-sig")
            plt.bar(["简单排序", "约束优化"], [greedy_value, value]); plt.ylabel("假设价值总分")
            plt.tight_layout(); plt.savefig(OUT_DIR / "任务选择.png"); plt.close()
            save_results({"status":status,"selected":chosen,"value":value,"greedy_value":greedy_value,
                          "cost":sum(tasks[i]["cost"] for i in chosen),"hours":sum(tasks[i]["hours"] for i in chosen),
                          "infeasible_test":impossible},
                         f"推荐任务{chosen}，价值{value}，基线{greedy_value}；求解状态{status}。"
                         "已逐条检查预算、工时和依赖，并用1024个组合核对本小实例最优值。评分变化时需要重算。")
        ''')], [
        exercise('补全：依赖检查', '任务3依赖任务0。比较[3]与[0,3]，其余条件宽松。', '''
            assert not valid_choice([3], tasks, 100, 100, dependencies)
            assert valid_choice([0,3], tasks, 100, 100, dependencies)
            print("选中依赖任务，必须先包含其前置任务。")
        '''),
        exercise('修改：预算增加但工时不变', '将预算增至20，检查新方案。不要预先断言价值一定严格增加。', '''
            state, picks = optimize_tasks(tasks, 20, hours, dependencies)
            assert picks is not None and valid_choice(picks, tasks, 20, hours, dependencies)
            changed_value = sum(tasks[i]["value"] for i in picks)
            assert changed_value >= value
            print(state, picks, changed_value)
        '''),
        exercise('迁移：选学习单元', '三个单元(成本,时间,价值)=(2,2,3),(2,2,4),(3,3,6)，预算4、时间4，第三单元依赖第一个。', '''
            units = [{"cost":c,"hours":h,"value":v} for c,h,v in [(2,2,3),(2,2,4),(3,3,6)]]
            state, picks = optimize_tasks(units, 4, 4, [(2,0)])
            assert set(picks) == {0,1} and state == "OPTIMAL"
            print("学习组合", picks)
        ''')], [('CP-SAT Solver 官方示例', 'https://developers.google.com/optimization/cp/cp_solver')],
         '普通一维背包不处理任意任务依赖。所有成本和时间使用整数单位。UNKNOWN表示尚未确定，不等于INFEASIBLE；评分是显式假设。')


def build_a03():
    make(3, '团队任务指派', '总工时最少与团队最早完工是一回事吗？',
         '四人同时开始、各做一项任务。你要分别求总工时最少和最晚完成时间最小的方案，再检查技能限制。', [
        first('## 1. 用矩阵表达成员与任务\n行是人，列是任务，元素是所需小时。`allowed` 是独立布尔矩阵，False代表不能接任务。数字大表示慢，不等于不允许。', '''
            from scipy.optimize import linear_sum_assignment
            durations = np.array([[1,4,20,20],[4,6,20,20],[20,20,1,4],[20,20,4,6]], dtype=int)
            allowed = np.ones_like(durations, dtype=bool)
            pd.DataFrame(durations, index=["甲","乙","丙","丁"], columns=list("ABCD")).to_csv(DATA_DIR / "耗时.csv", encoding="utf-8-sig")
            def check_assignment(permutation, matrix, mask):
                return (permutation is not None and len(permutation) == len(matrix)
                        and sorted(permutation) == list(range(len(matrix)))
                        and all(mask[i,j] for i,j in enumerate(permutation)))
        '''),
        sec('## 2. 先枚举，得到可检查的基准\n4!只有24种分配。`enumerate`同时给出成员编号和任务编号。总工时用sum，最晚完成用max，这两个目标不能混用。', '''
            def enumerate_assignment(matrix, mask, objective="sum"):
                candidates = []
                for assignment in itertools.permutations(range(len(matrix))):
                    if check_assignment(assignment, matrix, mask):
                        times = [int(matrix[i,j]) for i,j in enumerate(assignment)]
                        score = sum(times) if objective == "sum" else max(times)
                        candidates.append((score, assignment))
                return min(candidates) if candidates else (None, None)
            sum_value, sum_plan = enumerate_assignment(durations, allowed, "sum")
            end_value, end_plan = enumerate_assignment(durations, allowed, "max")
            assert check_assignment(sum_plan, durations, allowed)
            assert check_assignment(end_plan, durations, allowed)
        '''),
        sec('## 3. 使用线性和指派，并独立核对\nSciPy函数求的是总成本最小。给不允许的组合置无穷值；若无可行完整指派，会抛出ValueError。不能将它直接当成最晚完工时间求解器。', '''
            def sum_assignment(matrix, mask):
                costs = np.where(mask, matrix.astype(float), np.inf)
                try:
                    rows, cols = linear_sum_assignment(costs)
                except ValueError:
                    return None
                assignment = tuple(int(c) for c in cols)
                return assignment if check_assignment(assignment, matrix, mask) else None
            solver_plan = sum_assignment(durations, allowed)
            assert sum(durations[i,j] for i,j in enumerate(solver_plan)) == sum_value
            impossible_mask = allowed.copy(); impossible_mask[0,:] = False
            assert sum_assignment(durations, impossible_mask) is None
            assert enumerate_assignment(durations, impossible_mask)[1] is None
        '''),
        sec('## 4. 输出分工和两类目标\n图的横轴是成员，条长是其耗时。所有人同时开工这一假设决定了max的含义；若有人晚到或任务有依赖，必须换模型。', '''
            rows = []
            for label, plan in [("最少总工时", sum_plan), ("最早团队完工", end_plan)]:
                for worker, job in enumerate(plan):
                    rows.append({"objective":label,"worker":worker,"task":job,"hours":int(durations[worker,job])})
            table = pd.DataFrame(rows); table.to_csv(OUT_DIR / "分工对照.csv", index=False, encoding="utf-8-sig")
            table.pivot(index="worker", columns="objective", values="hours").plot.bar()
            plt.ylabel("工时"); plt.tight_layout(); plt.savefig(OUT_DIR / "分工.png"); plt.close()
            sum_finish = max(int(durations[i,j]) for i,j in enumerate(sum_plan))
            save_results({"minimum_total_hours":sum_value,"sum_optimal_finish":sum_finish,
                          "minimum_finish_time":end_value,"sum_plan":sum_plan,"finish_plan":end_plan},
                         f"若目标是尽早全部完成，选择{end_plan}，{end_value}小时完工；"
                         f"总工时方案{sum_plan}的总耗时为{sum_value}、完工时间为{sum_finish}。先确认团队真正要优化哪个指标。")
        ''')], [
        exercise('补全：识别重复任务', '合法分配必须每人一项、每项一人，检查两份分配。', '''
            assert not check_assignment([0,0,2,3], durations, allowed)
            assert check_assignment([0,1,2,3], durations, allowed)
        '''),
        exercise('修改：成员甲不能做任务B', '修改技能矩阵重新求总工时最优，并与枚举核对。', '''
            changed = allowed.copy(); changed[0,1] = False
            p = sum_assignment(durations, changed)
            optimum, _ = enumerate_assignment(durations, changed)
            assert check_assignment(p, durations, changed)
            assert sum(durations[i,j] for i,j in enumerate(p)) == optimum
            print(p, optimum)
        '''),
        exercise('迁移：两台设备加工两件物品', '耗时[[1,3],[2,8]]。分别比较总工时与最晚完成目标。', '''
            machines = np.array([[1,3],[2,8]])
            mask = np.ones((2,2), bool)
            total, p = enumerate_assignment(machines, mask, "sum")
            finish, q = enumerate_assignment(machines, mask, "max")
            assert total == 5 and finish == 3 and p == q == (1,0)
            print("此实例两个目标恰好选中同一方案，并非普遍规律。")
        ''')], [('SciPy linear_sum_assignment', 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html')],
         '本例要求人数等于任务数、每人只做一项且同时开工。存在任务依赖或多人协作时，需要调度模型。')


def build_a04():
    make(4, '项目关键路径与资源', '项目最早何时结束，优先改善哪个环节？',
         '输入12项任务的耗时和前置依赖，先找资源充足时的关键路径，再用两名可执行任意任务的成员安排实际日程。', [
        first('## 1. 建立依赖图\n有向边u→v表示v必须等u完成。图必须无环，循环依赖会使这个项目无法按定义开始。任务耗时为整数天。', '''
            import networkx as nx
            from ortools.sat.python import cp_model
            lengths = dict(zip(list("ABCDEFGHIJKL"), [2,3,2,4,3,2,4,2,3,2,1,2]))
            edges = [("A","C"),("A","D"),("B","E"),("C","F"),("D","G"),("E","G"),
                     ("E","H"),("F","I"),("G","I"),("H","J"),("I","K"),("J","K"),("K","L")]
            graph = nx.DiGraph(); graph.add_nodes_from(lengths); graph.add_edges_from(edges)
            assert nx.is_directed_acyclic_graph(graph)
            pd.DataFrame({"task":list(lengths),"duration":list(lengths.values())}).to_csv(DATA_DIR / "任务.csv", index=False)
        '''),
        sec('## 2. 正向算最早时间，反向算最晚时间\n最早开始是所有前置任务完成时间的最大值。最晚开始倒推自项目终点；两者相减为松弛时间，0表示关键任务。', '''
            def critical_times(durations, precedences):
                g = nx.DiGraph(); g.add_nodes_from(durations); g.add_edges_from(precedences)
                if not nx.is_directed_acyclic_graph(g): raise ValueError("依赖存在环")
                order = list(nx.topological_sort(g)); early = {}; finish = {}
                for job in order:
                    early[job] = max((finish[p] for p in g.predecessors(job)), default=0)
                    finish[job] = early[job] + durations[job]
                makespan = max(finish.values(), default=0); late = {}
                for job in reversed(order):
                    latest_finish = min((late[s] for s in g.successors(job)), default=makespan)
                    late[job] = latest_finish - durations[job]
                return makespan, early, late
            lower_bound, early, late = critical_times(lengths, edges)
            critical = [j for j in lengths if late[j] == early[j]]
            assert all(early[v] >= early[u] + lengths[u] for u,v in edges)
            print("资源充足下最短工期", lower_bound, "关键任务", critical)
        '''),
        sec('## 3. 加入人员容量：资源竞争可能再推迟\n这是完成A10后可回看的求解器扩展。每项任务占一人且不可中断，`add_cumulative`约束任一时刻最多两项任务。独立检查按每天重新统计资源占用。', '''
            def resource_schedule(durations, precedences, capacity):
                model = cp_model.CpModel(); horizon = sum(durations.values())
                starts = {}; ends = {}; intervals = []
                for job, d in durations.items():
                    starts[job] = model.new_int_var(0, horizon, "start_"+job)
                    ends[job] = model.new_int_var(0, horizon, "end_"+job)
                    intervals.append(model.new_interval_var(starts[job], d, ends[job], job))
                for u,v in precedences: model.add(starts[v] >= ends[u])
                model.add_cumulative(intervals, [1]*len(intervals), capacity)
                end = model.new_int_var(0,horizon,"project_end")
                model.add_max_equality(end, list(ends.values())); model.minimize(end)
                solver = cp_model.CpSolver(); solver.parameters.max_time_in_seconds=5
                solver.parameters.num_search_workers=1
                status = solver.solve(model)
                result = ({j:int(solver.value(s)) for j,s in starts.items()}
                          if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None)
                return solver.status_name(status), result
            def check_schedule(starts, durations, precedences, capacity):
                if starts is None or set(starts) != set(durations): return False
                if any(s < 0 for s in starts.values()): return False
                if any(starts[v] < starts[u]+durations[u] for u,v in precedences): return False
                end = max(starts[j]+durations[j] for j in durations)
                return all(sum(starts[j] <= t < starts[j]+durations[j] for j in durations) <= capacity for t in range(end))
            status, actual = resource_schedule(lengths, edges, 2)
            assert actual is not None, f"未获得方案：{status}；UNKNOWN不等于INFEASIBLE。"
            assert check_schedule(actual, lengths, edges, 2)
            actual_end = max(actual[j]+lengths[j] for j in lengths)
            assert actual_end >= lower_bound
        '''),
        sec('## 4. 比较资源充足时缩短任务的影响\n关键路径可能不止一条；单独缩短一个关键任务不保证项目一定缩短。这里逐项重算资源充足的关键路径工期，任务最短仍为1天，因此已是1天的K不能再缩短；不能将该敏感性直接套到两人受限排程，后者需要逐项重新求解。', '''
            improvements = {}
            for job in lengths:
                changed = lengths.copy(); changed[job] = max(1, changed[job]-1)
                improvements[job] = lower_bound - critical_times(changed, edges)[0]
            rows = [{"task":j,"duration":lengths[j],"early":early[j],"slack":late[j]-early[j],
                     "resource_start":actual[j],"unlimited_one_day_improvement":improvements[j]} for j in lengths]
            pd.DataFrame(rows).to_csv(OUT_DIR / "项目时间表.csv", index=False, encoding="utf-8-sig")
            for k,j in enumerate(lengths): plt.barh(j,lengths[j],left=actual[j],color="#428c87")
            plt.xlabel("天"); plt.title("两名成员容量下的任务时间表")
            plt.tight_layout(); plt.savefig(OUT_DIR / "甘特图.png"); plt.close()
            save_results({"unlimited_resource_days":lower_bound,"two_worker_days":actual_end,
                          "status":status,"critical_tasks":critical,"unlimited_shorten_one_day_effect":improvements},
                         f"资源充足下工期为{lower_bound}天；两名成员下得到{actual_end}天计划，状态{status}。"
                         f"仅在资源充足模型下，逐项缩短1天（任务最短仍为1天）的工期影响为{improvements}。两人受限计划的缩短效果需要重新求解。")
        ''')], [
        exercise('补全：顺序任务的工期', 'X耗时2，Y耗时3，X是Y前置。检查最早时间和零松弛。', '''
            span, es, ls = critical_times({"X":2,"Y":3}, [("X","Y")])
            assert span == 5 and es == {"X":0,"Y":2} and es == ls
        '''),
        exercise('修改：人员增为3名', '重排并检查资源约束。比较求得的工期，只有状态OPTIMAL时才能作最优值比较。', '''
            state, plan = resource_schedule(lengths, edges, 3)
            assert check_schedule(plan, lengths, edges, 3)
            span = max(plan[j]+lengths[j] for j in lengths)
            assert span >= lower_bound
            if state == status == "OPTIMAL": assert span <= actual_end
            print(state, span)
        '''),
        exercise('迁移：检测循环依赖', '构造数据获取依赖训练、训练又依赖数据获取的错误计划，要求明确报错。', '''
            try:
                critical_times({"data":2,"train":3}, [("data","train"),("train","data")])
            except ValueError as exc:
                assert "环" in str(exc)
                print("已发现不可按序执行的依赖：", exc)
            else: raise AssertionError("应当拒绝有环任务图")
        ''')], [('NetworkX topological_sort', 'https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.dag.topological_sort.html'),
                ('OR-Tools Job Shop 官方实例', 'https://developers.google.com/optimization/scheduling/job_shop')],
         '关键路径的最早工期先假定资源充足且耗时确定。资源受限扩展使用不可中断任务、同质成员；真实技能差异需要额外约束。')


def build_a05():
    make(5, '共享实验室预约', '六个团队怎样预约两间实验室？',
         '团队各有可用时间、实验时长、房间要求和成员。输出一份不重叠、成员不分身的安排，并展示无解时该怎样描述。', [
        first('## 1. 先列出每个团队的合法选项\n一个预约用(room,start,end)表示，区间采用左闭右开：[1,3)与[3,4)不冲突。先过滤房间和可用时间，再解决团队间冲突。', '''
            from ortools.sat.python import cp_model
            jobs = [dict(name="光学组",duration=2,rooms=[0],starts=[0,1,2],members={"Li","Wu"}),
                    dict(name="编程组",duration=1,rooms=[0,1],starts=[0,1,2,3],members={"Sun"}),
                    dict(name="测试组",duration=2,rooms=[0,1],starts=[1,2,3,4],members={"Li"}),
                    dict(name="阅读组",duration=1,rooms=[1],starts=[0,1,2,3],members={"Qian"}),
                    dict(name="装配组",duration=2,rooms=[0,1],starts=[0,1,2,3,4],members={"Zhou"}),
                    dict(name="演示组",duration=1,rooms=[0],starts=[3,4,5],members={"Sun"})]
            def options(job):
                return [(room,start,start+job["duration"]) for room in job["rooms"] for start in job["starts"]]
            def conflict(a, b, job_a, job_b):
                overlap = max(a[1],b[1]) < min(a[2],b[2])
                shared_resource = a[0] == b[0] or bool(job_a["members"] & job_b["members"])
                return overlap and shared_resource
            assert not conflict((0,0,1),(0,1,2),jobs[0],jobs[1])
            (DATA_DIR / "预约需求.json").write_text(json.dumps(jobs,ensure_ascii=False,default=list,indent=2),encoding="utf-8")
        '''),
        sec('## 2. 回溯寻找第一份合法计划\n先安排选项少的团队。递归每增加一个预约就检查已安排团队，遇到冲突撤销选择。它有助于理解约束，不保证最小完工时间。', '''
            def backtrack_booking(requests):
                order = sorted(range(len(requests)),key=lambda i:len(options(requests[i])))
                plan = {}
                def visit(k):
                    if k == len(order): return dict(plan)
                    i = order[k]
                    for choice in options(requests[i]):
                        if all(not conflict(choice,other,requests[i],requests[j]) for j,other in plan.items()):
                            plan[i] = choice
                            found = visit(k+1)
                            if found is not None: return found
                            del plan[i]
                    return None
                return visit(0)
            baseline = backtrack_booking(jobs)
            assert baseline is not None
        '''),
        sec('## 3. 用CP-SAT选择互不冲突的选项\n每个团队恰好选一个选项。冲突的两个选项不能同时被选中。我们最小化最后结束时刻；求解状态和预约合法性分开检查。', '''
            def solve_booking(requests):
                model=cp_model.CpModel(); candidates=[options(j) for j in requests]; variables={}
                for i,choices in enumerate(candidates):
                    variables[i] = [model.new_bool_var(f"team{i}_option{k}") for k in range(len(choices))]
                    model.add_exactly_one(variables[i])
                for i in range(len(requests)):
                    for j in range(i+1,len(requests)):
                        for k,a in enumerate(candidates[i]):
                            for l,b in enumerate(candidates[j]):
                                if conflict(a,b,requests[i],requests[j]): model.add(variables[i][k]+variables[j][l] <= 1)
                horizon=max((c[2] for choices in candidates for c in choices),default=0)
                end=model.new_int_var(0,horizon,"end")
                for i,choices in enumerate(candidates):
                    for k,c in enumerate(choices): model.add(end>=c[2]).only_enforce_if(variables[i][k])
                model.minimize(end)
                solver=cp_model.CpSolver(); solver.parameters.max_time_in_seconds=5; solver.parameters.num_search_workers=1
                state=solver.solve(model)
                result=({i:next(c for k,c in enumerate(choices) if solver.value(variables[i][k])) for i,choices in enumerate(candidates)}
                        if state in (cp_model.OPTIMAL,cp_model.FEASIBLE) else None)
                return solver.status_name(state),result
            def check_booking(plan,requests):
                if plan is None or set(plan)!=set(range(len(requests))): return False
                # 直接核对原始需求，不复用求解器生成选项或冲突约束的函数。
                for i,(room,start,end) in plan.items():
                    request=requests[i]
                    if room not in request['rooms'] or start not in request['starts']: return False
                    if end-start!=request['duration']: return False
                for i in plan:
                    for j in plan:
                        if i>=j: continue
                        a,b=plan[i],plan[j]
                        if a[1]<b[2] and b[1]<a[2]:
                            if a[0]==b[0]: return False
                            if any(member in requests[j]['members'] for member in requests[i]['members']): return False
                return True
            status,plan=solve_booking(jobs)
            assert plan is not None, f"未获得预约表：{status}；UNKNOWN不等于无解。"
            assert check_booking(plan,jobs) and check_booking(baseline,jobs)
            corrupted=dict(plan); room,start,end=corrupted[0]; corrupted[0]=(room,start,end+1)
            assert not check_booking(corrupted,jobs)
            impossible=[dict(name=str(i),duration=1,rooms=[0],starts=[0],members={str(i)}) for i in range(2)]
            bad_status,bad_plan=solve_booking(impossible)
            assert bad_status=="INFEASIBLE" and bad_plan is None
        '''),
        sec('## 4. 保存预约表与冲突反例\n空结果不一定表示数学上无解：只有INFEASIBLE是证明无解。这个小例的冲突需求可通过增加房间或时间修复，但一般可能需要同时放宽多个条件。', '''
            table=pd.DataFrame([dict(team=jobs[i]["name"],room=c[0],start=c[1],end=c[2]) for i,c in plan.items()])
            table.to_csv(OUT_DIR / "预约表.csv",index=False,encoding="utf-8-sig")
            for i,c in plan.items():
                plt.barh(c[0],c[2]-c[1],left=c[1],height=.5,alpha=.7)
                plt.text((c[1]+c[2])/2,c[0],jobs[i]["name"],ha="center",va="center",fontsize=8)
            plt.yticks([0,1],["实验室0","实验室1"]); plt.xlabel("时段")
            plt.tight_layout(); plt.savefig(OUT_DIR / "预约安排.png"); plt.close()
            end=max(c[2] for c in plan.values()); baseline_end=max(c[2] for c in baseline.values())
            save_results({"status":status,"last_end":end,"backtracking_last_end":baseline_end,
                          "constraint_check":True,"infeasible_example":bad_status},
                         f"采用预约表中的方案，最后在时段{end}结束，回溯基线为{baseline_end}；状态{status}。"
                         "房间重叠和成员分身均独立检查通过。若可用时间变化，应先更新需求再求解。")
        ''')], [
        exercise('补全：同一成员不能同时出现', '房间不同但两组共享成员Li，检查两个重叠预约。', '''
            assert conflict((0,1,3),(1,2,4),jobs[0],jobs[2])
            assert not conflict((0,0,2),(1,2,4),jobs[0],jobs[2])
        '''),
        exercise('修改：修复无解需求', '给两个冲突团队都开放第二个时段，验证出现合法解。', '''
            repaired=[dict(j,starts=[0,1]) for j in impossible]
            state,p=solve_booking(repaired)
            assert check_booking(p,repaired) and state in ("OPTIMAL","FEASIBLE")
            print(p)
        '''),
        exercise('迁移：共享相机预约', '只有一台相机，两人各借2小时，可从0、2开始。直接用资源编号0表示相机。', '''
            camera=[dict(name=n,duration=2,rooms=[0],starts=[0,2],members={n}) for n in ["A","B"]]
            state,p=solve_booking(camera)
            assert check_booking(p,camera)
            assert sorted(c[1] for c in p.values())==[0,2]
            print("相机预约",p)
        ''')], [('OR-Tools 人员排班完整实例','https://developers.google.com/optimization/scheduling/employee_scheduling')],
         '本例时间离散、时长固定且实验不可中断；成员共享也属于资源互斥。逐条放宽仅是诊断方法，不保证找到最少的修改。')


def build_a06():
    make(6, '电量受限路径', '最快路线超出电量时，怎样重新规划？',
         '校园机器人只有6单位电量。每条路有时间和耗电两个属性。目标是满足电量限制时尽快到达，而不是把耗电任意加进时间。', [
        first('## 1. 一条边包含两个不同维度\n字典中每个地点映射到邻接道路列表。时间是目标，电量是硬限制，两者不混成一个没有单位解释的分数。', '''
            import heapq
            import networkx as nx
            road_data=[("S","A",1,5),("A","G",1,5),("S","B",2,2),("B","C",2,2),
                       ("C","G",2,2),("S","D",3,1),("D","G",4,3),("B","D",2,1)]
            roads={}
            for u,v,time,energy in road_data:
                roads.setdefault(u,[]).append((v,time,energy)); roads.setdefault(v,[]).append((u,time,energy))
            battery=6; start="S"; goal="G"
            graph=nx.Graph()
            for u,v,time,energy in road_data: graph.add_edge(u,v,time=time,energy=energy)
            fastest=nx.shortest_path(graph,start,goal,weight="time")
            pd.DataFrame(road_data,columns=["from","to","time","energy"]).to_csv(DATA_DIR / "道路.csv",index=False)
        '''),
        sec('## 2. 扩展状态：位置相同，剩余电量不同\n状态用元组(position,remaining)表示，可作字典键。优先队列按累计时间弹出；此处用启发式为0的Dijkstra确保容易检查。保留路径便于初学者观察。', '''
            def battery_route(network,origin,target,capacity):
                queue=[(0,origin,capacity,[origin])]; best={(origin,capacity):0}; expanded=0
                while queue:
                    elapsed,node,remaining,path=heapq.heappop(queue)
                    if elapsed != best.get((node,remaining)): continue
                    expanded+=1
                    if node==target: return path,elapsed,capacity-remaining,expanded
                    for nxt,duration,energy in network.get(node,[]):
                        if energy>remaining: continue
                        state=(nxt,remaining-energy); candidate=elapsed+duration
                        if candidate<best.get(state,float("inf")):
                            best[state]=candidate
                            heapq.heappush(queue,(candidate,nxt,remaining-energy,path+[nxt]))
                return None,None,None,expanded
            route,time,used,expanded=battery_route(roads,start,goal,battery)
            assert route is not None
        '''),
        sec('## 3. 用原始道路独立验证\n重新累计每条边的时间与电量；不相信搜索函数自行声称合法。小图再枚举所有简单路径，核对目标代价。正耗时和正耗电使绕圈不能改善本题目标。', '''
            def path_totals(path,network):
                if path is None: return None
                total_time=0;total_energy=0
                for u,v in zip(path,path[1:]):
                    matching=[edge for edge in network.get(u,[]) if edge[0]==v]
                    if not matching: return None
                    _,dt,de=matching[0];total_time+=dt;total_energy+=de
                return total_time,total_energy
            assert path_totals(route,roads)==(time,used) and used<=battery
            simple_paths=list(nx.all_simple_paths(graph,start,goal))
            feasible=[(path_totals(p,roads)[0],p) for p in simple_paths if path_totals(p,roads)[1]<=battery]
            assert time==min(t for t,p in feasible)
            assert battery_route(roads,start,goal,1)[0] is None
            fastest_time,fastest_energy=path_totals(fastest,roads)
        '''),
        sec('## 4. 区分不可行基线与有效方案\n无电量限制的最快路线只是说明冲突，不能当作可执行推荐。结果表显式标记可行性。', '''
            table=pd.DataFrame([dict(method="忽略电量",path="→".join(fastest),time=fastest_time,energy=fastest_energy,feasible=fastest_energy<=battery),
                                dict(method="电量约束",path="→".join(route),time=time,energy=used,feasible=True)])
            table.to_csv(OUT_DIR / "路线对照.csv",index=False,encoding="utf-8-sig")
            pos={"S":(0,0),"A":(1,1),"B":(1,0),"C":(2,0),"D":(1,-1),"G":(3,0)}
            nx.draw(graph,pos,with_labels=True,node_color="#a9d5dc",node_size=900)
            nx.draw_networkx_edges(graph,pos,edgelist=list(zip(route,route[1:])),edge_color="#df7843",width=3)
            plt.title("满足电量限制的路线");plt.tight_layout();plt.savefig(OUT_DIR / "路线.png");plt.close()
            save_results({"route":route,"time":time,"energy":used,"battery":battery,
                          "unconstrained_time":fastest_time,"unconstrained_energy":fastest_energy,"expanded_states":expanded},
                         f"推荐{'→'.join(route)}，时间{time}、电量{used}/{battery}。"
                         f"忽略电量的最快路线耗电{fastest_energy}，可行性为{fastest_energy<=battery}。电池容量或耗电估计改变时重算。")
        ''')], [
        exercise('补全：按原始边累计', '检查S-A-G的时间和电量，说明为什么不能推荐。', '''
            assert path_totals(["S","A","G"],roads)==(2,10)
            assert path_totals(["S","A","G"],roads)[1]>battery
        '''),
        exercise('修改：更换大电池', '电量增至10，求新路线并校验。', '''
            p,t,e,_=battery_route(roads,"S","G",10)
            assert t==2 and e<=10 and path_totals(p,roads)==(t,e)
            print(p,t,e)
        '''),
        exercise('迁移：步行路线受爬升限制', '道路时间保持不变，把energy解释为爬升单位，预算4；检查模型所需条件。', '''
            p,t,e,_=battery_route(roads,"S","G",4)
            assert p is not None and e<=4 and path_totals(p,roads)==(t,e)
            print("累计爬升预算模型",p,t,e,"；不能用净海拔变化替代累计爬升。")
        ''')], [('NetworkX A* 与路径代价接口','https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.astar.astar_path.html')],
         '电量离散且只减少，无充电、再生制动或不确定耗电。路径上的电量是硬约束；扩展状态数量随电量精度增加。')


def build_a07():
    make(7, '多列车无冲突规划', '三列小火车共用轨道，怎样全部到站？',
         '用5×5教学轨道网规划三列车。每步移动一格或等待，禁止同点碰撞和相向交换。比较按顺序规划与小规模CBS，并逐时刻检查整组轨迹。', [
        first('## 1. 定义地图、动作和碰撞\n坐标是(row,column)元组。车辆到达终点后持续占据终点，必须检查后来到达的车辆。这个约定属于模型，不应藏在动画里。', '''
            import heapq
            from collections import deque
            height=5;width=5;walls={(1,1),(1,3),(3,1),(3,3)}
            starts=[(2,0),(2,4),(0,2)]; goals=[(2,4),(2,0),(4,2)]
            def moves(pos):
                return [(pos[0]+dr,pos[1]+dc) for dr,dc in [(0,0),(1,0),(-1,0),(0,1),(0,-1)]
                        if 0<=pos[0]+dr<height and 0<=pos[1]+dc<width and (pos[0]+dr,pos[1]+dc) not in walls]
            def at(path,t): return path[min(t,len(path)-1)]
            def first_conflict(paths):
                end=max(map(len,paths))
                for t in range(end):
                    for i in range(len(paths)):
                        for j in range(i+1,len(paths)):
                            if at(paths[i],t)==at(paths[j],t): return ("v",i,j,t,at(paths[i],t))
                            if t>0 and at(paths[i],t-1)==at(paths[j],t) and at(paths[j],t-1)==at(paths[i],t):
                                return ("e",i,j,t,at(paths[i],t-1),at(paths[i],t))
                return None
            (DATA_DIR/"地图.json").write_text(json.dumps(dict(height=height,width=width,walls=list(walls),starts=starts,goals=goals)),encoding="utf-8")
        '''),
        sec('## 2. 时空A*：同一地点在不同时间是不同状态\n约束分两种：时刻t禁止占据某点，或禁止在t-1到t走某条边。到终点时还要检查未来约束。有限时间窗没有找到路径只能报告窗口内未找到。', '''
            def low_level(origin,target,constraints,horizon=30):
                vertices={(c[1],c[2]) for c in constraints if c[0]=="v"}
                edges={(c[1],c[2],c[3]) for c in constraints if c[0]=="e"}
                if (0,origin) in vertices: return None
                def h(p): return abs(p[0]-target[0])+abs(p[1]-target[1])
                count=itertools.count(); queue=[(h(origin),0,next(count),origin,[origin])];seen={(origin,0)}
                while queue:
                    _,t,_,pos,path=heapq.heappop(queue)
                    future_block=any(tt>=t and cell==target for tt,cell in vertices)
                    if pos==target and not future_block: return path
                    if t>=horizon: continue
                    for nxt in moves(pos):
                        tt=t+1
                        if (tt,nxt) in vertices or (tt,pos,nxt) in edges: continue
                        state=(nxt,tt)
                        if state in seen: continue
                        seen.add(state)
                        heapq.heappush(queue,(tt+h(nxt),tt,next(count),nxt,path+[nxt]))
                return None
            independent=[low_level(s,g,[]) for s,g in zip(starts,goals)]
            assert all(p is not None for p in independent)
            assert first_conflict(independent) is not None
        '''),
        sec('## 3. 先规划的车占用后续车辆的时空资源\n优先规划把已规划路径当成预约表。这通常较快，但车辆次序会影响结果，也可能让后车在当前时间窗内找不到路。', '''
            def prioritized(order,horizon=30):
                result=[None]*len(starts); fixed=[]
                for agent in order:
                    constraints=[]
                    for old in fixed:
                        for t in range(horizon+1):
                            constraints.append(("v",t,at(old,t)))
                            if t>0: constraints.append(("e",t,at(old,t),at(old,t-1)))
                    path=low_level(starts[agent],goals[agent],constraints,horizon)
                    if path is None: return None
                    result[agent]=path;fixed.append(path)
                return result
            priority_paths=prioritized([0,1,2])
            assert priority_paths is not None and first_conflict(priority_paths) is None
        '''),
        sec('## 4. CBS：发现冲突，再分支限制其中一辆车\n高层节点保存每辆车的限制与路径；每次按总路径长度取最便宜节点。冲突产生两个子节点，分别禁止其中一辆车造成该冲突；仅重算受影响车辆。资源上限返回UNKNOWN，不能当成无解。', '''
            def cbs(horizon=30,node_limit=2000):
                root_paths=[low_level(s,g,[],horizon) for s,g in zip(starts,goals)]
                if any(p is None for p in root_paths): return "UNKNOWN_HORIZON",None,0
                counter=itertools.count(); queue=[];visited=set(); expanded=0
                cost=lambda paths:sum(len(p)-1 for p in paths)
                heapq.heappush(queue,(cost(root_paths),next(counter),[[] for _ in starts],root_paths))
                while queue and expanded<node_limit:
                    _,_,restrictions,paths=heapq.heappop(queue); expanded+=1
                    issue=first_conflict(paths)
                    if issue is None: return "OPTIMAL_WITHIN_TIME_MODEL",paths,expanded
                    kind,i,j,t,*positions=issue
                    for agent in (i,j):
                        new=[list(c) for c in restrictions]
                        if kind=="v": added=("v",t,positions[0])
                        else:
                            u,v=positions
                            added=("e",t,u,v) if agent==i else ("e",t,v,u)
                        new[agent].append(added)
                        signature=tuple(tuple(sorted(cs)) for cs in new)
                        if signature in visited: continue
                        visited.add(signature)
                        updated=low_level(starts[agent],goals[agent],new[agent],horizon)
                        if updated is not None:
                            child=list(paths);child[agent]=updated
                            heapq.heappush(queue,(cost(child),next(counter),new,child))
                return "UNKNOWN_LIMIT_OR_HORIZON",None,expanded
            cbs_status,cbs_paths,cbs_expanded=cbs()
            assert cbs_paths is not None
            def validate_paths(paths):
                if paths is None or len(paths)!=len(starts): return False
                if any(not p or p[0]!=s or p[-1]!=g for p,s,g in zip(paths,starts,goals)): return False
                # 重新按业务规则回放，不调用搜索使用的 moves/first_conflict。
                for path in paths:
                    for row,col in path:
                        if not (0<=row<height and 0<=col<width) or (row,col) in walls: return False
                    for u,v in zip(path,path[1:]):
                        if abs(u[0]-v[0])+abs(u[1]-v[1])>1: return False
                previous=None
                for t in range(max(map(len,paths))):
                    current=[p[t] if t<len(p) else p[-1] for p in paths]
                    if len(set(current))!=len(current): return False
                    if previous is not None:
                        for i in range(len(paths)):
                            for j in range(i+1,len(paths)):
                                if current[i]==previous[j] and current[j]==previous[i]: return False
                    previous=current
                return True
            assert validate_paths(cbs_paths) and validate_paths(priority_paths)
            assert sum(len(p)-1 for p in cbs_paths)<=sum(len(p)-1 for p in priority_paths)
            assert cbs(node_limit=0)[0].startswith("UNKNOWN")
        '''),
        sec('## 5. 保存逐时刻轨迹并解释目标\n本例最小化各车到达时间之和，另报告全部到站时间。时间图的每一行能与碰撞检查器对应；不是仅展示单车路径。', '''
            rows=[]
            for method,paths in [("优先规划",priority_paths),("CBS",cbs_paths)]:
                for t in range(max(map(len,paths))):
                    for agent,p in enumerate(paths): rows.append(dict(method=method,time=t,train=agent,row=at(p,t)[0],col=at(p,t)[1]))
            pd.DataFrame(rows).to_csv(OUT_DIR/"列车时刻表.csv",index=False,encoding="utf-8-sig")
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            for ax,label,paths in zip(axes,["优先规划","CBS"],[priority_paths,cbs_paths]):
                for a,p in enumerate(paths): ax.plot([q[1] for q in p],[q[0] for q in p],"o-",label=f"车{a}",alpha=.7)
                for r,c in walls: ax.add_patch(plt.Rectangle((c-.45,r-.45),.9,.9,color="gray"))
                ax.set(xlim=(-.5,4.5),ylim=(4.5,-.5),title=label);ax.legend();ax.grid()
            plt.tight_layout();plt.savefig(OUT_DIR/"多车路线.png");plt.close()
            # 平面路线可能交叉；是否碰撞取决于时间。用时间表看清等待和到站占用。
            fig,ax=plt.subplots(figsize=(10,2.5));ax.axis("off")
            timeline=range(max(map(len,cbs_paths)))
            positions=[[f"({at(p,t)[0]},{at(p,t)[1]})" for t in timeline] for p in cbs_paths]
            table=ax.table(cellText=positions,rowLabels=[f"车{a}" for a in range(len(cbs_paths))],
                           colLabels=[f"t={t}" for t in timeline],loc="center",cellLoc="center")
            table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.7)
            ax.set_title("CBS逐时刻位置：到站后继续占用终点")
            plt.tight_layout();plt.savefig(OUT_DIR/"逐时刻位置.png");plt.close()
            cost=lambda ps:sum(len(p)-1 for p in ps)
            save_results({"cbs_status":cbs_status,"cbs_sum_cost":cost(cbs_paths),"priority_sum_cost":cost(priority_paths),
                          "cbs_makespan":max(len(p)-1 for p in cbs_paths),"cbs_high_level_nodes":cbs_expanded,"collision_free":True},
                         f"CBS方案总到达时间{cost(cbs_paths)}，优先方案{cost(priority_paths)}；"
                         f"CBS高层展开{cbs_expanded}个节点。逐步合法性、同点冲突、对向交换和终点占用均检查通过。"
                         "最优结论限定在本离散动作、总到达时间目标和30步时间窗模型内；实际列车的速度、制动和安全间隔未建模。")
        ''')], [
        exercise('补全：交换碰撞', '两辆车在一条边上对向交换位置，不能只检查同点碰撞。', '''
            swap=[[(0,0),(0,1)],[(0,1),(0,0)]]
            issue=first_conflict(swap)
            assert issue[0]=="e" and issue[3]==1
            print(issue)
        '''),
        exercise('修改：改变优先顺序', '检查所有6种优先次序；失败应记录，不能删除失败次序再报告全成功。', '''
            results=[]
            for order in itertools.permutations(range(3)):
                paths=prioritized(order)
                if paths is not None: assert validate_paths(paths)
                results.append((order,None if paths is None else sum(len(p)-1 for p in paths)))
            assert len(results)==6
            print(results)
        '''),
        exercise('迁移：仓库机器人到站后停留', '一车先到终点，另一车之后经过该点，应检测碰撞。再构造不冲突的等待方案。', '''
            bad=[[(0,0),(0,1)],[(1,2),(0,2),(0,1)]]
            assert first_conflict(bad)[0]=="v"
            good=[[(0,0),(0,1)],[(1,2),(1,1),(1,0)]]
            assert first_conflict(good) is None
            print("抵达后的占用必须继续纳入预约表。")
        ''')], [('Sharon等：Conflict-based search for optimal multi-agent pathfinding（原论文）','https://doi.org/10.1016/j.artint.2014.11.006'),
                ('CBS原研究会议公开记录','https://ojs.aaai.org/index.php/SOCS/article/view/18222')],
         '教学模型不是铁路安全控制系统。各边移动耗时为1，到站后永久占位；CBS的最优性依赖低层最短路、冲突划分和目标定义，有限窗口/节点预算未找到解只能报告UNKNOWN。')


def build_a08():
    make(8, '变化环境中的重规划', '道路封闭或拥堵后，怎样修订剩余计划？',
         '机器人在出发后才逐步获知封路和拥堵。比较坚持原计划、道路不可行时修订、每到节点重新计算三种规则；所有规则收到相同时间发生的信息。', [
        first('## 1. 建立道路和公开的事件机制\n图的weight表示当前已知通行时间。事件按模拟时间发生，策略只能在到达节点时观察；不得提前把未来封路泄露给规划器。', '''
            import networkx as nx
            initial=nx.Graph()
            initial.add_weighted_edges_from([("S","A",1),("A","G",1),("A","B",1),("B","G",3),("B","C",1),("C","G",1),("S","B",3)])
            events=[dict(time=1,kind="close",u="A",v="G"),dict(time=2,kind="cost",u="B",v="C",value=9)]
            def observe(graph,clock,applied):
                messages=[]
                for k,event in enumerate(events):
                    if k in applied or event["time"]>clock: continue
                    if event["kind"]=="close": graph.remove_edge(event["u"],event["v"])
                    else: graph[event["u"]][event["v"]]["weight"]=event["value"]
                    applied.add(k);messages.append(event["kind"]+":"+event["u"]+"-"+event["v"])
                return messages
            old_path=nx.shortest_path(initial,"S","G",weight="weight")
            (DATA_DIR/"事件.json").write_text(json.dumps(events,indent=2),encoding="utf-8")
        '''),
        sec('## 2. 三种反馈规则共用一个执行环境\n每轮先观测，再决定是否重算，再执行下一段。已经走过的历史固定。修改量统计为未来路线发生变化的次数，并不把一次函数调用当成真正修改。', '''
            def execute(policy):
                graph=initial.copy();applied=set();clock=0;node="S";planned=list(old_path);trace=[];changes=0
                for step in range(20):
                    news=observe(graph,clock,applied)
                    if node=="G": return dict(policy=policy,success=True,time=clock,changes=changes,trace=trace)
                    legal=all(graph.has_edge(u,v) for u,v in zip(planned,planned[1:]))
                    should_replan=policy=="rolling" or (policy=="repair_blocked" and not legal)
                    if should_replan:
                        try: replacement=nx.shortest_path(graph,node,"G",weight="weight")
                        except nx.NetworkXNoPath: return dict(policy=policy,success=False,time=clock,changes=changes,trace=trace,reason="NO_PATH_CURRENT_MAP")
                        changes+=int(replacement!=planned);planned=replacement
                    if len(planned)<2 or not graph.has_edge(node,planned[1]):
                        return dict(policy=policy,success=False,time=clock,changes=changes,trace=trace,reason="OLD_PLAN_INFEASIBLE")
                    nxt=planned[1];duration=graph[node][nxt]["weight"]
                    trace.append(dict(step=step,start=clock,origin=node,target=nxt,duration=duration,news=";".join(news)))
                    clock+=duration;node=nxt;planned=planned[1:]
                return dict(policy=policy,success=False,time=clock,changes=changes,trace=trace,reason="EXECUTION_LIMIT")
            results=[execute(p) for p in ["fixed","repair_blocked","rolling"]]
            assert not results[0]["success"]
            assert results[1]["success"] and results[2]["success"]
        '''),
        sec('## 3. 按实际观测时刻复核执行轨迹\n检查器重新播放事件，验证每条路当时确实可通行且使用当时的耗时。它不从策略的文字解释判断对错。', '''
            def verify_execution(result):
                graph=initial.copy();applied=set();clock=0;location="S"
                for row in result["trace"]:
                    observe(graph,clock,applied)
                    if row["origin"]!=location or row["start"]!=clock: return False
                    if not graph.has_edge(location,row["target"]): return False
                    if row["duration"]!=graph[location][row["target"]]["weight"]: return False
                    clock+=row["duration"];location=row["target"]
                return clock==result["time"] and (not result["success"] or location=="G")
            assert all(verify_execution(r) for r in results)
            assert results[2]["time"]<results[1]["time"]
            # 原计划失败耗时不是可执行方案的总成本，图中标成未完成。
            summary=pd.DataFrame([{k:v for k,v in r.items() if k!="trace"} for r in results])
        '''),
        sec('## 4. 给出触发规则和局限\n周期重算多花计算，不应只看路线更短。这里图很小，完整实验仍记录计算次数/修改次数；更大场景要加入实际求解时间和改计划的组织成本。', '''
            summary.to_csv(OUT_DIR/"策略对照.csv",index=False,encoding="utf-8-sig")
            pd.DataFrame([dict(policy=r["policy"],**row) for r in results for row in r["trace"]]).to_csv(OUT_DIR/"执行记录.csv",index=False,encoding="utf-8-sig")
            successful=[r for r in results if r["success"]]
            plt.bar([r["policy"] for r in successful],[r["time"] for r in successful]);plt.ylabel("完成任务的实际耗时")
            plt.title("固定旧计划未完成，不作为有效低成本方案")
            plt.tight_layout();plt.savefig(OUT_DIR/"重规划.png");plt.close()
            best=min(successful,key=lambda r:r["time"])
            save_results({"strategies":[{k:v for k,v in r.items() if k!="trace"} for r in results]},
                         f"本情景推荐{best['policy']}，实际耗时{best['time']}、修改{best['changes']}次。"
                         "仅在封路时修复会忽视仍可通行但变慢的道路；可将成本明显变化也设为事件触发。"
                         "本例是图上反馈重规划，没有声称具备连续控制MPC的稳定性保证。")
        ''')], [
        exercise('补全：辨认不可行的旧计划', '回放时刻1事件，检查原路线哪条边已经失效。', '''
            g=initial.copy();seen=set();observe(g,1,seen)
            invalid=[(u,v) for u,v in zip(old_path,old_path[1:]) if not g.has_edge(u,v)]
            assert invalid==[("A","G")]
            print(invalid)
        '''),
        exercise('修改：没有任何环境事件', '暂时设事件为空，重新比较三种策略。完成后恢复事件，避免后续单元格使用错误环境。', '''
            saved_events=events
            try:
                events=[]
                unchanged=[execute(p) for p in ["fixed","repair_blocked","rolling"]]
                assert all(r["success"] and r["time"]==2 for r in unchanged)
                assert all(r["changes"]==0 for r in unchanged)
            finally: events=saved_events
            print("稳定环境下，本例反复重算没有路线收益。")
        '''),
        exercise('迁移：物流执行审计', '复制滚动策略轨迹，篡改一个道路耗时，验证审计器能发现不一致。', '''
            import copy
            corrupted=copy.deepcopy(results[2])
            corrupted["trace"][0]["duration"]+=3
            assert not verify_execution(corrupted)
            assert verify_execution(results[2])
            print("执行日志必须与当时可观测数据一致。")
        ''')], [('NetworkX shortest_path 官方接口','https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.generic.shortest_path.html'),
                ('do-mpc 模型预测控制与反馈说明','https://www.do-mpc.com/en/latest/theory_mpc.html')],
         '此教学环境假设到达节点时观测，通行中耗时不再变化。周期重算与事件触发哪种更好取决于信息更新、计算成本和变更成本。')


PDDL_DOMAIN = '''(define (domain delivery)
 (:requirements :strips)
 (:predicates (at-hall) (at-room) (key-hall) (has-key) (door-open)
              (package-hall) (carrying) (delivered))
 (:action take-key :parameters () :precondition (and (at-hall) (key-hall))
  :effect (and (has-key) (not (key-hall))))
 (:action open-door :parameters () :precondition (and (at-hall) (has-key))
  :effect (door-open))
 (:action take-package :parameters () :precondition (and (at-hall) (package-hall))
  :effect (and (carrying) (not (package-hall))))
 (:action enter-room :parameters () :precondition (and (at-hall) (door-open))
  :effect (and (at-room) (not (at-hall))))
 (:action leave-room :parameters () :precondition (and (at-room) (door-open))
  :effect (and (at-hall) (not (at-room))))
 (:action deliver :parameters () :precondition (and (at-room) (carrying))
  :effect (and (delivered) (not (carrying)))))
'''
PDDL_PROBLEM = '''(define (problem deliver-one)
 (:domain delivery)
 (:init (at-hall) (key-hall) (package-hall))
 (:goal (and (delivered))))
'''


def build_a09():
    make(9, '动作推理与模型校验', '怎样自动决定拿钥匙、开门与送货的顺序？',
         '初始在走廊，钥匙和包裹都在走廊，房门关闭。你要让程序从目标反推出可执行动作序列，并识别一份漏写“开门需要钥匙”的离线模型草稿。', [
        first('## 1. 事实集合与动作三元组\n状态是当前为真的事实集合。动作包含pre前提、add新增事实、delete删除事实。集合差与并集表达状态更新。未出现的事实按本模型视为假。', '''
            from collections import deque
            def action(name,pre,add,delete):
                return dict(name=name,pre=set(pre),add=set(add),delete=set(delete))
            actions=[action("take-key",["at-hall","key-hall"],["has-key"],["key-hall"]),
                     action("open-door",["at-hall","has-key"],["door-open"],[]),
                     action("take-package",["at-hall","package-hall"],["carrying"],["package-hall"]),
                     action("enter-room",["at-hall","door-open"],["at-room"],["at-hall"]),
                     action("leave-room",["at-room","door-open"],["at-hall"],["at-room"]),
                     action("deliver",["at-room","carrying"],["delivered"],["carrying"])]
            initial_state=frozenset(["at-hall","key-hall","package-hall"]); target={"delivered"}
            def apply_action(state,rule):
                if not rule["pre"]<=set(state): raise ValueError("前提未满足："+rule["name"])
                return frozenset((set(state)-rule["delete"])|rule["add"])
            print("初始事实",sorted(initial_state))
        '''),
        sec('## 2. 前向广度优先搜索\n队列保存状态与已走动作。每个动作单位代价时BFS找到最少动作数；如果不同动作成本不同，需要换成带代价搜索。frozenset可以放入visited避免循环。', '''
            def plan_actions(start,goal,rules):
                queue=deque([(frozenset(start),[])]);visited={frozenset(start)}
                while queue:
                    state,plan=queue.popleft()
                    if set(goal)<=set(state): return plan,len(visited)
                    for rule in rules:
                        if rule["pre"]<=set(state):
                            nxt=apply_action(state,rule)
                            if nxt not in visited:
                                visited.add(nxt);queue.append((nxt,plan+[rule["name"]]))
                return None,len(visited)
            plan,visited_count=plan_actions(initial_state,target,actions)
            assert plan is not None and len(plan)==5
            impossible,_=plan_actions(initial_state,{"teleported"},actions)
            assert impossible is None
            print("动作计划",plan)
        '''),
        sec('## 3. 按原任务写独立执行器\n验证器不读取动作表中的pre/add/delete，而按房间、钥匙、门和包裹的业务规则执行。这能抓住动作模型遗漏前提的错误。验证器本身也用正例和反例检查。', '''
            def validate_business_plan(sequence):
                location="hall";key=False;door=False;carrying=False;delivered=False;key_on_floor=True;package_on_floor=True
                trace=[]
                for step,name in enumerate(sequence):
                    ok=True
                    if name=="take-key":
                        ok=location=="hall" and key_on_floor
                        if ok:key=True;key_on_floor=False
                    elif name=="open-door":
                        ok=location=="hall" and key
                        if ok:door=True
                    elif name=="take-package":
                        ok=location=="hall" and package_on_floor
                        if ok:carrying=True;package_on_floor=False
                    elif name=="enter-room":
                        ok=location=="hall" and door
                        if ok:location="room"
                    elif name=="leave-room":
                        ok=location=="room" and door
                        if ok:location="hall"
                    elif name=="deliver":
                        ok=location=="room" and carrying
                        if ok:delivered=True;carrying=False
                    else:ok=False
                    trace.append(dict(step=step,action=name,valid=ok,location=location,key=key,door=door,carrying=carrying,delivered=delivered))
                    if not ok:return False,trace
                return delivered,trace
            valid,trace=validate_business_plan(plan)
            assert valid
            assert not validate_business_plan(["open-door","enter-room","deliver"])[0]
            assert not validate_business_plan(["invent-action"])[0]
        '''),
        sec('## 4. 对照PDDL，检查离线错误草稿\n同目录domain.pddl与problem.pddl表达相同有限动作问题。本例没有安装外部PDDL求解器，不能声称已通过它；实际执行的是上面的STRIPS搜索。拓展草稿是人为构造的错误样例，不冒充模型实时输出。', '''
            import copy
            draft=copy.deepcopy(actions)
            draft[1]["pre"].remove("has-key")
            draft_plan,_=plan_actions(initial_state,target,draft)
            draft_ok,draft_trace=validate_business_plan(draft_plan)
            assert not draft_ok and len(draft_plan)==4
            (OUT_DIR/"离线错误草稿.json").write_text(json.dumps(draft,ensure_ascii=False,default=sorted,indent=2),encoding="utf-8")
            (OUT_DIR/"计划.txt").write_text("\\n".join(plan),encoding="utf-8")
            pd.DataFrame(trace).to_csv(OUT_DIR/"状态追踪.csv",index=False,encoding="utf-8-sig")
            pd.DataFrame(draft_trace).to_csv(OUT_DIR/"错误草稿检查.csv",index=False,encoding="utf-8-sig")
            plt.barh(range(len(plan)),[1]*len(plan),left=range(len(plan)),color="#498a9c")
            plt.yticks(range(len(plan)),plan);plt.xlabel("动作步")
            plt.tight_layout();plt.savefig(OUT_DIR/"动作序列.png");plt.close()
            save_results({"plan":plan,"action_count":len(plan),"visited_states":visited_count,
                          "business_validator_pass":valid,"draft_plan":draft_plan,"draft_validator_pass":draft_ok,
                          "external_pddl_solver_run":False},
                         f"采用{len(plan)}步计划：{' → '.join(plan)}。独立业务执行检查通过。"
                         f"漏写钥匙前提的草稿产生{len(draft_plan)}步计划，但独立检查拒绝；求解器只能解已写出的模型，不能自动补齐现实需求。")
        ''')], [
        exercise('补全：动作前提', '初始状态不能执行开门；先拿钥匙后可以。', '''
            try:apply_action(initial_state,actions[1])
            except ValueError:pass
            else:raise AssertionError("无钥匙不应开门")
            s=apply_action(initial_state,actions[0]);s=apply_action(s,actions[1])
            assert "door-open" in s
        '''),
        exercise('修改：包裹不存在', '从初始状态移除package-hall，原动作模型应报告目标不可达。', '''
            changed=set(initial_state)-{"package-hall"}
            p,_=plan_actions(changed,target,actions)
            assert p is None
            print("当前模型内完整状态搜索后未发现可达目标。")
        '''),
        exercise('迁移：实验前置步骤', '使用同一搜索器表达“未准备→已校准→已实验→报告完成”，并检查缺少校准动作时不能完成。', '''
            lab=[action("calibrate",["ready"],["calibrated"],[]),
                 action("experiment",["calibrated"],["measured"],[]),
                 action("report",["measured"],["reported"],[])]
            p,_=plan_actions({"ready"},{"reported"},lab)
            assert p==["calibrate","experiment","report"]
            assert plan_actions({"ready"},{"reported"},lab[1:])[0] is None
            print(p)
        ''')], [('Unified Planning 基础教程','https://unified-planning.readthedocs.io/en/latest/notebooks/01-basic-example.html'),
                ('LLMFP 形式化规划与失败分析','https://arxiv.org/html/2410.12112'),
                ('LLM-Modulo 外部检查器框架','https://arxiv.org/html/2411.14484v1')],
         '世界完全可观测、动作确定、单位代价且没有并行动作。PDDL文本提供阅读对照，未运行外部PDDL工具；离线错误草稿不是实时语言模型实验。',
         extra_files={'domain.pddl':PDDL_DOMAIN,'problem.pddl':PDDL_PROBLEM})


def build_a10():
    make(10, '志愿者排班', '八名志愿者怎样兼顾覆盖、技能和公平？',
         '四个服务时段，每段需要3人且至少1名急救员；每人最多两段、不能连续三段。先满足硬要求，再减少人数负担差并兼顾偏好。', [
        first('## 1. 人员时间表与硬条件\navailability[w,s]说明能否上班，preferences是0/1偏好，不等于可用性。急救员覆盖是硬限制，不能被高偏好分抵消。', '''
            from ortools.sat.python import cp_model
            availability=np.array([[1,1,1,1],[1,1,0,1],[1,0,1,1],[1,1,1,0],
                                   [0,1,1,1],[1,1,0,1],[1,0,1,1],[0,1,1,1]],int)
            preferences=np.array([[1,0,1,0],[1,1,0,0],[0,0,1,1],[1,0,1,0],
                                  [0,1,0,1],[1,0,0,1],[0,0,1,1],[0,1,1,0]],int)
            skilled={0,1,2};demand=[3,3,3,3]
            pd.DataFrame(availability,columns=[f"时段{s}" for s in range(4)]).to_csv(DATA_DIR/"可用时间.csv",encoding="utf-8-sig")
        '''),
        sec('## 2. 用二元变量表达排班\n每个x[w,s]是是否上岗。先用约束表达不能违背的事，再定义负担差。优先级权重设为人员数×时段数+1，大于偏好收益的可能变化，因此先压低负担差，再改善偏好。', '''
            def schedule(available,preferred,qualified,needed):
                workers,slots=available.shape;model=cp_model.CpModel()
                x={(w,s):model.new_bool_var(f"w{w}s{s}") for w in range(workers) for s in range(slots)}
                for w in range(workers):
                    for s in range(slots):model.add(x[w,s]<=int(available[w,s]))
                    model.add(sum(x[w,s] for s in range(slots))<=2)
                    for s in range(slots-2):model.add(sum(x[w,k] for k in range(s,s+3))<=2)
                for s in range(slots):
                    model.add(sum(x[w,s] for w in range(workers))==int(needed[s]))
                    model.add(sum(x[w,s] for w in qualified)>=1)
                loads=[model.new_int_var(0,2,f"load{w}") for w in range(workers)]
                for w in range(workers):model.add(loads[w]==sum(x[w,s] for s in range(slots)))
                maximum=model.new_int_var(0,2,"maximum");minimum=model.new_int_var(0,2,"minimum")
                model.add_max_equality(maximum,loads);model.add_min_equality(minimum,loads)
                pref=sum(int(preferred[w,s])*x[w,s] for w in range(workers) for s in range(slots))
                priority_weight=workers*slots+1
                model.minimize(priority_weight*(maximum-minimum)-pref)
                solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=8;solver.parameters.num_search_workers=1
                state=solver.solve(model)
                plan=(np.array([[solver.value(x[w,s]) for s in range(slots)] for w in range(workers)],int)
                      if state in (cp_model.OPTIMAL,cp_model.FEASIBLE) else None)
                return solver.status_name(state),plan
            status,roster=schedule(availability,preferences,skilled,demand)
            assert roster is not None, f"未获得排班：{status}；UNKNOWN不能解释为已证明无解。"
        '''),
        sec('## 3. 独立检查原始业务约束\n向量化表达简洁，但先解释axis：沿axis=0求和得到每个时段人数，axis=1得到每人总班数。不要把缺人案例的UNKNOWN误称为已证明无解。', '''
            def check_roster(roster,available,qualified,needed):
                if roster is None or roster.shape!=available.shape:return False
                if not np.isin(roster,[0,1]).all() or np.any(roster>available):return False
                if not np.array_equal(roster.sum(axis=0),np.array(needed)):return False
                if np.any(roster.sum(axis=1)>2):return False
                if np.any(roster[list(qualified)].sum(axis=0)<1):return False
                return all(roster[w,s:s+3].sum()<=2 for w in range(len(roster)) for s in range(roster.shape[1]-2))
            assert check_roster(roster,availability,skilled,demand)
            no_skill=availability.copy();no_skill[list(skilled),0]=0
            bad_status,bad_roster=schedule(no_skill,preferences,skilled,demand)
            assert bad_status=="INFEASIBLE" and bad_roster is None
            broken=roster.copy();broken[:,0]=0
            assert not check_roster(broken,availability,skilled,demand)
        '''),
        sec('## 4. 保存排班并说明公平定义\n此处公平只指总班数差，不包含岗位强度、个人需求和历史负担。若想改变公平原则，应修改模型而非事后给结果贴公平标签。', '''
            table=pd.DataFrame(roster,index=[f"志愿者{w}" for w in range(len(roster))],columns=[f"时段{s}" for s in range(roster.shape[1])])
            table["总班数"]=roster.sum(axis=1);table["急救资格"]=[w in skilled for w in range(len(roster))]
            table.to_csv(OUT_DIR/"排班表.csv",encoding="utf-8-sig")
            plt.imshow(roster,cmap="Blues",vmin=0,vmax=1,aspect="auto");plt.yticks(range(8),table.index);plt.xticks(range(4),[f"时段{s}" for s in range(4)])
            plt.title("深色表示上岗");plt.tight_layout();plt.savefig(OUT_DIR/"排班.png");plt.close()
            spread=int(np.ptp(roster.sum(axis=1)));preferred_count=int((roster*preferences).sum())
            save_results({"status":status,"workload_spread":spread,"preferred_assignments":preferred_count,
                          "coverage":roster.sum(axis=0).tolist(),"all_constraints_pass":True,"infeasible_test":bad_status},
                         f"推荐排班表，4段覆盖{roster.sum(axis=0).tolist()}，总班数差{spread}，满足{preferred_count}个偏好。"
                         f"状态{status}；如果急救员在首段都不可用，需求已被证明无解，需要改需求或增加合格人员。")
        ''')], [
        exercise('补全：覆盖人数', '分别计算每段覆盖和每人班数，用断言核对需求。', '''
            coverage=roster.sum(axis=0);loads=roster.sum(axis=1)
            assert coverage.tolist()==demand and loads.max()<=2
            print(coverage,loads)
        '''),
        exercise('修改：偏好全部相同', '去掉偏好差异，只保留覆盖与负担差目标。不要要求排班表逐格保持相同。', '''
            state,p=schedule(availability,np.zeros_like(preferences),skilled,demand)
            assert check_roster(p,availability,skilled,demand)
            assert np.ptp(p.sum(axis=1))==1
            print(state,p)
        '''),
        exercise('迁移：六人值守三段', '每段2人、前两人合格，所有人所有时段可用。能否让所有人恰好一段，同时每段都有合格人员？用方案检查你的判断。', '''
            avail=np.ones((6,3),int)
            state,p=schedule(avail,np.zeros_like(avail),{0,1},[2,2,2])
            assert check_roster(p,avail,{0,1},[2,2,2])
            # 三段都需合格者，只有两名合格者，至少一人必须值守两段。
            assert p[:2].sum()>=3
            print("不能同时要求六人各一段并满足三段技能覆盖。",state,p.sum(axis=1))
        ''')], [('OR-Tools Employee Scheduling','https://developers.google.com/optimization/scheduling/employee_scheduling'),
                ('CP-SAT 返回状态','https://developers.google.com/optimization/cp/cp_solver')],
         '本例时段等长，每人最多两段。公平定义只是总班数差；多目标优先级权重应大于低优先级目标可能变化范围。')


def build_a21():
    make(21, '时段竞争与纳什均衡', '两个团队为什么都挤在黄金时段？',
         '两个团队各选黄金或普通时段。评分体现曝光与拥挤的假设取舍。找出最佳回应、纳什均衡与总收益最优组合，并检验“均衡就是最好”是否成立。', [
        first('## 1. 收益矩阵必须解释每个格子\nA[i,j]是行团队选i、列团队选j时行团队的收益；B[i,j]是同一局面的列团队收益。分数为教学假设，不能视为真实团队偏好的测量。', '''
            import nashpy as nash
            labels=["黄金","普通"]
            A=np.array([[2.,4.],[1.,3.]])
            B=np.array([[2.,1.],[4.,3.]])
            records=[dict(team1=labels[i],team2=labels[j],payoff1=A[i,j],payoff2=B[i,j]) for i in range(2) for j in range(2)]
            pd.DataFrame(records).to_csv(DATA_DIR/"收益矩阵.csv",index=False,encoding="utf-8-sig")
            def pure_equilibria(pay_a,pay_b):
                return [(i,j) for i in range(pay_a.shape[0]) for j in range(pay_a.shape[1])
                        if pay_a[i,j]>=pay_a[:,j].max()-1e-9 and pay_b[i,j]>=pay_b[i,:].max()-1e-9]
        '''),
        sec('## 2. 单方改变能否获利？\n纳什均衡只要求在别人不变时，任何一方都没有更好的单方选择。不要求社会总收益最大。逐格计算最佳回应比背术语更容易核对。', '''
            pure=pure_equilibria(A,B)
            assert pure==[(0,0)]
            total=A+B;social=tuple(int(x) for x in np.unravel_index(np.argmax(total),total.shape))
            assert social==(1,1)
            best_responses={"team1_to_gold":int(np.argmax(A[:,0])),"team1_to_normal":int(np.argmax(A[:,1])),
                            "team2_to_gold":int(np.argmax(B[0,:])),"team2_to_normal":int(np.argmax(B[1,:]))}
            print("均衡",pure,"总收益更高的组合",social,"最佳回应",best_responses)
        '''),
        sec('## 3. 用Nashpy求混合策略并独立检查\n概率向量x、y分别描述两队选择。`x @ A @ y`是期望收益；独立检查所有纯动作的期望收益不高于当前混合收益，从而检查没有单方获利偏离。', '''
            def check_equilibrium(pay_a,pay_b,x,y,tol=1e-7):
                x=np.asarray(x);y=np.asarray(y)
                if np.any(x < -tol) or np.any(y < -tol) or not np.isclose(x.sum(),1) or not np.isclose(y.sum(),1):return False
                u=float(x@pay_a@y);v=float(x@pay_b@y)
                return np.max(pay_a@y)<=u+tol and np.max(x@pay_b)<=v+tol
            equilibria=list(nash.Game(A,B).support_enumeration())
            assert equilibria and all(check_equilibrium(A,B,x,y) for x,y in equilibria)
            assert not check_equilibrium(A,B,np.array([0.,1.]),np.array([0.,1.]))
            # 拥挤收费是规则改变，支付是否算社会损失需单独定义。
            fee=1.5; adjusted_a=A.copy();adjusted_b=B.copy()
            adjusted_a[0,:]-=fee;adjusted_b[:,0]-=fee
            after=pure_equilibria(adjusted_a,adjusted_b)
            assert after==[(1,1)]
        '''),
        sec('## 4. 推荐协调规则而不冒充行为预测\n矩阵假设给定后，均衡分析可以解释激励。若加入黄金时段费用会改变均衡，应说明费用由谁收取、能否转移，而不能直接说总收益凭空增加。', '''
            plt.imshow(total,cmap="YlGnBu");plt.xticks(range(2),labels);plt.yticks(range(2),labels)
            for i in range(2):
                for j in range(2):plt.text(j,i,f"({A[i,j]:g},{B[i,j]:g})",ha="center",va="center")
            plt.xlabel("团队2");plt.ylabel("团队1");plt.title("收益矩阵；底色为两队收益之和")
            plt.tight_layout();plt.savefig(OUT_DIR/"收益与均衡.png");plt.close()
            save_results({"pure_equilibria":pure,"social_optimum":social,"equilibrium_total":float(total[pure[0]]),
                          "social_total":float(total[social]),"fee":fee,"equilibria_after_fee":after},
                         f"当前唯一纯策略均衡为双方选择黄金，总收益{total[0,0]:g}；双方普通的总收益为{total[1,1]:g}。"
                         f"在本收益假设下，每方黄金费用{fee}使普通组合成为均衡；这只是规则敏感性实验，需要确认真实偏好与费用归属。")
        ''')], [
        exercise('补全：均衡检查', '验证黄金组合满足均衡，普通组合不满足；写出哪方可以单独改善。', '''
            assert check_equilibrium(A,B,[1,0],[1,0])
            assert not check_equilibrium(A,B,[0,1],[0,1])
            assert A[0,1]>A[1,1] and B[1,0]>B[1,1]
        '''),
        exercise('修改：减小黄金费用', '费用只收0.5，重新计算纯均衡，解释为何规则改变幅度不足。', '''
            aa=A.copy();bb=B.copy();aa[0,:]-=.5;bb[:,0]-=.5
            assert pure_equilibria(aa,bb)==[(0,0)]
            print("费用0.5尚未抵消当前单方使用黄金时段的收益差。")
        '''),
        exercise('迁移：匹配硬币博弈', '矩阵[[1,-1],[-1,1]]与其相反矩阵，没有纯均衡；验证均匀随机策略。', '''
            coin=np.array([[1.,-1.],[-1.,1.]])
            assert pure_equilibria(coin,-coin)==[]
            assert check_equilibrium(coin,-coin,[.5,.5],[.5,.5])
            es=list(nash.Game(coin,-coin).support_enumeration())
            assert any(check_equilibrium(coin,-coin,x,y) for x,y in es)
            print(es)
        ''')], [('Nashpy 教程：Two player games','https://nashpy.readthedocs.io/en/stable/tutorial/index.html'),
                ('Nashpy Support enumeration','https://nashpy.readthedocs.io/en/stable/how-to/solve-with-support-enumeration.html')],
         '完全信息、同时选择、收益可量化是本例假设。均衡不是必然发生的心理预测；总收益更高也不自动代表公平。')


def build_a22():
    make(22, '重复合作与公共人手', '共享人手怎样避免总有人等待别人贡献？',
         '四个团队连续20轮决定是否贡献一人。贡献有个体成本，公共收益由四队共享；比较固定贡献、搭便车、条件合作及事先约定的轮值。', [
        first('## 1. 先写清公共品规则\n每队每轮有1单位资源，贡献c_i∈{0,1}后保留1-c_i；公共贡献乘1.6后均分。因此集体贡献有益，但个人在其他人不变时少贡献能获得更多。', '''
            n=4;rounds=20;multiplier=1.6;required=2
            def payoffs(contributions,m=multiplier):
                c=np.asarray(contributions,float)
                assert np.all((c==0)|(c==1))
                return 1-c+m*c.sum()/len(c)
            assert np.allclose(payoffs([1,1,1,1]),1.6)
            assert payoffs([0,1,1,1])[0]>payoffs([1,1,1,1])[0]
            rng=np.random.default_rng(22)
            # 相同可用性情景供所有规则比较；5%概率该团队本轮无法兑现投入。
            availability=rng.random((40,rounds,n))>=.05
            rules={"固定贡献":["cooperate"]*4,"各自不贡献":["defect"]*4,
                   "条件合作":["conditional"]*4,"混入搭便车":["defect"]+["conditional"]*3,
                   "约定轮值":["rotation"]*4}
        '''),
        sec('## 2. 明确策略如何读取历史\n条件合作只看上一轮实际贡献率；第一轮合作。轮值每轮安排两队，因此是提前协调的规则，不是个体自发选择的纳什均衡。', '''
            def simulate(kinds,available):
                previous=np.ones(len(kinds),int);history=[]
                for t in range(len(available)):
                    intended=[]
                    for i,kind in enumerate(kinds):
                        if kind=="cooperate":choice=1
                        elif kind=="defect":choice=0
                        elif kind=="conditional":choice=int(previous.mean()>=.75)
                        elif kind=="rotation":choice=int(i in {t%len(kinds),(t+1)%len(kinds)})
                        else:raise ValueError(kind)
                        intended.append(choice)
                    actual=np.array(intended,int)*available[t].astype(int)
                    rewards=payoffs(actual)
                    history.append(dict(round=t,contributions=actual.tolist(),payoffs=rewards.tolist(),
                                        contributed=int(actual.sum()),shortage=max(0,required-int(actual.sum()))))
                    previous=actual
                return history
            experiments={label:[simulate(kinds,a) for a in availability] for label,kinds in rules.items()}
        '''),
        sec('## 3. 用会计恒等式与配对情景检查\n每轮总收益应等于N+(乘数-1)×总贡献。对每个规则用同一组40个可用性情景，比较差值而不是挑选一次漂亮结果。', '''
            summary=[]
            for label,runs in experiments.items():
                total_rewards=[];shortages=[]
                for run in runs:
                    for row in run:
                        assert np.isclose(sum(row["payoffs"]),n+(multiplier-1)*row["contributed"])
                    total_rewards.append(sum(sum(row["payoffs"]) for row in run))
                    shortages.append(np.mean([row["shortage"]>0 for row in run]))
                summary.append(dict(rule=label,mean_total_reward=float(np.mean(total_rewards)),
                                    reward_sd=float(np.std(total_rewards,ddof=1)),shortage_rate=float(np.mean(shortages))))
            table=pd.DataFrame(summary)
            assert table.loc[table.rule=="各自不贡献","shortage_rate"].iloc[0]==1
        '''),
        sec('## 4. 看清规则、行为与故障的区别\n图展示规则在假设行为和故障机制下的结果。它不能证明人真实一定条件合作，也不能忽略轮值执行所需的协调。缺岗指标是业务结果，博弈收益只是模型内评价。', '''
            table.to_csv(OUT_DIR/"合作规则对照.csv",index=False,encoding="utf-8-sig")
            pd.DataFrame([dict(rule=label,**row) for label,runs in experiments.items() for row in runs[0]]).to_csv(OUT_DIR/"首个情景逐轮记录.csv",index=False,encoding="utf-8-sig")
            for label,runs in experiments.items():
                plt.plot(range(rounds),np.mean([[r["contributed"] for r in run] for run in runs],axis=0),label=label)
            plt.axhline(required,color="black",linestyle="--",label="业务需要");plt.xlabel("轮次");plt.ylabel("平均贡献人数");plt.legend(fontsize=8)
            plt.tight_layout();plt.savefig(OUT_DIR/"合作演变.png");plt.close()
            rotation=table.loc[table.rule=="约定轮值"].iloc[0]
            fixed=table.loc[table.rule=="固定贡献"].iloc[0]
            save_results({"rules":table.to_dict(orient="records"),"replications":len(availability)},
                         f"约定轮值在本情景组中的缺岗比例{rotation.shortage_rate:.1%}，固定全员贡献为{fixed.shortage_rate:.1%}。"
                         "轮值节省投入但冗余较少，可研究设置备用人员；条件合作遇到缺席可能连锁退化。"
                         "这些是指定策略规则的模拟结果，未声称已经证明重复博弈均衡。")
        ''')], [
        exercise('补全：公共收益会计', '贡献向量[1,0,1,0]，手算四队收益并检查总量。', '''
            reward=payoffs([1,0,1,0])
            assert np.allclose(reward,[.8,1.8,.8,1.8])
            assert np.isclose(reward.sum(),5.2)
        '''),
        exercise('修改：不发生缺席', '在全员可用情景下比较轮值与固定全员贡献，统计实际投入总人数。', '''
            full=np.ones((20,4),bool)
            rotation=simulate(["rotation"]*4,full)
            fixed=simulate(["cooperate"]*4,full)
            assert all(r["shortage"]==0 for r in rotation)
            assert sum(r["contributed"] for r in rotation)==40
            assert sum(r["contributed"] for r in fixed)==80
            print("轮值满足业务最低需求，但公共品收益目标不必与最少投入目标一致。")
        '''),
        exercise('迁移：开源项目维护', '把贡献解释为代码审查，每轮需要2人。构造全部搭便车与全员条件合作的无故障对照。', '''
            full=np.ones((6,4),bool)
            free=simulate(["defect"]*4,full)
            conditional=simulate(["conditional"]*4,full)
            assert sum(r["shortage"] for r in free)==12
            assert sum(r["shortage"] for r in conditional)==0
            print("实际协作需验证贡献成本、收益分配与可观测性。")
        ''')], [('Fischbacher等：Are people conditionally cooperative?（公共品实验原研究）','https://www.sciencedirect.com/science/article/pii/S0165176501003949')],
         '本例是重复公共品中的指定策略比较，不是联盟价值合作博弈，也未求子博弈精炼均衡。条件合作阈值、乘数、缺席率均为教学假设。')


def build_a23():
    make(23, '拥堵博弈与系统规划', '大家各选最快路，为什么总耗时还偏高？',
         '六个团队同时出发，可走容量紧张的捷径或固定耗时的绕行路。分别求个体无单方改进的配置和系统总耗时最低配置。', [
        first('## 1. 耗时依赖同路人数\n捷径上n人时每人耗时n，绕行路每人恒为5。每队是一名不可分割参与者；本例是有限原子拥堵博弈。', '''
            players=6
            def latency(route,count):return float(count if route==0 else 5)
            def individual_costs(profile):
                counts=[profile.count(r) for r in [0,1]]
                return np.array([latency(r,counts[r]) for r in profile])
            def potential(profile):
                return sum(sum(latency(r,k) for k in range(1,profile.count(r)+1)) for r in [0,1])
            def equilibrium(profile):
                current=individual_costs(profile)
                for i in range(len(profile)):
                    alternative=list(profile);alternative[i]=1-profile[i]
                    if individual_costs(alternative)[i]<current[i]-1e-9:return False
                return True
            assert individual_costs([0]*6).tolist()==[6]*6
        '''),
        sec('## 2. 逐个参与者严格改进\n每次只让一名参与者改变，而且只有成本严格下降才行动。有限拥堵博弈的势函数随这种更新下降；同步更新或无差别切换不能套用同一停止论证。', '''
            def best_response(start,limit=100):
                profile=list(start);trace=[dict(step=0,profile=list(profile),potential=potential(profile),total=float(individual_costs(profile).sum()))]
                for step in range(limit):
                    changed=False
                    for i in range(len(profile)):
                        alternative=list(profile);alternative[i]=1-profile[i]
                        if individual_costs(alternative)[i]<individual_costs(profile)[i]-1e-9:
                            before=potential(profile);profile=alternative
                            assert potential(profile)<before
                            trace.append(dict(step=len(trace),profile=list(profile),potential=potential(profile),total=float(individual_costs(profile).sum())))
                            changed=True;break
                    if not changed:return "EQUILIBRIUM",profile,trace
                return "LIMIT",profile,trace
            status,selfish,trace=best_response([0]*players)
            assert status=="EQUILIBRIUM" and equilibrium(selfish)
        '''),
        sec('## 3. 系统优化与均衡分别枚举\n6人2条路只有64种配置，用穷举取得系统最优与全部纯均衡。最佳回应稳定只保证个人不愿单独改变，未优化系统总耗时。', '''
            profiles=[list(p) for p in itertools.product([0,1],repeat=players)]
            optimal=min(profiles,key=lambda p:individual_costs(p).sum())
            equilibria=[p for p in profiles if equilibrium(p)]
            optimum=float(individual_costs(optimal).sum())
            equilibrium_costs=[float(individual_costs(p).sum()) for p in equilibria]
            assert len(equilibria)>0 and optimum<=min(equilibrium_costs)
            worst_ratio=max(equilibrium_costs)/optimum
            forced_extra=[i for i in range(players) if individual_costs(optimal)[i]>individual_costs(selfish)[i]]
        '''),
        sec('## 4. 输出协调收益和个体代价\n统一安排可能使某些参与者比其原来更慢，不能只展示总量。建议提供协调方式或补偿讨论；本例不把协调配置假装成自发均衡。', '''
            pd.DataFrame([dict(person=i,selfish_route=selfish[i],planned_route=optimal[i],
                               selfish_time=individual_costs(selfish)[i],planned_time=individual_costs(optimal)[i]) for i in range(players)]).to_csv(OUT_DIR/"个体对照.csv",index=False,encoding="utf-8-sig")
            pd.DataFrame(trace).to_csv(OUT_DIR/"最佳回应轨迹.csv",index=False,encoding="utf-8-sig")
            split_costs=[float(individual_costs([0]*n+[1]*(players-n)).sum()) for n in range(players+1)]
            plt.plot(range(players+1),split_costs,"o-");plt.xlabel("走捷径的人数");plt.ylabel("系统总耗时")
            plt.tight_layout();plt.savefig(OUT_DIR/"拥堵取舍.png");plt.close()
            save_results({"equilibrium_profile":selfish,"equilibrium_total":float(individual_costs(selfish).sum()),
                          "optimal_profile":optimal,"optimal_total":optimum,"equilibrium_count":len(equilibria),
                          "worst_equilibrium_ratio":worst_ratio,"potential_trace":[r["potential"] for r in trace]},
                         f"系统安排总耗时{optimum:g}，当前自发稳定配置为{individual_costs(selfish).sum():g}；"
                         f"需要协调的路线分配为{optimal}。相对该初始稳定配置变慢的参与者编号{forced_extra}，应解释或处理个体取舍。")
        ''')], [
        exercise('补全：检查单方偏离', '检查全走捷径不是均衡，并找出至少一次可改善的偏离。', '''
            full=[0]*players;switched=full.copy();switched[0]=1
            assert not equilibrium(full)
            assert individual_costs(switched)[0]<individual_costs(full)[0]
        '''),
        exercise('修改：从全绕行开始', '改变初始策略，观察得到的均衡与势函数下降，不要求人物编号完全相同。', '''
            state,p,history=best_response([1]*players)
            assert state=="EQUILIBRIUM" and equilibrium(p)
            assert all(b["potential"]<a["potential"] for a,b in zip(history,history[1:]))
            print(p,individual_costs(p))
        '''),
        exercise('迁移：同步跟随建议会怎样', '所有人同步切到当前看起来更快的另一条路：从全捷径到全绕行，再回来。验证这不是逐个严格改进算法。', '''
            all_short=[0]*6;all_long=[1]*6
            assert individual_costs(all_short)[0]>individual_costs([1]+[0]*5)[0]
            assert individual_costs(all_long)[0]>individual_costs([0]+[1]*5)[0]
            # 两组人同时移动后，原先假定其他人不变的判断不再成立。
            assert individual_costs(all_short).sum()==36 and individual_costs(all_long).sum()==30
            print("同步跟随会在这两种配置间循环；不能套用顺序严格改进的势函数结论。")
        ''')], [('Rosenthal：A class of games possessing pure-strategy Nash equilibria','https://doi.org/10.1007/BF01737559'),
                ('原论文公开PDF','https://ranger.uta.edu/~weems/NOTES6319/PAPERSONE/rosenthal1973.pdf')],
         '本例有限、原子、不可分割参与者，成本只依赖使用者数量。只对顺序严格改进使用势函数停止论证；其他动态需重新分析。')


def build_a24():
    make(24, '稀缺设备分配规则', '一台设备该先到先得、抽签，还是拍卖？',
         '四个团队竞争一个不可分割设备时段。价值为私人假设值，比较资源配置效率、机会、支付及个体收益，并检查第二价格拍卖的激励条件。', [
        first('## 1. 把价值、出价和到达顺序分开\n价值是获得设备的收益；出价是消息；支付是机制输出，三者不能混同。效用=获胜价值-支付，未获胜且未支付者效用0。', '''
            values=np.array([9.,6.,4.,2.]);arrival=[2,1,3,0]
            def second_price(bids):
                bids=np.asarray(bids,float)
                if len(bids)<2 or np.any(bids<0):raise ValueError("至少两人且出价非负")
                order=sorted(range(len(bids)),key=lambda i:(-bids[i],i))
                winner=order[0];payment=float(bids[order[1]])
                return winner,payment
            def utility(values,winner,payment):
                rewards=np.zeros(len(values));rewards[winner]=values[winner]-payment
                return rewards
            pd.DataFrame(dict(team=range(4),value=values,arrival_rank=[arrival.index(i) for i in range(4)])).to_csv(DATA_DIR/"团队价值.csv",index=False)
        '''),
        sec('## 2. 实现三个可检查机制\n先到先得直接用arrival[0]；公平抽签给予等概率；第二价格拍卖把设备给最高出价者，收取第二高出价。同价按编号固定打破平局。', '''
            first_winner=arrival[0]
            rng=np.random.default_rng(24);draws=rng.integers(0,len(values),size=4000)
            auction_winner,payment=second_price(values)
            counts=np.bincount(draws,minlength=len(values))
            expected_lottery_value=float(values.mean())
            assert auction_winner==0 and payment==6
            assert np.isclose(utility(values,auction_winner,payment).sum()+payment,values[auction_winner])
            table=pd.DataFrame([dict(rule="先到先得",allocated_value=values[first_winner],payment=0),
                                dict(rule="均匀抽签期望",allocated_value=expected_lottery_value,payment=0),
                                dict(rule="第二价格拍卖",allocated_value=values[auction_winner],payment=payment)])
        '''),
        sec('## 3. 单方改变报价的反例检查\n固定他人报价，遍历自己的报价，检查真实报价不会更差。这个有限枚举是本实例检查，不代替一般性定理；私人价值、准线性效用、无预算限制等条件必须明确。', '''
            deviations=[]
            for person,value in enumerate(values):
                truthful_winner,truthful_pay=second_price(values)
                truthful_u=utility(values,truthful_winner,truthful_pay)[person]
                for bid in np.arange(0,13,.5):
                    bids=values.copy();bids[person]=bid
                    winner,pay=second_price(bids);u=utility(values,winner,pay)[person]
                    deviations.append(dict(person=person,bid=bid,utility=float(u),truthful_utility=float(truthful_u)))
                    assert u<=truthful_u+1e-9
            # 获得设备的人支付第二高报价；卖方收入不等于被创造的社会价值。
            assert payment<=values[auction_winner]
        '''),
        sec('## 4. 同时报告效率与分配机会\n拍卖在本条件下配置给价值最高者，但支付能力与机会公平是另外的问题。均匀抽签的期望可精确计算，随机模拟只用来展示频率波动。', '''
            table.to_csv(OUT_DIR/"规则对照.csv",index=False,encoding="utf-8-sig")
            pd.DataFrame(deviations).to_csv(OUT_DIR/"报价偏离检查.csv",index=False)
            pd.DataFrame(dict(team=range(4),draw_count=counts,draw_frequency=counts/len(draws))).to_csv(OUT_DIR/"抽签频率.csv",index=False)
            table.set_index("rule")["allocated_value"].plot.bar(color="#688cb0")
            plt.ylabel("获配者价值／抽签期望价值");plt.xticks(rotation=15)
            plt.tight_layout();plt.savefig(OUT_DIR/"分配规则.png");plt.close()
            save_results({"first_winner":first_winner,"first_value":float(values[first_winner]),"lottery_expected_value":expected_lottery_value,
                          "lottery_observed_value":float(values[draws].mean()),"auction_winner":auction_winner,
                          "auction_payment":payment,"deviation_checks":len(deviations)},
                         f"价值效率优先且满足私人价值等假设时，第二价格规则将设备分配给团队{auction_winner}，支付{payment:g}，价值{values[auction_winner]:g}。"
                         f"均匀抽签期望价值{expected_lottery_value:g}但机会相等。公平或预算目标不同，应明确另选规则。")
        ''')], [
        exercise('补全：支付不是最高报价', '报价[3,10,7]，给出获胜者、支付和价值为10时的效用。', '''
            w,p=second_price([3,10,7])
            assert (w,p)==(1,7)
            assert utility(np.array([3,10,7]),w,p)[1]==3
        '''),
        exercise('修改：低价值团队过高报价', '团队3真实价值2，却报价12；检查它赢了但效用为负。', '''
            bids=values.copy();bids[3]=12
            w,p=second_price(bids)
            assert w==3 and p==9
            assert utility(values,w,p)[3]==-7
            print("过高出价可能让自己为不值得的资源付款。")
        '''),
        exercise('迁移：两个同价团队', '报价[5,5,1]，检查平局规则可复现、支付为5。再改变到达顺序，说明拍卖并不使用到达先后。', '''
            assert second_price([5,5,1])==(0,5)
            assert second_price([5,5,1])==second_price([5,5,1])
            print("按编号破平局；如需平等机会，可使用预先公布且独立于报价的随机破平局。")
        ''')], [('Vickrey：Counterspeculation, Auctions, and Competitive Sealed Tenders','https://doi.org/10.1111/j.1540-6261.1961.tb02789.x')],
         '单物品、私人价值、准线性效用、无预算或外部性约束是本例条件；共同价值、合谋、多时段互补需求会改变结论。实验证明只是有限样例核对。')


def build_a25():
    make(25, '合作节省额分配', '三个团队合作节省的成本怎样分配？',
         '输入每个团队联盟可创造的节省额。比较平均分、按投入分和Shapley值，另行检查任何联盟是否愿意退出；公平原则与稳定性要分别讨论。', [
        first('## 1. 联盟价值不是个人成本\n用frozenset表示联盟，字典值是该联盟相对独立行动创造的可分配节省额。所有7个非空联盟都要给出数值，不能只知道大联盟收益就计算Shapley值。', '''
            names=["甲","乙","丙"];players=tuple(range(3))
            worth={frozenset():0.,frozenset([0]):0.,frozenset([1]):0.,frozenset([2]):0.,
                   frozenset([0,1]):8.,frozenset([0,2]):6.,frozenset([1,2]):4.,frozenset([0,1,2]):12.}
            def all_coalitions(n):
                return [frozenset(s) for size in range(n+1) for s in itertools.combinations(range(n),size)]
            assert set(worth)==set(all_coalitions(3))
            pd.DataFrame([dict(coalition="".join(names[i] for i in sorted(s)) or "空",saving=v) for s,v in worth.items()]).to_csv(DATA_DIR/"联盟价值.csv",index=False,encoding="utf-8-sig")
        '''),
        sec('## 2. Shapley：平均边际贡献\n遍历6种加入顺序，每次计算新成员使联盟价值增加多少。平均后得到每人分配。它依据一组公平公理，不代表每个人都一定接受。', '''
            def shapley(values,n):
                allocation=np.zeros(n);orders=list(itertools.permutations(range(n)))
                for order in orders:
                    before=frozenset()
                    for person in order:
                        after=before|{person}
                        allocation[person]+=values[after]-values[before]
                        before=after
                return allocation/len(orders)
            allocation=shapley(worth,3)
            assert np.isclose(allocation.sum(),worth[frozenset(players)]-worth[frozenset()])
            equal=np.full(3,worth[frozenset(players)]/3)
            effort=np.array([5.,3.,2.]);proportional=effort/effort.sum()*worth[frozenset(players)]
        '''),
        sec('## 3. 单独检查个体和联盟接受条件\n如果某联盟成员分到的总额小于它独立合作能得到的价值，这个联盟有离开的利益。满足所有联盟条件且总额守恒称为核心中的分配；Shapley值不保证总在核心中。', '''
            def blocking_coalitions(shares,values):
                return [s for s in values if s and sum(shares[i] for i in s)<values[s]-1e-8]
            def in_core(shares,values):
                n=len(shares)
                return np.isclose(sum(shares),values[frozenset(range(n))]) and not blocking_coalitions(shares,values)
            schemes={"平均":equal,"投入比例":proportional,"Shapley":allocation}
            core_results={name:in_core(shares,worth) for name,shares in schemes.items()}
            # 三人简单多数游戏：任意两人值1，大联盟值1，其Shapley均分但任意两人想退出。
            majority={s:float(len(s)>=2) for s in all_coalitions(3)}
            majority_shapley=shapley(majority,3)
            assert np.allclose(majority_shapley,[1/3]*3)
            assert not in_core(majority_shapley,majority)
        '''),
        sec('## 4. 输出公平原则与稳定性对照\n表中列出不同规则分配及被哪些联盟反对。投入比例依赖投入测量，Shapley依赖所有联盟价值；真实价值估计有误时，应做敏感性检查再谈谈判建议。', '''
            table=pd.DataFrame(schemes,index=names)
            table.to_csv(OUT_DIR/"节省额分配.csv",encoding="utf-8-sig")
            table.plot.bar();plt.ylabel("分得节省额");plt.xticks(rotation=0)
            plt.tight_layout();plt.savefig(OUT_DIR/"合作分配.png");plt.close()
            blocking={name:[sorted(s) for s in blocking_coalitions(shares,worth)] for name,shares in schemes.items()}
            save_results({"shapley":allocation.tolist(),"equal":equal.tolist(),"proportional":proportional.tolist(),
                          "in_core":core_results,"blocking_coalitions":blocking,"majority_counterexample":majority_shapley.tolist()},
                         f"本联盟价值下Shapley分配为{np.round(allocation,3).tolist()}，是否在核心：{core_results['Shapley']}。"
                         f"三种规则的联盟阻挡检查为{blocking}。建议先确认所采用的公平原则，再核查联盟退出激励；Shapley并非自动稳定。")
        ''')], [
        exercise('补全：效率性质', '检查三种方案分配总额是否等于12，并解释总额相同为何分配仍不同。', '''
            assert all(np.isclose(shares.sum(),12) for shares in schemes.values())
            assert not np.allclose(equal,allocation)
            print("总额守恒不决定公平原则。")
        '''),
        exercise('修改：甲乙合作价值增加', '把联盟{甲,乙}的价值改成10，重新计算Shapley及联盟阻挡。', '''
            changed=dict(worth);changed[frozenset([0,1])]=10.
            new=shapley(changed,3)
            assert np.isclose(new.sum(),12)
            assert new[0]>allocation[0] and new[1]>allocation[1] and new[2]<allocation[2]
            print(new,"阻挡联盟",blocking_coalitions(new,changed))
        '''),
        exercise('迁移：没有边际贡献的团队', '两队甲乙共同创造6，丙加入任何联盟都不改变价值，检查丙的Shapley值为0。', '''
            dummy={s:(6. if {0,1}<=s else 0.) for s in all_coalitions(3)}
            answer=shapley(dummy,3)
            assert np.allclose(answer,[3,3,0])
            print(answer)
        ''')], [('Shapley：A Value for N-Person Games（RAND原始报告）','https://www.rand.org/pubs/papers/P295.html')],
         '采用可转移效用合作博弈，所有联盟价值为教学假设。Shapley值满足其公平公理但不保证联盟稳定；真实谈判还涉及非金钱偏好和退出成本。')


if __name__ == '__main__':
    builders=[build_a01,build_a02,build_a03,build_a04,build_a05,build_a06,build_a07,build_a08,
              build_a09,build_a10,build_a21,build_a22,build_a23,build_a24,build_a25]
    for build in builders:
        build()
        print('已生成',build.__name__)
