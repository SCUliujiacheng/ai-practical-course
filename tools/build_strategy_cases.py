"""生成仿真与连续决策的十个独立案例。"""
from course_builder import write_case
from textwrap import dedent as D
import os
import ast

def section(text, code): return (text,ast.unparse(ast.parse(D(code).strip())))
def exercise(title,prompt,answer,starter='# TODO：先写预期，再补全代码。\nresult = None\nprint("尚待完成：", result)'):
    answer=D(answer).strip()
    if starter.startswith('# TODO：先写预期'):
        tree=ast.parse(answer)
        names=[]
        for node in tree.body:
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    if isinstance(target,ast.Name) and target.id not in names:names.append(target.id)
                    elif isinstance(target,(ast.Tuple,ast.List)):
                        names.extend(x.id for x in target.elts if isinstance(x,ast.Name) and x.id not in names)
        names=names or ['your_result']
        starter='# 本题：'+prompt+'\n# 输入与函数来自上面的本例初始化；先在纸上写预期值。\n'+'\n'.join(name+' = None  # TODO：替换为你的计算' for name in names)+'\n\n# 完成后把finished改为True，并补充自己的结果检查。\nfinished = False\nif finished:\n'
        assertions=[ast.get_source_segment(answer,node) for node in tree.body if isinstance(node,ast.Assert)]
        starter+='\n'.join('    '+line for a in assertions for line in a.splitlines()) if assertions else '    print('+', '.join(names)+')'
        starter+='\nelse:\n    print("本题尚未完成，补全变量后启用检查。")'
    return dict(title=title,prompt=prompt,answer=answer,starter=starter)
def finish():
    return section('### 保存可核验的证据\n把指标写入 JSON，并用本次输出形成有条件的建议。json.dumps 把字典变成文本；不会替你判断方案是否适合现实。', r'''
    import json
    (OUT_DIR/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT_DIR/'决策建议.md').write_text('# 本次计算的决策建议\n\n'+recommendation+'\n\n## 实际指标\n```json\n'+json.dumps(metrics,ensure_ascii=False,indent=2)+'\n```\n\n这些结论只适用于本例显式假设；目标权重和风险偏好由人决定。',encoding='utf-8')
    print(metrics)
    ''')

def create(n,slug,title,intro,sections,exercises,sources,pitfalls,prereqs='基础课 Python、NumPy、实验评价'):
    selected=os.environ.get('CASE_IDS','')
    if selected and f'A{n:02d}' not in selected.split(','):return None
    return write_case('A',n,slug,title,intro,sections+[finish()],exercises,sources,prereqs=prereqs,pitfalls=pitfalls,
      decision='运行后阅读 outputs/决策建议.md。核对硬约束、成本、风险和假设；不要把模拟最高分当成现实最优。')

create(16,'增开窗口能减少多少等待','校园活动增加一个窗口，能少排多久？',
'活动服务台要决定开两个还是三个窗口。你将用 SimPy 表达排队，而不是直接套一个平均等待公式；再学习实现验证与模型现实有效性之间的区别。',[
section('### 1. 明确输入与单位\narrival 是到达时刻，service 是服务分钟数。这里生成的是教学观测，已知机制便于检验恢复参数；不能称为真实校园调查。前一天用于校准，后一天留出检验。', r'''
import numpy as np
import pandas as pd
import simpy
rng=np.random.default_rng(16)
calibration=pd.DataFrame({'interarrival':rng.exponential(2.0,400),'service':rng.gamma(3,1.3,400)})
holdout=pd.DataFrame({'interarrival':rng.exponential(2.0,200),'service':rng.gamma(3,1.3,200)})
calibration.to_csv(DATA_DIR/'calibration.csv',index=False)
holdout.to_csv(DATA_DIR/'holdout.csv',index=False)
arrival_mean=float(calibration.interarrival.mean());service_mean=float(calibration.service.mean())
print('校准均值/留出均值',arrival_mean,service_mean,holdout.mean().to_dict())
'''),
section('### 2. 写出一个顾客的过程\nyield env.timeout 表示等待模拟时间，不是真的让电脑睡几分钟。Resource 的 capacity 限制同时服务人数。先用两个人的手算小例子验证实现。', r'''
def simulate(arrivals,services,capacity):
    assert capacity>=1 and len(arrivals)==len(services)
    env=simpy.Environment();counter=simpy.Resource(env,capacity=capacity);waits=[]
    def customer(arrival,service):
        yield env.timeout(float(arrival))
        with counter.request() as request:
            yield request
            waits.append(env.now-arrival)
            yield env.timeout(float(service))
    for arrival,service in zip(arrivals,services):env.process(customer(arrival,service))
    env.run()
    return np.array(waits)
assert np.allclose(simulate([0,1],[3,2],1),[0,2])
assert np.allclose(simulate([0,1],[3,2],2),[0,0])
'''),
section('### 3. 用相同顾客情景比较\n共同随机数让两种配置面对相同顾客。每次重复生成一整天，记录日均等待与高分位等待；随后在独立留出记录上复评。服务记录采用经验重采样，不假定一定服从指数分布。', r'''
records=[]
for seed in range(40):
    day_rng=np.random.default_rng(1000+seed)
    arrivals=np.cumsum(day_rng.choice(calibration.interarrival,120))
    services=day_rng.choice(calibration.service,120)
    for capacity in [2,3]:
        w=simulate(arrivals,services,capacity)
        records.append({'seed':seed,'windows':capacity,'wait_mean':w.mean(),'wait_p90':np.quantile(w,.9),'staff_cost':capacity*160})
comparison=pd.DataFrame(records)
summary=comparison.groupby('windows')[['wait_mean','wait_p90','staff_cost']].mean()
actual_arrivals=np.cumsum(holdout.interarrival.to_numpy())
held_wait={str(c):float(simulate(actual_arrivals,holdout.service,c).mean()) for c in [2,3]}
comparison.to_csv(OUT_DIR/'daily_comparison.csv',index=False)
summary.wait_mean.plot.bar(color=['#2a6f97','#e9a23b']);plt.ylabel('每位顾客平均等待（分钟）');plt.title('共同情景下的窗口配置');plt.tight_layout();plt.savefig(OUT_DIR/'waiting.png');plt.close()
'''),
section('### 4. 把等待变化翻译成取舍\n这里额外窗口每班160元是教学假设。先展示每减少一分钟等待所需的成本，不擅自为顾客时间定价。留出数据也是模拟生成，因此这里只验证已知机制内的表现。', r'''
reduction=float(summary.loc[2,'wait_mean']-summary.loc[3,'wait_mean'])
metrics={'calibrated_interarrival_mean':arrival_mean,'calibrated_service_mean':service_mean,'mean_wait_reduction_minutes':reduction,'extra_staff_cost':160,'heldout_wait_minutes':held_wait}
recommendation=f'从两个增至三个窗口，在校准情景中平均减少等待 {reduction:.2f} 分钟/人，增加160元/班。是否值得取决于顾客时间价值和服务目标。留出记录结果为 {held_wait}；真实使用前须收集真实到达/服务记录。'
''')], [
exercise('补全：手算验证','三个顾客在0、1、2分钟到达，服务均为3分钟，只有一个窗口。写出等待时间并验证。','assert np.allclose(simulate([0,1,2],[3,3,3],1),[0,2,4])'),
exercise('修改：服务变慢','将留出服务时长乘1.5，比较两个与三个窗口。','slow={c:simulate(actual_arrivals,holdout.service*1.5,c).mean() for c in [2,3]}\nassert slow[3]<=slow[2]\nprint(slow)'),
exercise('迁移：计算机任务队列','5个任务同时提交，每个运行2秒，分别使用1与2个worker，比较平均等待。','one=simulate([0]*5,[2]*5,1);two=simulate([0]*5,[2]*5,2)\nassert np.allclose(one,[0,2,4,6,8]) and two.mean()<one.mean()\nprint(one,two)')],
[('SimPy共享资源示例','https://simpy.readthedocs.io/en/latest/simpy_intro/shared_resources.html')],
'不要混淆到达间隔和到达时刻。不能用调参时的同一情景宣称最终有效。随机种子固定仅帮助复现，不能证明现实准确。')

create(17,'不确定需求下的备餐','明天应该准备多少份餐食？',
'你面对缺货损失和剩余浪费。比较平均需求备货、分位数规则和验证集选量，把预测转成行动。',[
section('### 1. 从需求分布开始\n随机变量代表明天可能出现的需求；同一批需求必须用于公平比较。教学需求带有混合分布，避免把世界默认成正态分布。', r'''
import numpy as np
import pandas as pd
rng=np.random.default_rng(17)
def sample_demand(rng,n):
    return np.maximum(0,np.rint(rng.normal(95,18,n)+rng.binomial(1,.15,n)*45)).astype(int)
train=sample_demand(rng,500);validation=sample_demand(rng,250);test=sample_demand(rng,1000)
pd.DataFrame({'demand':train}).to_csv(DATA_DIR/'historical_demand.csv',index=False)
underage_cost=12.;overage_cost=4.
'''),
section('### 2. 定义行动损失\n损失是缺少一份的机会损失与多一份的净浪费成本，这两个参数已经包含采购/售价/残值的差额，不能再次重复加采购成本。np.maximum 将负数截为0。', r'''
def cost(quantity,demand,underage=underage_cost,overage=overage_cost):
    demand=np.asarray(demand)
    return underage*np.maximum(demand-quantity,0)+overage*np.maximum(quantity-demand,0)
assert cost(10,8)==8 and cost(10,12)==24
critical_ratio=underage_cost/(underage_cost+overage_cost)
quantile_quantity=int(np.quantile(train,critical_ratio,method='higher'))
mean_quantity=int(np.rint(train.mean()))
'''),
section('### 3. 验证选量与最终评价分开\n在验证数据上选择候选备货量，锁定后再用test。测试集只报告一次效果，不能看完结果又重选数量。', r'''
candidates=np.arange(60,161)
validation_losses=np.array([cost(q,validation).mean() for q in candidates])
selected=int(candidates[validation_losses.argmin()])
policies={'mean':mean_quantity,'critical_quantile':quantile_quantity,'validation_search':selected}
rows=[]
for name,q in policies.items():
    rows.append({'policy':name,'quantity':q,'test_cost':cost(q,test).mean(),'stockout_rate':float(np.mean(test>q)),'leftovers':np.maximum(q-test,0).mean()})
results=pd.DataFrame(rows);results.to_csv(OUT_DIR/'policies.csv',index=False)
plt.plot(candidates,validation_losses);plt.axvline(selected,color='orange',label='验证集选量');plt.xlabel('备货份数');plt.ylabel('验证平均损失（元）');plt.legend();plt.tight_layout();plt.savefig(OUT_DIR/'cost_curve.png');plt.close()
'''),
section('### 4. 需求分布变化会改变建议\n把测试需求增加20份作为压力情景。它不是新的随机种子，而是机制发生了变化；要报告原策略失效时的代价。', r'''
metrics={'critical_ratio':critical_ratio,'selected_quantity':selected,'test_results':results.to_dict('records'),'shifted_demand_cost':float(cost(selected,test+20).mean())}
recommendation=f'验证数据推荐备货 {selected} 份，独立测试平均损失 {cost(selected,test).mean():.2f} 元/日。需求整体增加20份时损失变为 {cost(selected,test+20).mean():.2f} 元/日；发现需求变化后应重新估计分布。'
''')],[
exercise('补全：手工损失','备货100份，需求80或120，等概率。求期望损失。','assert cost(100,np.array([80,120])).mean()==160'),
exercise('修改：浪费更贵','将多备一份成本改为12，利用训练集计算新分位数和数量。','new_ratio=12/(12+12)\nnew_q=int(np.quantile(train,new_ratio,method="higher"))\nassert new_q<=quantile_quantity\nprint(new_q)'),
exercise('迁移：服务器容量','需求为[2,3,4,8]，缺一个容量损失10，多一个损失1；枚举0到10选择最小经验损失。','server_demand=np.array([2,3,4,8])\nserver_q=min(range(11),key=lambda q:cost(q,server_demand,10,1).mean())\nassert server_q==8\nprint(server_q)')],
[('Quantile Regression','https://scikit-learn.org/stable/auto_examples/linear_model/plot_quantile_regression.html')],
'预测均值不一定最小化行动损失。验证集有限，样本最优值可能不稳定。教学成本来自假设，需要现实业务方确认。')

create(18,'用有限仿真预算配置备用','堵车或故障时应该预留多少备用能力？',
'你要搜索车辆备用数和触发备用的阈值。每个策略都要模拟很多场景，所以把仿真当作黑箱，比较随机搜索与Ax贝叶斯优化。',[
section('### 1. 定义可行动的策略参数\nreserve 是预先购买的备用车数量，threshold 是积压触发阈值。假设每天订单批次已到齐、故障已检查后才决定是否启用备用，这是观察后的补救动作；不能把完整日需求提前用于事前购车。模型只包含车队总容量，不能推断真实道路最优路线。训练与复评种子分开。', r'''
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(2)
def fleet_cost(reserve,threshold,seeds):
    losses=[];late_rates=[]
    for seed in seeds:
        rng=np.random.default_rng(int(seed))
        demand=rng.poisson(95,30);failures=rng.binomial(10,.08,30)
        base_capacity=(10-failures)*10
        deploy=np.where(demand-base_capacity>threshold,reserve,0)
        late=np.maximum(0,demand-base_capacity-10*deploy)
        losses.append(float(np.mean(8*late+reserve*4+deploy*2)))
        late_rates.append(float(late.sum()/demand.sum()))
    return float(np.mean(losses)),float(np.mean(late_rates))
train_seeds=list(range(180,190));test_seeds=list(range(500,550));budget=10
assert fleet_cost(2,8,[1])==fleet_cost(2,8,[1])
'''),
section('### 2. 建立同预算随机搜索\n每次目标调用使用相同的10个校准情景。记录函数调用数；不能给某方法多十倍试验再宣称它更聪明。', r'''
rng=np.random.default_rng(18)
random_trials=[]
for i in range(budget):
    parameters={'reserve':int(rng.integers(0,7)),'threshold':int(rng.integers(0,26))}
    value,_=fleet_cost(**parameters,seeds=train_seeds)
    random_trials.append(dict(trial=i,**parameters,cost=value))
random_best=min(random_trials,key=lambda row:row['cost'])
'''),
section('### 3. 用Ax逐步选试验\nAx先用空间填充试验学习，再让代理模型建议候选。这里固定前4次Sobol、后6次贝叶斯选点，防止短教程全部停留在初始化。代理模型也有成本，不能保证十次试验就找到全局最优。', r'''
from ax.service.ax_client import AxClient,ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStrategy,GenerationStep
from ax.modelbridge.registry import Models
strategy=GenerationStrategy(steps=[GenerationStep(model=Models.SOBOL,num_trials=4,model_kwargs={'seed':18}),GenerationStep(model=Models.BOTORCH_MODULAR,num_trials=-1,model_gen_kwargs={'model_gen_options':{'optimizer_kwargs':{'num_restarts':2,'raw_samples':32}}})])
client=AxClient(generation_strategy=strategy,random_seed=18,verbose_logging=False)
client.create_experiment(name='fleet_reserve',parameters=[{'name':'reserve','type':'range','bounds':[0,6],'value_type':'int'},{'name':'threshold','type':'range','bounds':[0,25],'value_type':'int'}],objectives={'cost':ObjectiveProperties(minimize=True)})
ax_trials=[]
for i in range(budget):
    parameters,trial_index=client.get_next_trial()
    value,_=fleet_cost(**parameters,seeds=train_seeds)
    client.complete_trial(trial_index=trial_index,raw_data={'cost':(value,0.0)})
    ax_trials.append(dict(trial=i,**parameters,cost=value))
ax_best=min(ax_trials,key=lambda row:row['cost'])
'''),
section('### 4. 只在选型之后独立复评\n同样的最终情景评价两个已选策略，再记录无备用基线。学习目标是形成可靠比较，不能强求贝叶斯优化一定赢。图展示已消耗预算下的历史最好值。', r'''
evaluation=[]
for name,row in [('random',random_best),('ax',ax_best),('no_reserve',{'reserve':0,'threshold':0})]:
    value,late=fleet_cost(row['reserve'],row['threshold'],test_seeds)
    evaluation.append({'method':name,'reserve':int(row['reserve']),'threshold':int(row['threshold']),'test_cost':value,'late_rate':late})
pd.DataFrame(evaluation).to_csv(OUT_DIR/'independent_evaluation.csv',index=False)
for name,trials in [('random',random_trials),('Ax',ax_trials)]:
    plt.plot(range(1,budget+1),np.minimum.accumulate([row['cost'] for row in trials]),marker='o',label=name)
plt.xlabel('已用仿真目标调用次数');plt.ylabel('当前最好校准损失');plt.legend();plt.tight_layout();plt.savefig(OUT_DIR/'search.png');plt.close()
metrics={'calls_per_method':budget,'initial_sobol_trials':4,'bayesian_trials':6,'independent_results':evaluation}
recommendation='两个搜索方法各使用10次目标调用，候选通过独立情景复评。请按 independent_evaluation.csv 比较损失和延误率；本次小预算实验不支持宣布某搜索方法普遍更优。'
''')],[
exercise('补全：无备用边界','reserve=0时，触发阈值不应影响容量。验证两个阈值损失相同。','assert fleet_cost(0,0,[1,2,3])==fleet_cost(0,25,[1,2,3])'),
exercise('修改：枚举小空间','固定阈值为10，枚举0—6辆备用车，使用训练情景选参数，测试情景复评。','q=min(range(7),key=lambda n:fleet_cost(n,10,train_seeds)[0])\nprint(q,fleet_cost(q,10,test_seeds))\nassert 0<=q<=6'),
exercise('迁移：预算账本','若一次策略评估使用10个情景，每情景30天，两个搜索各10次，总共模拟多少天？最终复评另计。','search_days=2*budget*len(train_seeds)*30\nassert search_days==6000\nprint(search_days)')],
[('Ax service API tutorial','https://ax.dev/docs/0.5.0/tutorials/gpei_hartmann_service/')],
'Ax有拟合开销，简单小空间枚举可能更好。固定校准情景只减少比较噪声，不代表现实确定。须将搜索成本与执行策略成本分开。')

create(19,'自主选路为何拥堵','每个人选择最快通道，为什么整体变慢？',
'把30位参与者看成具有有限信息的个体。每人根据上一轮的通道时间改选路线，观察反馈、震荡与统一分流。',[
section('### 1. 写清容量反馈\n两个通道耗时随人数增长，参数是教学假设。每个人只能看到上一轮耗时；如果所有人同时改选，就可能集体挤向另一条路。', r'''
import numpy as np
import pandas as pd
def travel_times(counts):
    return np.array([4.+.45*counts[0],9.+.10*counts[1]])
assert np.allclose(travel_times([0,0]),[4,9])
population=30
'''),
section('### 2. 比较全部参与与部分参与的同步更新\n所有参与者根据上一轮的旧状态作决定；switch_probability控制本轮有多少人参与更新。这仍是同步更新。真正的异步更新需要逐人修改状态，让后面的人看到新的拥堵，本例没有实现那种机制。', r'''
def simulate_routes(seed,rounds=40,switch_probability=1.):
    rng=np.random.default_rng(seed);routes=rng.integers(0,2,population);history=[]
    for t in range(rounds):
        counts=np.bincount(routes,minlength=2);times=travel_times(counts)
        history.append({'round':t,'route_a':int(counts[0]),'route_b':int(counts[1]),'mean_minutes':float(np.dot(counts,times)/population)})
        best=int(times.argmin());switch=rng.random(population)<switch_probability
        routes=np.where(switch,best,routes)
    return pd.DataFrame(history)
all_switch=simulate_routes(19);partial_switch=simulate_routes(19,switch_probability=.15)
assert (all_switch.route_a+all_switch.route_b==population).all()
'''),
section('### 3. 统一分配的目标是什么\n枚举所有人数分配，最小化总耗时。系统总量更优不代表每个人都愿意接受，个体是否想改路将在博弈课单独讨论。', r'''
splits=np.arange(population+1)
total_times=np.array([np.dot([a,population-a],travel_times([a,population-a])) for a in splits])
best_a=int(splits[total_times.argmin()]);coordinated=float(total_times.min()/population)
all_switch.to_csv(OUT_DIR/'synchronous.csv',index=False);partial_switch.to_csv(OUT_DIR/'partial.csv',index=False)
plt.plot(all_switch['round'],all_switch.mean_minutes,label='全部同时更新');plt.plot(partial_switch['round'],partial_switch.mean_minutes,label='15%概率更新');plt.axhline(coordinated,linestyle='--',label='统一分流');plt.xlabel('反馈轮次');plt.ylabel('平均通行分钟');plt.legend();plt.tight_layout();plt.savefig(OUT_DIR/'feedback.png');plt.close()
'''),
section('### 4. 报告对行为假设的敏感性\n先给出这个种子的轨迹，再重复种子；不要把单条漂亮曲线当作普遍规律。', r'''
means=[float(simulate_routes(s,switch_probability=.15).mean_minutes.iloc[-10:].mean()) for s in range(20)]
metrics={'coordinated_route_a':best_a,'coordinated_mean':coordinated,'synchronous_last10':float(all_switch.mean_minutes.iloc[-10:].mean()),'partial_last10_mean_across_runs':float(np.mean(means)),'partial_run_std':float(np.std(means,ddof=1))}
recommendation=f'系统总耗时最小的人数分配是A通道{best_a}人、B通道{population-best_a}人，平均{coordinated:.2f}分钟。自主更新的表现依赖更新频率；实施分流前需要验证参与者是否遵守与通道容量模型是否合理。'
''')],[
exercise('补全：人数守恒','逐轮检查两条通道人数非负且总和恒为30。','assert ((all_switch[["route_a","route_b"]]>=0).all().all())\nassert (all_switch.route_a+all_switch.route_b==30).all()'),
exercise('修改：所有人不改路','把更新概率设为0，检查每轮人数相同。','fixed=simulate_routes(5,switch_probability=0)\nassert fixed.route_a.nunique()==1\nprint(fixed.head())'),
exercise('迁移：随机分流','每位参与者独立以0.5概率去A，模拟1000次；和确定分配15/15比较平均总耗时。','rng2=np.random.default_rng(21)\na=rng2.binomial(30,.5,1000)\nrandom_cost=np.mean([np.dot([x,30-x],travel_times([x,30-x])) for x in a])\nfixed_cost=np.dot([15,15],travel_times([15,15]))\nprint(random_cost,fixed_cost)\nassert random_cost>=fixed_cost-1')],
[('Mesa traffic agent model inspiration','https://mesa.readthedocs.io/latest/overview.html')],
'这是原创NumPy个体仿真，未依赖Mesa。行为参数未经现实校准；不能把同步更新导致的震荡误写成任何现实人群必然的行为。')

create(20,'压力测试与失效边界','在常规模拟中最好的方案，遇到变化还可靠吗？',
'比较固定资源与带备用的策略。正常、需求抬升、厚尾、相关故障分别构成测试情景，输出成本、尾部风险和服务目标的全部通过率。',[
section('### 1. 写出可手算的策略评价函数\n政策事前决定常备和备用人数。假设每时段服务开始前已观察该批待处理需求和故障，然后从已预约备用池中启用人员；这不是提前知道未来到达。先检查零需求和容量上限。服务目标是本次事件所有时段都满足容量。', r'''
import numpy as np
import pandas as pd
def evaluate_policy(policy,demand,failure):
    base,reserve=policy;available=np.maximum(0,base-failure)
    activate=np.minimum(reserve,np.maximum(0,np.ceil(demand/10)-available))
    shortage=np.maximum(0,demand-10*(available+activate))
    cost=base*12+reserve*3+activate*8+shortage*15
    return cost,shortage
assert evaluate_policy((2,0),np.array([0]),np.array([0]))[0][0]==24
assert evaluate_policy((2,0),np.array([30]),np.array([0]))[1][0]==10
policies={'lean':(10,0),'reserve':(10,3),'large':(13,0)}
'''),
section('### 2. 留出多类机制变化\n每个情景有100个活动日，每日8个时段。相关故障中，一次事件影响同一天的所有时段。用共享场景比较策略，但不给策略在这些最终情景上调参。', r'''
def scenario(kind,seed):
    rng=np.random.default_rng(seed);d=rng.poisson(85,(100,8)).astype(float);f=rng.binomial(10,.03,(100,8))
    if kind=='higher':d+=25
    if kind=='heavy_tail':d+=rng.binomial(1,.05,(100,8))*80
    if kind=='correlated_failure':f+=rng.binomial(1,.15,(100,1))*4
    return d,f
records=[];paired={}
for i,kind in enumerate(['normal','higher','heavy_tail','correlated_failure']):
    d,f=scenario(kind,2000+i)
    for name,policy in policies.items():
        c,s=evaluate_policy(policy,d,f);daily=c.sum(axis=1)
        paired[(kind,name)]=daily
        records.append({'scenario':kind,'policy':name,'mean_cost':float(daily.mean()),'p95_daily_cost':float(np.quantile(daily,.95)),'all_periods_pass_rate':float(np.mean((s==0).all(axis=1))),'period_pass_rate':float(np.mean(s==0))})
results=pd.DataFrame(records)
'''),
section('### 3. 成对差值区间与情景遗憾\n先按天聚合，再对天重采样，避免把同一天相关的时段当作独立样本。这里区间估计的是该仿真机制下的平均成本差，不是现实效果保证。', r'''
rng=np.random.default_rng(20);differences=paired[('normal','reserve')]-paired[('normal','lean')]
boot=np.array([rng.choice(differences,len(differences),replace=True).mean() for _ in range(500)])
interval=np.quantile(boot,[.025,.975])
results['scenario_regret']=results.mean_cost-results.groupby('scenario').mean_cost.transform('min')
results.to_csv(OUT_DIR/'stress_matrix.csv',index=False)
pivot=results.pivot(index='scenario',columns='policy',values='mean_cost')
pivot=pivot.rename(index={'normal':'常态','higher':'需求抬升','heavy_tail':'厚尾需求','correlated_failure':'相关故障'},columns={'lean':'精简配置','reserve':'预留备用','large':'常备增员'})
pivot.plot.bar();plt.xlabel('压力情景');plt.ylabel('平均每日成本（元，8时段合计）');plt.xticks(rotation=0);plt.legend(title='策略');plt.tight_layout();plt.savefig(OUT_DIR/'stress.png');plt.close()
'''),
section('### 4. 说明风险偏好与边界\n平均成本、尾部风险、全部通过率可能冲突。不要把满足95%时段写成95%的活动全部可行。本例是固定候选的压力测试，尚未对未观测极端情景提供保证。', r'''
metrics={'cost_unit':'元/活动日（8时段合计）','normal_reserve_minus_lean_mean':float(differences.mean()),'bootstrap_95_interval':interval.tolist(),'stress_results':results.to_dict('records')}
recommendation='请依据 stress_matrix.csv 同时比较平均成本、95%分位日成本与整日全通过率。若所有时段都必须有容量，优先检查 all_periods_pass_rate；表中出现缺口的方案不能因平均得分高就称为满足硬约束。'
''')],[
exercise('补全：两种通过率','2天各2时段，短缺矩阵[[0,1],[0,0]]，计算逐时段和整天通过率。','s=np.array([[0,1],[0,0]])\nassert (s==0).mean()==.75\nassert (s==0).all(axis=1).mean()==.5'),
exercise('修改：增加备用','独立新情景中比较(10,3)与(10,5)，解释为什么成本未必降低。','d,f=scenario("higher",9999)\na=evaluate_policy((10,3),d,f);b=evaluate_policy((10,5),d,f)\nassert b[1].sum()<=a[1].sum()\nprint(a[0].mean(),b[0].mean())'),
exercise('迁移：稳健选择','从已公布压力测试中，找出最大情景遗憾最小的策略，仅作描述性分析；若采用它，需新测试集复评。','worst=results.groupby("policy").scenario_regret.max()\nprint(worst.idxmin(),worst.to_dict())\nassert set(worst.index)==set(policies)')],
[('Simulation verification and validation','https://informs-sim.org/wsc11papers/016.pdf')],
'没有真实记录时只能做条件推演。测试情景不能覆盖未知的一切。图中95%分位是成本分布的分位数，与均值估计区间不是一回事。')

create(26,'连续补货与滚动规划','连续三天活动，每晚应补多少货？',
'将库存变成状态，采购变成动作。动态规划同时考虑今天与明天；滚动方法每晚更新真实库存，只执行当前动作。',[
section('### 1. 列出状态、动作与转移\n库存容量为8，三天需求可取0/2/4。先假设需求分布已知。采购、缺货与持有成本统一为元。末日剩余仍计持有成本，无残值。', r'''
import numpy as np
import pandas as pd
from functools import lru_cache
CAPACITY=8
support=np.array([0,2,4]);probability=np.array([.15,.45,.40])
def transition(stock,order,demand):
    assert 0<=order<=CAPACITY-stock
    available=stock+order;left=max(available-demand,0);short=max(demand-available,0)
    return left,2*order+9*short+.5*left
assert transition(2,2,3)==(1,4.5)
'''),
section('### 2. 倒推未来价值\nV(days,stock)返回未来最小期望成本和当前最优动作。lru_cache记住重复子问题，避免指数重复计算；每次动作都受容量限制。', r'''
@lru_cache(None)
def optimal(days,stock):
    if days==0:return 0.,0
    candidates=[]
    for order in range(CAPACITY-stock+1):
        value=0.
        for demand,p in zip(support,probability):
            nxt,now=transition(stock,order,int(demand))
            value+=p*(now+optimal(days-1,nxt)[0])
        candidates.append((value,order))
    return min(candidates)
policy_table=pd.DataFrame([{'remaining_days':days,'stock':stock,'order':optimal(days,stock)[1],'value':optimal(days,stock)[0]} for days in [1,2,3] for stock in range(CAPACITY+1)])
policy_table.to_csv(OUT_DIR/'policy.csv',index=False)
'''),
section('### 3. 执行反馈而非照抄整条计划\nfull使用剩余全部天数；rolling2只看最多两天；myopic只看一天。每执行一步都用真实需求更新库存，因此三者均为闭环策略。', r'''
def rollout(demands,mode):
    stock=0;total=0.;trace=[]
    for day,demand in enumerate(demands):
        remaining=len(demands)-day
        if mode=='full':horizon=remaining
        elif mode=='rolling2':horizon=min(2,remaining)
        else:horizon=1
        order=optimal(horizon,stock)[1]
        nxt,cost=transition(stock,order,int(demand))
        trace.append({'day':day+1,'stock':stock,'order':order,'observed_demand':int(demand),'next_stock':nxt,'cost':cost})
        stock=nxt;total+=cost
    return total,trace
rng=np.random.default_rng(26);paths=rng.choice(support,(600,3),p=probability)
records=[]
for mode in ['full','rolling2','myopic']:
    costs=np.array([rollout(d,mode)[0] for d in paths]);records.append({'mode':mode,'mean':costs.mean(),'std':costs.std()})
comparison=pd.DataFrame(records)
pd.DataFrame(rollout([4,0,4],'rolling2')[1]).to_csv(OUT_DIR/'execution_trace.csv',index=False)
comparison.plot.bar(x='mode',y='mean',legend=False);plt.ylabel('三天总成本（元）');plt.tight_layout();plt.savefig(OUT_DIR/'horizon.png');plt.close()
'''),
section('### 4. 检查理论值与模拟\n大量独立模拟的均值应接近动态规划的期望值，但不是逐次相等。再使用需求总为4的压力轨迹说明模型偏差。', r'''
full_costs=np.array([rollout(d,'full')[0] for d in paths]);se=full_costs.std(ddof=1)/np.sqrt(len(paths))
assert abs(full_costs.mean()-optimal(3,0)[0])<5*se+.1
metrics={'exact_optimal_expected_cost':optimal(3,0)[0],'simulation':comparison.to_dict('records'),'heavy_demand_total':rollout([4,4,4],'rolling2')[0]}
recommendation=f'已知小模型的最优期望成本为 {optimal(3,0)[0]:.2f} 元。滚动策略应在每天观察需求后重新求解；若需求机制变化，需要更新概率模型，不能仅增加求解次数。'
'''),
section('### 5. 什么条件下看得更远才有价值？\n上面的简单模型中三种策略恰好相同，这是实验结果，不能宣称动态规划总能改善结果。现在增加“每次下单固定支付8元配送费”，多买一些以减少未来下单次数才有价值。保持同一批需求轨迹，比较三种视野；最后一天仍没有残值。', r'''
@lru_cache(None)
def optimal_with_fee(days,stock,fee):
    if days==0:return 0.,0
    candidates=[]
    for order in range(CAPACITY-stock+1):
        value=0.
        for demand,p in zip(support,probability):
            nxt,now=transition(stock,order,int(demand))
            now+=fee if order>0 else 0
            value+=p*(now+optimal_with_fee(days-1,nxt,fee)[0])
        candidates.append((value,order))
    return min(candidates)
def rollout_with_fee(demands,mode,fee=8.):
    stock=0;total=0.
    for day,demand in enumerate(demands):
        remaining=len(demands)-day
        horizon=remaining if mode=='full' else min(2,remaining) if mode=='rolling2' else 1
        order=optimal_with_fee(horizon,stock,fee)[1]
        stock,now=transition(stock,order,int(demand))
        total+=now+(fee if order>0 else 0)
    return total
fee_comparison=[]
for mode in ['full','rolling2','myopic']:
    costs=np.array([rollout_with_fee(d,mode) for d in paths])
    fee_comparison.append({'mode':mode,'mean':float(costs.mean()),'std':float(costs.std())})
assert optimal_with_fee(3,0,0)==optimal(3,0)
assert optimal_with_fee(3,0,8)[1]==8 and optimal_with_fee(1,0,8)[1]==4
pd.DataFrame(fee_comparison).to_csv(OUT_DIR/'fixed_fee_comparison.csv',index=False)
pd.DataFrame(fee_comparison).plot.bar(x='mode',y='mean',legend=False,rot=0)
plt.xlabel('规划视野');plt.ylabel('三天总成本（元）');plt.title('加入每次8元配送费后的独立模拟');plt.tight_layout();plt.savefig(OUT_DIR/'fixed_fee.png');plt.close()
metrics['fixed_order_fee']=8
metrics['with_fee_exact_optimal_expected_cost']=optimal_with_fee(3,0,8)[0]
metrics['with_fee_simulation']=fee_comparison
recommendation+=' 基础设置中三种视野得到相同策略；加入每次8元配送费后再比较 fixed_fee_comparison.csv，理解跨日耦合为何改变规划价值。'
''')],[
exercise('补全：容量约束','验证策略表中每个动作都不超过剩余容量。','assert (policy_table.stock+policy_table.order<=CAPACITY).all()'),
exercise('修改：初始库存','分别查看初始库存0与6时三天问题的采购量和价值，解释沉没采购成本未计入。','print(optimal(3,0),optimal(3,6))\nassert optimal(3,6)[1]<=2'),
exercise('迁移：一期穷举','仅一天、初始0，直接枚举采购0到8，复算最小期望成本。','brute=min((sum(p*transition(0,q,int(d))[1] for d,p in zip(support,probability)),q) for q in range(9))\nassert np.isclose(brute[0],optimal(1,0)[0])\nprint(brute)')],
[('Model predictive control theory','https://www.do-mpc.com/en/latest/theory_mpc.html')],
'状态必须包含影响未来的信息。无限期策略不适用于有限三天活动；未来真实需求不能在决策时提前读取。')

create(27,'未知拥堵下学习配送策略','拥堵概率未知，怎样通过交互学习策略？',
'用小型配送过程比较规则、已知模型最优规划和表格Q-learning。规划器知道真实转移概率，是信息更充分的参照，不是与学习者完全同条件。',[
section('### 1. 环境与动作\n状态为距离终点剩余段数0—5，0是终点。动作0走稳定慢路，每次耗时2且前进一段；动作1走快路，耗时1但可能堵住原地。这里真实概率仅在环境和oracle中可见。', r'''
import numpy as np
import pandas as pd
SUCCESS=.58;STATES=6
def step(state,action,rng):
    if state==0:return 0,0.,True
    advance=(action==0) or rng.random()<SUCCESS
    return max(0,state-int(advance)),(-2. if action==0 else -1.),state-int(advance)==0
def evaluate(policy,seeds):
    totals=[]
    for seed in seeds:
        rng=np.random.default_rng(seed);state=5;total=0.
        for _ in range(200):
            state,reward,done=step(state,int(policy[state]),rng);total-=reward
            if done:break
        assert done
        totals.append(total)
    return np.array(totals)
'''),
section('### 2. 已知模型的对照\n对于每段独立且相同的环境，慢路期望2分钟，快路的几何等待期望1/p分钟。只有当p大于0.5时快路更优。规则基线固定慢路。', r'''
oracle=np.full(STATES,int(1/SUCCESS<2));safe=np.zeros(STATES,dtype=int)
assert oracle[5]==1
print('单段理论期望：慢路2，快路',1/SUCCESS)
'''),
section('### 3. 学习者只看交互样本\nQ表的每个元素是某状态采取某动作的估计回报。探索率逐渐下降；评价时停止更新和探索。每次完整训练使用独立种子，而不是训练一次换测试种子。', r'''
def train(seed,episodes=350):
    rng=np.random.default_rng(seed);q=np.zeros((STATES,2));interactions=0
    for episode in range(episodes):
        state=5;epsilon=max(.05,.8*(1-episode/episodes))
        for _ in range(200):
            action=int(rng.integers(2)) if rng.random()<epsilon else int(q[state].argmax())
            nxt,reward,done=step(state,action,rng);interactions+=1
            target=reward+(0 if done else q[nxt].max())
            q[state,action]+=.12*(target-q[state,action]);state=nxt
            if done:break
    return q.argmax(axis=1),interactions,q
seeds=list(range(2700,2900));runs=[]
for seed in range(8):
    policy,interactions,q=train(seed)
    runs.append({'seed':seed,'mean_time':float(evaluate(policy,seeds).mean()),'interactions':interactions,'policy':policy.tolist()})
pd.DataFrame(runs).to_csv(OUT_DIR/'independent_training_runs.csv',index=False)
'''),
section('### 4. 汇总独立训练的波动\n区间基于独立训练所得均值，用于展示训练随机性；八次运行只提供有限证据。业务评价为耗时而非只看累计reward上升。', r'''
scores=np.array([r['mean_time'] for r in runs]);halfwidth=2.365*scores.std(ddof=1)/np.sqrt(len(scores))
values=[evaluate(safe,seeds).mean(),evaluate(oracle,seeds).mean(),scores.mean()]
plt.bar(['固定慢路','已知模型参照','Q-learning'],values);plt.ylabel('每次配送分钟');plt.tight_layout();plt.savefig(OUT_DIR/'policies.png');plt.close()
metrics={'safe_time':float(values[0]),'oracle_time':float(values[1]),'q_mean_time':float(scores.mean()),'training_mean_95_t_interval':[float(scores.mean()-halfwidth),float(scores.mean()+halfwidth)],'mean_interactions':float(np.mean([r['interactions'] for r in runs]))}
recommendation=f'本环境快路真实成功率为{SUCCESS}，已知模型参照拥有更多信息。学习策略平均耗时{scores.mean():.2f}分钟，且需要训练交互；若模型已知且状态很小，应先用规划而不是增加神经网络。'
''')],[
exercise('补全：终点','终止状态不应产生奖励或继续移动。','assert step(0,1,np.random.default_rng(0))==(0,0.,True)'),
exercise('修改：减少样本','将训练降到30回合，与350回合在相同测试情景比较；不要求小样本每次都差。','p_small,n_small,_=train(88,30);p_large,n_large,_=train(88,350)\nassert n_small<n_large\nprint(evaluate(p_small,seeds).mean(),evaluate(p_large,seeds).mean())'),
exercise('迁移：计算临界成功率','快路每次成本变为1.4，慢路仍2。求临界成功概率并在0.6/0.8判断选路。','threshold=1.4/2\nassert threshold==.7\nprint({p:("快路" if 1.4/p<2 else "慢路") for p in [.6,.8]})')],
[('Reinforcement learning evaluation tips','https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html')],
'改变测试种子不等于独立训练。不要让oracle的信息泄漏给学习者；其优势须说明。连续不受限探索仅适用于可接受代价的模拟环境。')

create(28,'边试边选服务方案','不知道哪种服务引导有效，怎样边试边选择？',
'你有三种服务引导话术，只有被采用的话术能观察是否成功。比较随机、贪心、ε-greedy与UCB，理解探索本身具有成本。',[
section('### 1. 反馈机制\n真实成功率只给环境和评价器，策略仅收到被选动作的0/1结果。伪遗憾用真实率计算，因此只在此类已知机制模拟中可得。', r'''
import numpy as np
import pandas as pd
rates=np.array([.40,.48,.58]);horizon=600
def run_bandit(method,seed,steps=horizon):
    rng=np.random.default_rng(seed);counts=np.zeros(3,int);success=np.zeros(3);rewards=[];regret=[]
    for t in range(steps):
        average=np.divide(success,counts,out=np.zeros(3),where=counts>0)
        if t<3:action=t
        elif method=='random':action=int(rng.integers(3))
        elif method=='epsilon' and rng.random()<.1:action=int(rng.integers(3))
        elif method=='ucb':action=int(np.argmax(average+np.sqrt(2*np.log(t+1)/counts)))
        else:action=int(np.argmax(average))
        reward=int(rng.random()<rates[action]);counts[action]+=1;success[action]+=reward
        rewards.append(reward);regret.append(float(rates.max()-rates[action]))
    return np.cumsum(rewards),np.cumsum(regret),counts
'''),
section('### 2. 单次轨迹看直觉\n先观察一个种子。UCB把估计均值与不确定性奖励相加；没有充分尝试的动作会得到更高探索奖励。', r'''
methods=['random','greedy','epsilon','ucb']
for method in methods:
    reward,regret,counts=run_bandit(method,28)
    assert counts.sum()==horizon and np.all(np.diff(regret)>=-1e-10)
    plt.plot(regret,label=method)
plt.xlabel('决策轮次');plt.ylabel('累计伪遗憾');plt.legend();plt.tight_layout();plt.savefig(OUT_DIR/'one_run.png');plt.close()
'''),
section('### 3. 多次独立重复\n重复100条轨迹，报告累计成功和伪遗憾的均值及分布。比较相同轮数；不预设UCB在每个短期问题都优于ε-greedy。', r'''
rows=[]
for method in methods:
    for seed in range(100):
        reward,regret,counts=run_bandit(method,seed+2800)
        rows.append({'method':method,'seed':seed,'successes':int(reward[-1]),'regret':float(regret[-1]),'best_arm_selections':int(counts[2])})
results=pd.DataFrame(rows)
summary=results.groupby('method').agg(successes=('successes','mean'),regret=('regret','mean'),regret_std=('regret','std'))
results.to_csv(OUT_DIR/'trials.csv',index=False)
summary.regret.plot.bar();plt.ylabel('600轮平均伪遗憾');plt.tight_layout();plt.savefig(OUT_DIR/'repeat.png');plt.close()
'''),
section('### 4. 区分短期目标与长期学习\n这里假设成功率不随时间变化且每次结果独立。真实话术可能影响后续顾客构成或产生延迟反馈，届时该模型不再直接适用。', r'''
metrics={'true_rates':rates.tolist(),'horizon':horizon,'runs_per_method':100,'summary':summary.reset_index().to_dict('records')}
recommendation='根据 trials.csv 选择与时间预算和探索代价相符的策略。本例动作不改变后续状态，使用老虎机即可；若行动影响未来库存或用户状态，需重新建立序贯模型。'
''')],[
exercise('补全：次数守恒','100轮UCB总选择次数应为100，各动作至少试一次。','_,_,counts=run_bandit("ucb",5,100)\nassert counts.sum()==100 and (counts>=1).all()'),
exercise('修改：极短活动','将活动限制为20轮，重复50次后比较各策略成功数。','short={m:np.mean([run_bandit(m,s,20)[0][-1] for s in range(50)]) for m in methods}\nprint(short)\nassert all(0<=v<=20 for v in short.values())'),
exercise('迁移：遗憾的含义','连续选第一种方案10次，计算期望伪遗憾；说明它不同于本次真实少成功几人。','expected_regret=10*(rates.max()-rates[0])\nassert np.isclose(expected_regret,1.8)\nprint(expected_regret)')],
[('Sutton and Barto Reinforcement Learning book','http://incompleteideas.net/book/the-book-2nd.html')],
'策略不能读取未选动作的实际反馈。模拟中计算遗憾所用真实概率不应传给策略。固定成功率只是教学假设。')

create(29,'调查与预测的信息价值','再做一次调查，最多值得付多少钱？',
'活动组织者可以直接决定是否转场，也可以先购买有噪声的天气信号。计算贝叶斯更新、完美信息上界与有噪声信息价值。',[
section('### 1. 定义状态、动作与损失\n状态依次为晴、雨，动作依次为室外、转场。每个格子的损失为元。概率与损失来自本题假设，不能冒充天气机构预报。', r'''
import numpy as np
import pandas as pd
prior=np.array([.65,.35])
loss=np.array([[0.,1200.],[350.,350.]])
def choose(probability):
    expected=loss@probability
    return int(expected.argmin()),float(expected.min())
action,baseline_loss=choose(prior)
assert np.isclose(prior.sum(),1)
'''),
section('### 2. 完美信息给出上界\n先观察真实状态再行动的期望损失，不会高于不知道状态时。EVPI是两者差额，调查无论多准确都不应在这个模型下比它更值钱。', r'''
perfect_loss=float(prior@loss.min(axis=0))
evpi=baseline_loss-perfect_loss
assert evpi>=0
'''),
section('### 3. 噪声信号与贝叶斯更新\nlikelihood[signal,state]表示给定天气状态时出现该信号的概率。列和必须为1。posterior与prior不同，不能把准确率直接当作下雨后验。', r'''
likelihood=np.array([[.8,.25],[.2,.75]]) # 两行分别为晴信号/雨信号
assert np.allclose(likelihood.sum(axis=0),1)
records=[];after_information_loss=0.
for signal in range(2):
    joint=likelihood[signal]*prior;signal_probability=float(joint.sum());posterior=joint/signal_probability
    chosen,conditional_loss=choose(posterior)
    after_information_loss+=signal_probability*conditional_loss
    records.append({'signal':signal,'signal_probability':signal_probability,'posterior_rain':float(posterior[1]),'action':chosen,'conditional_loss':conditional_loss})
evsi=baseline_loss-after_information_loss
assert -1e-8<=evsi<=evpi+1e-8
pd.DataFrame(records).to_csv(OUT_DIR/'signal_decisions.csv',index=False)
'''),
section('### 4. 信息需要改变行动才可能有价值\n比较不同购买价格的净收益。只有行动选择、信息可靠性和损失模型都正确时，这个购买上限才成立。', r'''
prices=np.arange(0,301,10);plt.plot(prices,evsi-prices);plt.axhline(0,color='gray');plt.xlabel('调查价格（元）');plt.ylabel('购买信息的期望净价值（元）');plt.tight_layout();plt.savefig(OUT_DIR/'information_value.png');plt.close()
metrics={'baseline_action':action,'baseline_loss':baseline_loss,'perfect_information_value':evpi,'noisy_information_value':evsi,'signal_policy':records}
recommendation=f'当前先验下直接行动的最小期望损失为{baseline_loss:.2f}元；该噪声信号价值为{evsi:.2f}元，因此在其他成本不变时购买价格应低于该值。完美信息价值{evpi:.2f}元只是上界。'
''')],[
exercise('补全：后验','出现雨信号时，手算下雨后验并与公式验证。','post=.75*.35/(.75*.35+.2*.65)\nassert np.isclose(post,records[1]["posterior_rain"])\nprint(post)'),
exercise('修改：无信息信号','两种天气下信号均为50/50，证明信息价值为0。','uninformative=np.full((2,2),.5)\nexpected=sum(float((row*prior).sum())*choose(row*prior/(row*prior).sum())[1] for row in uninformative)\nassert np.isclose(baseline_loss-expected,0)'),
exercise('迁移：信息购买决策','报价分别为50、150、300元，给出是否值得购买，注意等于价值时无净收益。','decisions={price:price<evsi for price in [50,150,300]}\nprint(decisions)\nassert not decisions[300]')],
[('Bayesian decision theory textbook chapter','https://artint.info/3e/html/ArtInt3e.Ch12.html')],
'先验、似然、后验不要混淆。信息价值不是信息准确率；同一信号对不同决策可能价值不同。')

create(30,'校园开放日综合参谋项目','怎样形成能执行、能应变的开放日计划？',
'综合任务：预测6个时段的客流、安排志愿者、推演服务容量、加入应急触发，并独立检查排班。先读任务书，再尝试自己实现；不要求把所有算法都用一遍。',[
section('### 1. 任务书与预测信息边界\n6个时段，12名常规志愿者各最多工作4段，每段每人服务12人。另有独立预约的2名外援，可各待命6段；待命每人每段10元，实际启用再加15元，总外援预算300元。历史记录是教学模拟。按天划分训练、校准、测试，预测只用时段和是否周末，不读取未来实际客流。', r'''
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from ortools.sat.python import cp_model
rng=np.random.default_rng(30)
rows=[]
for day in range(100):
    for slot in range(6):
        weekend=int(day%7>=5);mean=20+20*np.sin((slot+1)*np.pi/7)+12*weekend
        rows.append({'day':day,'slot':slot,'weekend':weekend,'visitors':int(rng.poisson(mean))})
history=pd.DataFrame(rows);history.to_csv(DATA_DIR/'history.csv',index=False)
features=['slot','weekend'];train=history[history.day<60];cal=history[(history.day>=60)&(history.day<80)];test=history[history.day>=80]
model=RandomForestRegressor(n_estimators=40,min_samples_leaf=8,random_state=30,n_jobs=1).fit(train[features],train.visitors)
residual=cal.visitors-model.predict(cal[features]);buffer=float(np.quantile(residual,.9,method='higher'))
tomorrow=pd.DataFrame({'slot':range(6),'weekend':[1]*6});forecast=model.predict(tomorrow)
required=np.ceil((forecast+max(0,buffer))/12).astype(int)
assert len(required)==6
'''),
section('### 2. 约束求解与独立检查\n变量x[worker,slot]表示是否工作。CP-SAT要求整数约束。求解状态表示模型中的结论，独立检查器重新依据任务要求计算覆盖与工时。', r'''
def schedule(requirements,workers=12,max_shifts=4):
    m=cp_model.CpModel();x={(w,t):m.NewBoolVar(f'w{w}_t{t}') for w in range(workers) for t in range(6)}
    for t in range(6):m.Add(sum(x[w,t] for w in range(workers))>=int(requirements[t]))
    for w in range(workers):m.Add(sum(x[w,t] for t in range(6))<=max_shifts)
    m.Minimize(sum(x.values()));solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=5;solver.parameters.num_search_workers=1
    status=solver.Solve(m)
    if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):return solver.StatusName(status),None
    return solver.StatusName(status),np.array([[solver.Value(x[w,t]) for t in range(6)] for w in range(workers)])
def check_plan(plan,requirements,max_shifts=4):
    return bool(plan is not None and plan.shape==(12,6) and np.isin(plan,[0,1]).all() and (plan.sum(axis=0)>=requirements).all() and (plan.sum(axis=1)<=max_shifts).all())
status,plan=schedule(required);assert check_plan(plan,required)
broken=plan.copy();broken[:,0]=0;assert not check_plan(broken,required)
impossible_status,impossible=schedule(np.full(6,13));assert impossible is None and impossible_status=='INFEASIBLE'
pd.DataFrame(plan,columns=[f'slot_{t}' for t in range(6)]).to_csv(OUT_DIR/'staff_schedule.csv',index_label='worker')
'''),
section('### 3. 留出日回放与压力情景\n开放日目标是周末，因此主评价仅使用留出的6个周末，不能混入低需求工作日来稀释风险。全部20天另存作辅助分析；6天仅支持小样本演示。每时段未完成人数进入下一时段，这个聚合模型不等于A16的离散事件排队。独立外援池最多2人、总启用最多12人时段；启用决策只看上段积压。单独检查人员、工时和300元预算，不把外援当作免费无限资源。', r'''
def execute(demands,capacity,use_reserve=False):
    backlog=0;trace=[];remaining_reserve_shifts=12
    for t,d in enumerate(demands):
        # 本段开始只看到上段积压，不偷看本段将到达的客流。
        reserve=min(2,remaining_reserve_shifts) if use_reserve and backlog>10 else 0
        remaining_reserve_shifts-=reserve
        backlog=max(0,backlog+int(d)-int(capacity[t])-12*reserve)
        trace.append({'slot':t,'arrivals':int(d),'reserve_staff':reserve,'backlog_end':backlog})
    return trace
capacity=plan.sum(axis=0)*12
target_test=test[test.weekend==1]
normal_days=[g.sort_values('slot').visitors.to_numpy() for _,g in target_test.groupby('day')]
assert len(normal_days)==6 and target_test.weekend.eq(1).all()
evaluations=[]
for scenario,days in [('normal',normal_days),('demand_plus20',[d+20 for d in normal_days])]:
    for reserve in [False,True]:
        traces=[execute(d,capacity,reserve) for d in days]
        reserve_shifts=[sum(r['reserve_staff'] for r in tr) for tr in traces]
        reserve_costs=[120+15*n if reserve else 0 for n in reserve_shifts]
        assert all(all(0<=r['reserve_staff']<=2 for r in tr) for tr in traces)
        assert max(reserve_shifts)<=12 and max(reserve_costs)<=300
        evaluations.append({'scenario':scenario,'reserve':reserve,'mean_final_backlog':float(np.mean([tr[-1]['backlog_end'] for tr in traces])),'all_slots_zero_backlog_rate':float(np.mean([all(r['backlog_end']==0 for r in tr) for tr in traces])),'mean_reserve_shifts':float(np.mean(reserve_shifts)),'mean_reserve_cost':float(np.mean(reserve_costs)),'reserve_constraints_pass':True})
pd.DataFrame(evaluations).to_csv(OUT_DIR/'stress_evaluation.csv',index=False)
all_days=[g.sort_values('slot').visitors.to_numpy() for _,g in test.groupby('day')]
auxiliary=[]
for scenario,days in [('normal',all_days),('demand_plus20',[d+20 for d in all_days])]:
    for reserve in [False,True]:
        traces=[execute(d,capacity,reserve) for d in days]
        auxiliary.append({'population':'全部留出日期，非周末目标主评价','days':len(days),'scenario':scenario,'reserve':reserve,'mean_final_backlog':float(np.mean([tr[-1]['backlog_end'] for tr in traces]))})
pd.DataFrame(auxiliary).to_csv(OUT_DIR/'all_days_auxiliary.csv',index=False)
trace=execute(normal_days[0]+20,capacity,True);pd.DataFrame(trace).to_csv(OUT_DIR/'execution_trace.csv',index=False)
'''),
section('### 4. 条件改变时修订\n若第3段开始收到后续需求抬升的信息，只修订未执行时段。此处展示容量目标的更新与可行性重检；真实排班要同时扣除已工作工时，不能把新全日表当成可直接执行的后半日表。', r'''
revised_requirements=required.copy();revised_requirements[3:]+=2
def replan_remaining(old,requirements,cut=3):
    m=cp_model.CpModel();x={(w,t):m.NewBoolVar(f'r_{w}_{t}') for w in range(12) for t in range(6)}
    for w in range(12):
        for t in range(cut):m.Add(x[w,t]==int(old[w,t]))
        m.Add(sum(x[w,t] for t in range(6))<=4)
    for t in range(6):m.Add(sum(x[w,t] for w in range(12))>=int(requirements[t]))
    m.Minimize(sum(x.values()));s=cp_model.CpSolver();s.parameters.max_time_in_seconds=5;s.parameters.num_search_workers=1
    st=s.Solve(m)
    if st not in (cp_model.OPTIMAL,cp_model.FEASIBLE):return s.StatusName(st),None
    return s.StatusName(st),np.array([[s.Value(x[w,t]) for t in range(6)] for w in range(12)])
revision_status,revised=replan_remaining(plan,revised_requirements)
if revised is not None:
    assert np.array_equal(revised[:,:3],plan[:,:3]) and check_plan(revised,revised_requirements)
    pd.DataFrame(revised).to_csv(OUT_DIR/'revised_schedule.csv',index=False)
plt.plot(forecast,label='预测客流');plt.plot(capacity,label='计划容量');plt.plot(forecast+buffer,linestyle='--',label='含经验缓冲的目标');plt.xlabel('时段');plt.ylabel('人数');plt.legend();plt.tight_layout();plt.savefig(OUT_DIR/'capacity.png');plt.close()
metrics={'solver_status':status,'hard_constraints_pass':check_plan(plan,required),'total_staff_shifts':int(plan.sum()),'evaluation_population':'留出周末','heldout_target_days':len(normal_days),'forecast_mae_on_heldout_weekends':float(np.mean(abs(target_test.visitors-model.predict(target_test[features])))),'empirical_buffer':buffer,'stress_evaluation':evaluations,'revision_status':revision_status,'revision_feasible':revised is not None}
recommendation=f'原排班状态{status}，全部排班硬约束检查通过，总工时段{plan.sum()}。主评价使用留出的{len(normal_days)}个周末，压力情景结果见stress_evaluation.csv；积压超过10人时下一段启用2名预留人员。需求抬升后的修订状态为{revision_status}。缓冲为历史经验分位数，未声称所有未来活动达到90%整案保证。'
''')],[
exercise('补全：独立工时检查','不调用check_plan，逐人检查每人最多4段且取值为0/1。','assert np.isin(plan,[0,1]).all()\nassert all(sum(row)<=4 for row in plan)'),
exercise('修改：不可行的需求','每段要求13人而只有12人，求解器应证明无解；不能拿上一份排班冒充新结果。','st,p=schedule(np.full(6,13))\nassert st=="INFEASIBLE" and p is None'),
exercise('迁移：预约处理中心','6时段到达人数均60，每段基础容量48，启用同样积压触发规则。独立计算首两段积压。','tr=execute([60]*6,[48]*6,True)\nassert tr[0]["backlog_end"]==12 and tr[1]["reserve_staff"]==2 and tr[1]["backlog_end"]==0\nprint(tr)')],
[('OR-Tools employee scheduling','https://developers.google.com/optimization/scheduling/employee_scheduling'),('Time series lagged features','https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html')],
'预测偏差、容量模型误差、志愿者缺勤都可能导致失败。排班约束满足不等于真实服务不会排队。预留人员成本、服务公平和人员偏好需要在真实任务中补充。')

if __name__=='__main__': print('Created A16—A20 and A26—A30')

