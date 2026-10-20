"""离线拓展：固定主案例的八个测试窗口，对比 Chronos-Bolt-tiny。

先运行同目录 solution.py 生成模拟序列和基线回测结果。
本脚本不训练模型、不改主线结果、不从测试误差选提示或参数。
"""
from pathlib import Path
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
import json
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from chronos import BaseChronosPipeline

CASE_DIR=Path(__file__).resolve().parent
ROOT=CASE_DIR.parents[1]
OUT=CASE_DIR/'outputs'
torch.set_num_threads(2)
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})

# 1. 复用主案例的数据、测试窗口与结果。这里不用互联网示例 CSV。
input_path=CASE_DIR/'data/simulated_daily_energy.csv'
reference_path=OUT/'rolling_predictions.csv'
if not input_path.is_file() or not reference_path.is_file():
    raise FileNotFoundError('先运行本例 solution.py，生成日用电序列和主线回测结果。')
series=pd.read_csv(input_path,parse_dates=['date']).set_index('date')['kwh']
reference=pd.read_csv(reference_path,parse_dates=['origin','date'])
origins=sorted(reference['origin'].unique())
assert len(origins)==8 and len(reference)==56

# 2. 模型配置中的分位数不是随机样本轴。按配置找到0.1/0.5/0.9。
model_path=ROOT/'assets/models/chronos-bolt-tiny'
config=json.loads((model_path/'config.json').read_text(encoding='utf-8'))
quantiles=config['chronos_config']['quantiles']
lower_index=quantiles.index(.1);median_index=quantiles.index(.5);upper_index=quantiles.index(.9)
started=time.perf_counter()
pipeline=BaseChronosPipeline.from_pretrained(
    str(model_path),device_map='cpu',torch_dtype=torch.float32,local_files_only=True)
load_seconds=time.perf_counter()-started

# 3. 每个预测起点的上下文严格结束于该日，输出未来7日，不传未来实际值。
contexts=[]
for origin in origins:
    history=series.loc[:pd.Timestamp(origin)]
    assert history.index[-1]==pd.Timestamp(origin)
    contexts.append(torch.tensor(history.to_numpy(),dtype=torch.float32))
started=time.perf_counter()
with torch.inference_mode():
    forecast=pipeline.predict(contexts,prediction_length=7).cpu().numpy()
inference_seconds=time.perf_counter()-started
assert forecast.shape==(len(origins),len(quantiles),7)
assert np.isfinite(forecast).all()
rows=[]
for i,origin in enumerate(origins):
    for h in range(7):
        rows.append({'origin':pd.Timestamp(origin),'date':pd.Timestamp(origin)+pd.Timedelta(days=h+1),
                     'horizon':h+1,'chronos_median':float(forecast[i,median_index,h]),
                     'chronos_q10':float(forecast[i,lower_index,h]),'chronos_q90':float(forecast[i,upper_index,h])})
result=reference.merge(pd.DataFrame(rows),on=['origin','date','horizon'],validate='one_to_one')
assert len(result)==56
assert np.allclose(result['actual'],series.loc[pd.DatetimeIndex(result['date'])].to_numpy())

# 4. 公平比较相同56个预测位置；80%模型分位数不是已证明覆盖的区间。
metrics={
    'model':'amazon/chronos-bolt-tiny','test_windows':len(origins),'evaluation_points':len(result),
    'chronos_mae':float(np.abs(result['actual']-result['chronos_median']).mean()),
    'forest_mae':float(np.abs(result['actual']-result['forest']).mean()),
    'seasonal_mae':float(np.abs(result['actual']-result['seasonal']).mean()),
    'q10_q90_empirical_coverage':float(((result['actual']>=result['chronos_q10'])&(result['actual']<=result['chronos_q90'])).mean()),
    'q10_q90_mean_width':float((result['chronos_q90']-result['chronos_q10']).mean()),
    'model_load_seconds':round(load_seconds,3),'batch_inference_seconds':round(inference_seconds,3),
    'data':'author-generated synthetic series; same test origins as core case',
    'coverage_guarantee_claimed':False,'new_domain_zero_shot_claimed':False,
    'comparison_boundary':'univariate Chronos vs calendar-plus-lags forest; complete pipeline comparison, not isolated architecture effect'}
result.to_csv(OUT/'chronos_comparison.csv',index=False,encoding='utf-8-sig')
(OUT/'chronos_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].bar(['同星期基线','随机森林','Chronos tiny'],[metrics['seasonal_mae'],metrics['forest_mae'],metrics['chronos_mae']])
axes[0].set_title('同一测试窗口 MAE / kWh')
last=result.tail(7)
axes[1].plot(last['date'],last['actual'],'o-',label='实际')
axes[1].plot(last['date'],last['chronos_median'],'o-',label='Chronos 中位数')
axes[1].fill_between(last['date'],last['chronos_q10'],last['chronos_q90'],alpha=.2,label='q10—q90')
axes[1].legend();axes[1].tick_params(axis='x',rotation=30)
fig.tight_layout();fig.savefig(OUT/'chronos_comparison.png');plt.close(fig)
print(json.dumps(metrics,ensure_ascii=False,indent=2))
