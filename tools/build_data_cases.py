"""B01–B15 / A11–A15 作者脚本；算法展开到各例，不依赖共享黑盒。"""
from textwrap import dedent, indent
import ast
import os
from course_builder import write_case

def S(text, code):
    return text, dedent(code).strip() + '\n'

def E(title, prompt, answer, starter=None):
    answer = dedent(answer).strip()
    if starter is None:
        # 给出本题专用脚手架；默认不开启，避免未完成的省略号阻断整本预览。
        tree = ast.parse(answer)
        candidates = [node for node in ast.walk(tree)
                      if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
                      and node.value is not None
                      and not isinstance(node.value, (ast.Constant, ast.Name))]
        assertion_candidates = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
        if candidates:
            value = max(candidates, key=lambda node: node.lineno).value
            replacement = '...'
        elif assertion_candidates:
            value = max(assertion_candidates, key=lambda node: node.lineno).test
            replacement = 'False'
        else:
            value = None
        if value is not None:
            lines = answer.splitlines(keepends=True)
            def offset(line, byte_column):
                return sum(map(len, lines[:line-1])) + len(lines[line-1].encode('utf-8')[:byte_column].decode('utf-8'))
            start = offset(value.lineno, value.col_offset)
            end = offset(value.end_lineno, value.end_col_offset)
            scaffold = answer[:start] + replacement + answer[end:]
        else:
            scaffold = '# TODO：按题意编写实现，再运行下面的检查。\n' + answer
        starter = ('# 先独立思考；需要提示时使用下面的本题脚手架。\n'
                   '# 补全 ...（或 assert False 占位条件）后，将 RUN_EXERCISE 改为 True。\n'
                   'RUN_EXERCISE = False\nif RUN_EXERCISE:\n' + indent(scaffold, '    '))
    return dict(title=title, prompt=prompt, starter=starter, answer=answer)

def make(track, number, slug, title, intro, sections, exercises, sources, **kw):
    assert len(sections) >= 4 and len(exercises) == 3
    selected = os.environ.get('CASE_IDS', '')
    if selected and f'{track}{number:02d}' not in selected.split(','):
        return None
    return write_case(track, number, slug, title, intro, sections, exercises, sources, **kw)

PANDAS = 'https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html'
SK = 'https://scikit-learn.org/stable/'

make('B', 1, '天气CSV统计', '怎样从天气 CSV 找到最热的一天？',
     '输入是一份 10 天的教学天气记录。先不用 pandas：亲手把文本转成数字，遍历记录，输出日最高温、平均值和高温天数。', [
S('## 1. 读入：文件中的数字最初也是字符串\n`with` 负责关闭文件，`DictReader` 用首行作字段名；`list` 把逐行读取器变成可重复查看的列表。输入是 CSV，输出是字典列表。先观察类型，避免按字符串大小排序。', r'''
import csv, json
with (DATA_DIR / 'weather.csv').open(encoding='utf-8', newline='') as stream:
    raw_rows = list(csv.DictReader(stream))
print(raw_rows[:2], type(raw_rows[0]['temp_max']))
assert len(raw_rows) == 10
'''),
S('## 2. 转换：建立清楚的数据约定\n逐条转成浮点数，同时检查日期和合理范围。`append` 将一条结果加入列表。检查失败时停止，比忽略错误继续出结论更容易排查。', r'''
from datetime import date
rows = []
for raw in raw_rows:
    value = float(raw['temp_max'])
    date.fromisoformat(raw['date'])
    assert -60 <= value <= 60
    rows.append({'date': raw['date'], 'temp_max': value})
print(rows[0])
'''),
S('## 3. 函数：把统计过程变成能重复调用的步骤\n函数接收记录与阈值，返回字典。最热日相同温度时保留最先出现的日期；空列表必须显式报错。循环里的累计和与计数，是以后向量化计算的基础。', r'''
def summarize(records, threshold=30):
    if not records:
        raise ValueError('没有记录，不能计算平均值')
    hottest = records[0]
    total = 0.0
    count = 0
    for row in records:
        total += row['temp_max']
        if row['temp_max'] > hottest['temp_max']:
            hottest = row
        if row['temp_max'] >= threshold:
            count += 1
    return {'hottest_date': hottest['date'], 'hottest_temp': hottest['temp_max'],
            'mean_temp': total / len(records), 'hot_days': count, 'threshold': threshold}
metrics = summarize(rows)
assert metrics['hottest_date'] == '2024-07-06'
assert metrics['hot_days'] == 4
print(metrics)
'''),
S('## 4. 交付：数值与图互相检查\n把结果写为 JSON，便于程序再次读取；图用于发现异常，不代替精确数值。折线的每个点应对应一行输入，阈值线帮助核对计数。', r'''
(OUT_DIR / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
plt.plot([r['date'][5:] for r in rows], [r['temp_max'] for r in rows], marker='o')
plt.axhline(30, linestyle='--', color='orange', label='30°C')
plt.ylabel('最高温度 / °C'); plt.xlabel('日期'); plt.legend(); plt.tight_layout()
plt.savefig(OUT_DIR / 'temperature.png'); plt.close()
''')], [
E('补全：核对平均值', '用循环重算平均值，并与 summarize 的结果比较；不要直接抄结果数字。', r'''
total = 0
for row in rows:
    total += row['temp_max']
average = total / len(rows)
assert abs(average - metrics['mean_temp']) < 1e-10
print(average)
'''),
E('修改：改变高温定义', '将阈值从 30 改成 32；结果应不超过原来的高温天数，并核对恰有 2 天。', r'''
changed = summarize(rows, threshold=32)
assert changed['hot_days'] == 2
assert changed['hot_days'] <= metrics['hot_days']
print(changed)
'''),
E('迁移：另一座城市与空输入', '给另一座城市的 3 天记录计算结果，再验证空输入确实报错。', r'''
other = [{'date': '2024-01-01', 'temp_max': 10}, {'date': '2024-01-02', 'temp_max': 15}, {'date': '2024-01-03', 'temp_max': 12}]
assert summarize(other, 13)['hot_days'] == 1
try:
    summarize([])
except ValueError:
    print('空输入检查通过')
else:
    raise AssertionError('空输入不应返回有效平均值')
''')], [('Python CSV 官方文档','https://docs.python.org/3/library/csv.html')],
prereqs='能读懂变量赋值即可；本例讲解列表、字典、循环、函数和文件路径。',
pitfalls='温度字符串必须转为数字；这里的均值是日最高温均值，不是全天平均温。小数据为手工构造，不能用于气候结论。',
data_note='手工构造的 10 天教学天气数据，单位 °C，无真实城市含义。',
extra_files={'data/weather.csv':'date,temp_max\n2024-07-01,25\n2024-07-02,27\n2024-07-03,28\n2024-07-04,30\n2024-07-05,32\n2024-07-06,34\n2024-07-07,31\n2024-07-08,29\n2024-07-09,26\n2024-07-10,24\n'})

make('B', 2, '合并多个气象站', '怎样合并多个气象站并按城市统计？',
     '把分散的记录纵向合并，再把站点属性横向连接。区分 concat 与 merge，发现遗漏站点，并用同一任务练习可选 SQLite 查询。', [
S('## 1. 多文件读取：每个文件是一块同结构数据\n`glob` 找匹配文件，`sorted` 固定处理顺序。列表推导生成多个 DataFrame；`concat` 连接行，不会按站点自动匹配。保留来源文件便于追查。', r'''
import pandas as pd, json
frames = []
for path in sorted(DATA_DIR.glob('station_*.csv')):
    frame = pd.read_csv(path, parse_dates=['date'])
    frame['source'] = path.name
    frames.append(frame)
weather = pd.concat(frames, ignore_index=True)
assert len(weather) == 9
assert not weather.duplicated(['station', 'date']).any()
print(weather.head())
'''),
S('## 2. 属性连接：站点表必须一站一行\n`many_to_one` 要求右表键唯一，避免一次合并把记录数翻倍。左连接保留所有天气记录；缺失城市必须查清，不能默默丢弃。', r'''
stations = pd.read_csv(DATA_DIR / 'stations.csv')
merged = weather.merge(stations, on='station', how='left', validate='many_to_one')
assert len(merged) == len(weather)
assert merged['city'].notna().all()
print(merged[['station', 'city', 'date', 'temp_max']])
'''),
S('## 3. 分组：一行代表什么会改变平均值\n先算每个城市每天的站点平均，再跨天平均，避免站点较多的一天被额外加权。分组后的表是新的分析单位。`pivot` 将日期变成行、城市变成列。', r'''
city_day = merged.groupby(['city', 'date'], as_index=False)['temp_max'].mean()
summary = city_day.groupby('city')['temp_max'].agg(['mean', 'max', 'count'])
daily_wide = city_day.pivot(index='date', columns='city', values='temp_max')
print(summary)
assert len(city_day) == 6
'''),
S('## 4. 导出与检查：让另一人能复算\n保存合并明细和城市日均表，同时输出样本量。城市间差异仅是这三天的教学样本差异，不应扩大为常年温度差异。', r'''
merged.to_csv(OUT_DIR / 'merged.csv', index=False, encoding='utf-8-sig')
city_day.to_csv(OUT_DIR / 'city_daily.csv', index=False, encoding='utf-8-sig')
metrics = {'records': len(merged), 'city_days': len(city_day), 'means': summary['mean'].to_dict()}
(OUT_DIR / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
daily_wide.plot(marker='o'); plt.ylabel('站点日最高温均值 / °C'); plt.tight_layout()
plt.savefig(OUT_DIR / 'city_comparison.png'); plt.close()
''')], [
E('补全：检查连接键', '统计每个站点几天有记录，并验证站点元数据唯一。', r'''
counts = merged.groupby('station')['date'].nunique()
assert counts.eq(3).all()
assert stations['station'].is_unique
print(counts)
'''),
E('修改：只统计前两天', '筛选日期早于 2024-07-03 的记录，再按城市和日期算均值。', r'''
early = merged.loc[merged['date'] < pd.Timestamp('2024-07-03')]
early_summary = early.groupby(['city', 'date'])['temp_max'].mean()
assert len(early) == 6 and len(early_summary) == 4
print(early_summary)
'''),
E('迁移：用 SQLite 复算城市日均', '可选 SQL 迁移：把已合并数据写入内存数据库，使用 GROUP BY 复算，不建立服务器。', r'''
import sqlite3
with sqlite3.connect(':memory:') as connection:
    merged.to_sql('weather', connection, index=False)
    sql_result = pd.read_sql_query('SELECT city, date, AVG(temp_max) AS temp_max FROM weather GROUP BY city, date ORDER BY city, date', connection)
sql_result['date'] = pd.to_datetime(sql_result['date'])
pd.testing.assert_frame_equal(sql_result, city_day.sort_values(['city','date']).reset_index(drop=True))
print(sql_result)
''')], [('pandas 合并与连接', 'https://pandas.pydata.org/docs/user_guide/merging.html')],
prereqs='B01；本例首次引入 DataFrame、按列选择、分组和连接。',
pitfalls='同名字段不代表同一含义；单位必须一致。merge 的重复键会制造伪样本；缺失城市不能用内连接掩盖。SQLite 是可选迁移，不阻塞 pandas 主线。',
extra_files={'data/station_A.csv':'station,date,temp_max\nA,2024-07-01,25\nA,2024-07-02,27\nA,2024-07-03,29\n', 'data/station_B.csv':'station,date,temp_max\nB,2024-07-01,27\nB,2024-07-02,29\nB,2024-07-03,31\n','data/station_C.csv':'station,date,temp_max\nC,2024-07-01,21\nC,2024-07-02,23\nC,2024-07-03,25\n','data/stations.csv':'station,city\nA,海城\nB,海城\nC,山城\n'})

make('B', 3, '清洗传感器记录', '缺失、重复和异常数据应该怎样处理？',
     '使用含脏值的传感器记录，制作逐项可追踪的清洗规则。展示缺测与异常的差别，并使用只看过去的填充，避免时间泄漏。', [
S('## 1. 保留原始输入\n原始文件不可被清洗覆盖。先展示缺陷：非法文本、重复时间和超出物理范围的读数。`errors="coerce"` 把无法转换的项变成缺失，但必须同时记录数量。', r'''
import pandas as pd, numpy as np, json
raw = pd.read_csv(DATA_DIR / 'sensor_dirty.csv')
work = raw.copy()
work['time'] = pd.to_datetime(work['time'], errors='coerce')
work['temperature'] = pd.to_numeric(work['temperature'], errors='coerce')
report = {'input_rows': len(raw), 'invalid_time': int(work['time'].isna().sum()), 'unparseable_or_missing': int(work['temperature'].isna().sum())}
print(work)
'''),
S('## 2. 定义行的身份与规则\n本题规定同一时间重复上传以最后一条为准。这是业务假设，不是所有任务的通用处理。先排序再去重，并保留修改标记；物理异常置缺失而非直接删除整天。', r'''
work = work.dropna(subset=['time']).sort_values('time', kind='stable')
report['duplicate_rows'] = int(work.duplicated('time', keep='last').sum())
work = work.drop_duplicates('time', keep='last').copy()
work['out_of_range'] = ~work['temperature'].between(-40, 60) & work['temperature'].notna()
report['range_violations'] = int(work['out_of_range'].sum())
work.loc[work['out_of_range'], 'temperature'] = np.nan
'''),
S('## 3. 填充只使用过去：留下可追查标志\n这里最多向前填 1 个点，连续长缺测不能伪造平滑曲线。`ffill` 只使用已观测历史；本例不做双向插值。删除无法处理的开头缺测时计入报告。', r'''
work['was_missing'] = work['temperature'].isna()
work['temperature_clean'] = work['temperature'].ffill(limit=1)
report['remaining_missing'] = int(work['temperature_clean'].isna().sum())
clean = work.dropna(subset=['temperature_clean']).copy()
assert clean['time'].is_unique
assert clean['temperature_clean'].between(-40, 60).all()
assert report['duplicate_rows'] == 1
'''),
S('## 4. 输出清洗日志，而不只交一张干净表\n清洗报告告诉使用者保留了多少记录、改动了哪些值。图中的空缺和填充值应能对应审计列；需要训练模型时，统计型填充值仍须只在训练集估计。', r'''
report['output_rows'] = len(clean)
clean.to_csv(OUT_DIR / 'clean.csv', index=False, encoding='utf-8-sig')
work.to_csv(OUT_DIR / 'audit.csv', index=False, encoding='utf-8-sig')
(OUT_DIR / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
plt.plot(work['time'], work['temperature'], 'o', label='有效原值')
plt.plot(work['time'], work['temperature_clean'], '--', label='单步前向填充')
plt.legend(); plt.ylabel('温度 / °C'); plt.xticks(rotation=25); plt.tight_layout()
plt.savefig(OUT_DIR / 'cleaning.png'); plt.close(); print(report)
''')], [
E('补全：数出被填充的点', '仅统计原先缺失且现在非空的记录，并核对本例为 2 个。', r'''
filled = work['was_missing'] & work['temperature_clean'].notna()
assert int(filled.sum()) == 2
print(work.loc[filled, ['time', 'temperature_clean']])
'''),
E('修改：允许连续填两点', '比较 limit=1 与 limit=2，确认缺失数量不增加，并解释为什么更多填充不一定更好。', r'''
longer = work['temperature'].ffill(limit=2)
assert longer.isna().sum() <= work['temperature_clean'].isna().sum()
print('多填充点数', int(longer.notna().sum() - work['temperature_clean'].notna().sum()))
'''),
E('迁移：湿度的物理范围', '为湿度 [45, -2, 110, 60, None] 设置 0—100 的有效范围；不要沿用温度规则。', r'''
humidity = pd.Series([45, -2, 110, 60, None], dtype=float)
valid = humidity.where(humidity.between(0, 100))
assert valid.notna().sum() == 2
assert valid.dropna().tolist() == [45, 60]
print(valid)
''')], [('pandas 缺失数据处理','https://pandas.pydata.org/docs/user_guide/missing_data.html')],
prereqs='B01—B02；布尔条件、缺失值、copy 与按条件赋值。', pitfalls='异常不一定是传感器坏了；实际规则应由测量范围与业务解释支持。此处只演示一条明确的教学规则。',
extra_files={'data/sensor_dirty.csv':'time,temperature\n2024-07-01 00:00,20\n2024-07-01 01:00,21\n2024-07-01 01:00,22\n2024-07-01 02:00,bad\n2024-07-01 03:00,24\n2024-07-01 04:00,999\n2024-07-01 05:00,\n2024-07-01 06:00,26\n'})

make('B', 4, '解析训练日志', '怎样把训练日志自动变成学习曲线？',
     '解析文本中的 epoch、训练误差和验证误差；识别未解析行，选择验证误差最低的轮次，并演示过拟合。日志为预设教学样例，不冒充真实训练结果。', [
S('## 1. 读取混杂日志\n真实日志可能混入警告。先读成逐行文本，不假设每一行都是训练结果；保留行号，解析失败才有位置可找。', r'''
import re, json
import pandas as pd
lines = (DATA_DIR / 'training.log').read_text(encoding='utf-8').splitlines()
print('\n'.join(lines[:4]))
'''),
S('## 2. 正则表达式提取有名字的字段\n`(?P<epoch>...)` 给捕获组命名；浮点模式接受小数与科学计数法。没有匹配的行另外保存，避免“成功运行”掩盖数据漏读。', r'''
number = r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?'
pattern = re.compile(r'epoch=(?P<epoch>\d+)\s+train=(?P<train>'+number+r')\s+val=(?P<val>'+number+r')')
parsed, skipped = [], []
for line_no, line in enumerate(lines, 1):
    match = pattern.search(line)
    if match:
        parsed.append({'epoch': int(match['epoch']), 'train': float(match['train']), 'val': float(match['val'])})
    else:
        skipped.append({'line': line_no, 'text': line})
table = pd.DataFrame(parsed).sort_values('epoch')
assert table['epoch'].is_unique
assert (table[['train', 'val']] >= 0).all().all()
'''),
S('## 3. 验证集选择停止点\n训练误差下降并不意味着泛化改善。`idxmin` 返回最小值的行索引，再用 loc 取整行。这里没有测试集，不能把最小验证误差当最终测试表现。', r'''
best = table.loc[table['val'].idxmin()]
table['gap'] = table['val'] - table['train']
metrics = {'parsed_epochs': len(table), 'skipped_lines': len(skipped), 'best_epoch': int(best['epoch']), 'best_validation_loss': float(best['val'])}
assert metrics['best_epoch'] == 4
print(table, metrics)
'''),
S('## 4. 保存供复核的结构化结果\n同时交付曲线、解析表和未解析行，便于分清格式错误与训练问题。纵轴应标损失而不是准确率。', r'''
table.to_csv(OUT_DIR / 'epochs.csv', index=False)
(OUT_DIR / 'unparsed.json').write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
plt.plot(table['epoch'], table['train'], label='训练损失')
plt.plot(table['epoch'], table['val'], label='验证损失')
plt.axvline(best['epoch'], linestyle='--', color='grey'); plt.legend(); plt.xlabel('epoch'); plt.ylabel('损失')
plt.tight_layout(); plt.savefig(OUT_DIR / 'learning_curve.png'); plt.close()
''')], [
E('补全：核对最优轮次', '不用 idxmin，用 Python min 与 key 找验证损失最低的一行。', r'''
chosen = min(parsed, key=lambda row: row['val'])
assert chosen['epoch'] == metrics['best_epoch']
print(chosen)
'''),
E('修改：处理科学计数法', '解析一行 train=1e-3、val=2.5e-2 的新日志，验证转换值。', r'''
match = pattern.search('epoch=9 train=1e-3 val=2.5e-2')
assert match is not None
assert abs(float(match['val']) - 0.025) < 1e-12
print(match.groupdict())
'''),
E('迁移：早停耐心值', '写 patience=2 的扫描函数：验证损失连续两轮未刷新最低值时停止。此样例应在第 6 轮触发。', r'''
lowest, stale, stop_epoch = float('inf'), 0, None
for row in parsed:
    if row['val'] < lowest:
        lowest, stale = row['val'], 0
    else:
        stale += 1
    if stale >= 2:
        stop_epoch = row['epoch']; break
assert stop_epoch == 6
print('触发早停轮次', stop_epoch)
''')], [('Python 正则表达式','https://docs.python.org/3/library/re.html')],
prereqs='B01；字符串、正则命名组、字典列表与图表。', pitfalls='日志是教学预设，不代表模型实测训练；本例实测结果指解析与选轮次的输出。验证集选轮次后仍需要独立测试集。',
data_note='training.log 是手工构造的过拟合演示日志，含一行警告和科学计数法。',
extra_files={'data/training.log':'epoch=1 train=0.90 val=0.95\nepoch=2 train=0.65 val=0.70\nWARNING: example warning\nepoch=3 train=0.48 val=0.54\nepoch=4 train=0.35 val=0.49\nepoch=5 train=0.27 val=0.53\nepoch=6 train=2.0e-1 val=0.59\nepoch=7 train=0.16 val=0.67\n'})

make('B', 5, '可复现数据清单', '怎样把带标签的文件整理成可复现的数据集？',
     '建立文件清单并划分训练、验证、测试。模拟同一人的多条记录，说明随机按行切分会让同一人的信息泄漏到测试集；本例按人切分。', [
S('## 1. 文件和标签是两种信息\n生成小型教学文件，文件名不编码标签。标签表记录 sample_id、person_id 和 label；同一人的记录必须保持在同一划分。路径使用相对路径才能搬迁课程。', r'''
import json, hashlib
import pandas as pd
from sklearn.model_selection import train_test_split
sample_dir = DATA_DIR / 'samples'; sample_dir.mkdir(exist_ok=True)
rows = []
for person in range(30):
    label = person % 2
    for visit in range(2):
        sample_id = f'p{person:02d}_v{visit}'
        path = sample_dir / (sample_id + '.txt')
        path.write_text(f'Teaching record for person {person}, visit {visit}.', encoding='utf-8')
        rows.append({'sample_id': sample_id, 'person_id': person, 'label': label, 'path': path.relative_to(CASE_DIR).as_posix()})
manifest = pd.DataFrame(rows)
assert manifest['sample_id'].is_unique
'''),
S('## 2. 在独立单位上分层抽样\n先取每人的一行信息，再分层保持类别比例。固定随机种子使同样的输入得到同样的名单；随机种子不能修复错误的划分单位。', r'''
people = manifest[['person_id', 'label']].drop_duplicates()
train_people, held_people = train_test_split(people, test_size=0.4, stratify=people['label'], random_state=5)
val_people, test_people = train_test_split(held_people, test_size=0.5, stratify=held_people['label'], random_state=5)
split_map = {}
for split, subset in [('train', train_people), ('validation', val_people), ('test', test_people)]:
    split_map.update({int(person): split for person in subset['person_id']})
manifest['split'] = manifest['person_id'].map(split_map)
'''),
S('## 3. 自动检查泄漏与内容版本\n断言不同划分没有相同 person_id。SHA-256 用于发现内容改变，不是匿名化工具。检查每个文件存在，并把校验值随清单保存。', r'''
groups = {name: set(part['person_id']) for name, part in manifest.groupby('split')}
assert groups['train'].isdisjoint(groups['test'])
assert groups['train'].isdisjoint(groups['validation'])
assert groups['test'].isdisjoint(groups['validation'])
manifest['sha256'] = [hashlib.sha256((CASE_DIR / p).read_bytes()).hexdigest() for p in manifest['path']]
assert manifest['sha256'].str.len().eq(64).all()
'''),
S('## 4. 清单成为数据集入口\n保存 JSON Lines 和 CSV，使训练脚本可以按 split 读取。输出类别计数图；如果数据具有时间关系，还应增加时间约束，本例仅演示独立人物分组。', r'''
manifest.to_json(OUT_DIR / 'manifest.jsonl', orient='records', lines=True, force_ascii=False)
manifest.to_csv(OUT_DIR / 'manifest.csv', index=False)
counts = pd.crosstab(manifest['split'], manifest['label'])
metrics = {'samples': len(manifest), 'people': len(people), 'counts': manifest['split'].value_counts().to_dict(), 'group_leakage': False}
(OUT_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
counts.plot.bar(); plt.ylabel('样本数'); plt.xticks(rotation=0); plt.tight_layout()
plt.savefig(OUT_DIR / 'splits.png'); plt.close(); print(counts)
''')], [
E('补全：数量守恒', '验证三个划分合计 60 条，且每人恰有 2 条记录。', r'''
assert manifest.groupby('split').size().sum() == 60
assert manifest.groupby('person_id').size().eq(2).all()
print(manifest['split'].value_counts())
'''),
E('修改：换种子不破坏分组', '用 random_state=19 重做按人 80/20 划分，验证人没有重叠。', r'''
left, right = train_test_split(people, test_size=0.2, stratify=people['label'], random_state=19)
assert set(left['person_id']).isdisjoint(right['person_id'])
assert len(right) == 6
print(right)
'''),
E('迁移：检测被修改的文件', '对内存中的内容加一个字符后重算哈希，证明与清单不同；不要损坏原始文件。', r'''
row = manifest.iloc[0]
original = (CASE_DIR / row['path']).read_bytes()
changed_hash = hashlib.sha256(original + b'!').hexdigest()
assert changed_hash != row['sha256']
assert hashlib.sha256(original).hexdigest() == row['sha256']
print('内容变更检测通过')
''')], [('scikit-learn 交叉验证与分组划分',SK+'modules/cross_validation.html')],
prereqs='B01—B04；Path、JSON、集合、不交集与随机种子。', pitfalls='这些小文件仅用来练习清单，不是真实医疗或人体数据。按行随机划分可能把同一人物放进训练和测试。')
make('B', 6, '预测明天最高温', '怎样根据历史天气预测明天的最高温度？',
     '使用本地真实 Jena 日统计数据。在今天观测结束后预测明天最高温，核心比较持续性基线与 Ridge 线性模型；用时间划分防止看到未来。', [
S('## 1. 对齐到每日时间轴\n日期缺口会让 shift(1) 变成上一条而非昨天，因此先补齐每日索引。保留缺失，让后续明确删除不完整窗口。输入来自 assets/data/jena_daily.csv，单位与来源见数据说明。', r'''
import pandas as pd, numpy as np, json
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
daily = pd.read_csv(COURSE_ROOT / 'assets/data/jena_daily.csv', parse_dates=['date']).sort_values('date')
assert daily['date'].is_unique
daily = daily.set_index('date').asfreq('D')
assert len(daily) > 365
print(daily.head(), daily.isna().sum())
'''),
S('## 2. 把预测问题变成表格\n一行的索引是今天。当前温度与过去 1—7 天可用，target 是明天。`shift(-1)` 只用于构造标签，绝不能放入特征。按时间前 60%/中 20%/后 20% 划分。', r'''
def make_weather_table(frame, window=7, target='temp_max'):
    table = frame[['temp_max', 'temp_min', 'temp_mean', 'pressure', 'humidity']].copy()
    for lag in range(1, window + 1):
        table[f'temp_lag_{lag}'] = frame['temp_max'].shift(lag)
    table['target'] = frame[target].shift(-1)
    table['forecast_date'] = frame.index + pd.Timedelta(days=1)
    return table.dropna()
table = make_weather_table(daily)
features = [c for c in table.columns if c not in ['target', 'forecast_date']]
a, b = int(len(table)*0.6), int(len(table)*0.8)
train, validation, test = table.iloc[:a], table.iloc[a:b], table.iloc[b:]
assert train['forecast_date'].max() < validation['forecast_date'].min()
assert validation['forecast_date'].max() < test['forecast_date'].min()
'''),
S('## 3. 只用验证集选择正则化强度\n标准化必须在训练集内拟合，所以与 Ridge 放入 Pipeline。验证选择完成后，用训练加验证重新拟合一次；测试段只用于最后报告。比较不同模型时不得更换测试日期。', r'''
validation_scores = {}
for alpha in [0.1, 1.0, 10.0, 100.0]:
    candidate = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    candidate.fit(train[features], train['target'])
    validation_scores[alpha] = float(mean_absolute_error(validation['target'], candidate.predict(validation[features])))
best_alpha = min(validation_scores, key=validation_scores.get)
model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
development = table.iloc[:b]
model.fit(development[features], development['target'])
prediction = model.predict(test[features])
baseline = test['temp_max'].to_numpy()
'''),
S('## 4. 评价与失败日期\nMAE 是平均绝对误差，单位 °C。先报告整体再检查误差最大的日期；不能删除难预测日来美化分数。图上不完整窗口保留为空缺，不能用连线暗示已补齐观测。若线性模型未胜出，也如实保留结论。', r'''
results = pd.DataFrame({'date': test['forecast_date'].to_numpy(), 'actual': test['target'].to_numpy(), 'persistence': baseline, 'ridge': prediction})
results['absolute_error'] = abs(results['actual'] - results['ridge'])
metrics = {'rows': len(table), 'train': len(train), 'validation': len(validation), 'test': len(test), 'best_alpha': best_alpha, 'validation_mae': validation_scores, 'persistence_mae': float(mean_absolute_error(test['target'], baseline)), 'ridge_mae': float(mean_absolute_error(test['target'], prediction))}
results.to_csv(OUT_DIR / 'predictions.csv', index=False)
results.nlargest(10, 'absolute_error').to_csv(OUT_DIR / 'largest_errors.csv', index=False)
(OUT_DIR / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
results.set_index('date')[['actual', 'persistence', 'ridge']].tail(90).asfreq('D').plot()
plt.ylabel('最高温度 / °C'); plt.tight_layout(); plt.savefig(OUT_DIR / 'forecast.png'); plt.close(); print(metrics)
''')], [
E('补全：手算基线 MAE', '用 NumPy 计算持续性基线 MAE，与指标文件对应值核对。', r'''
manual_mae = np.mean(np.abs(test['target'].to_numpy() - test['temp_max'].to_numpy()))
assert np.isclose(manual_mae, metrics['persistence_mae'])
print(manual_mae)
'''),
E('修改：历史窗口改成 3 天', '重建 3 天特征，只在原训练、验证日期比较；不要使用测试误差选窗口。', r'''
short = make_weather_table(daily, window=3)
short_features = [c for c in short.columns if c not in ['target','forecast_date']]
short_train, short_val = short.loc[train.index], short.loc[validation.index]
short_model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
short_model.fit(short_train[short_features], short_train['target'])
score = mean_absolute_error(short_val['target'], short_model.predict(short_val[short_features]))
assert np.isfinite(score) and short_train.index.max() < short_val.index.min()
print('3 天窗口验证 MAE', score)
'''),
E('迁移：预测最低温', '把标签改为明天最低温，基线也改为今天最低温；用相同日期验证。', r'''
low = make_weather_table(daily, target='temp_min')
low_train, low_val = low.loc[train.index], low.loc[validation.index]
low_model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
low_model.fit(low_train[features], low_train['target'])
low_mae = mean_absolute_error(low_val['target'], low_model.predict(low_val[features]))
low_base = mean_absolute_error(low_val['target'], low_val['temp_min'])
assert np.isfinite(low_mae) and np.isfinite(low_base)
print({'最低温模型MAE': low_mae, '最低温基线MAE': low_base})
''')], [('TensorFlow Jena 时间序列教程','https://www.tensorflow.org/tutorials/structured_data/time_series'),('scikit-learn 滞后特征预测',SK+'auto_examples/applications/plot_time_series_lagged_features.html')],
prereqs='B01—B05；学习 shift、训练/验证/测试、Pipeline、拟合与预测。',
pitfalls='这是单站历史统计预测，不是专业数值天气预报。相邻日特征可以重叠，但每次预测只能使用当时已观测的信息；未使用未来天气特征。',
data_note='真实 Jena 气象观测的日聚合数据，存放 assets/data/jena_daily.csv。date 为观测日，temp_* 单位 °C、pressure 单位 hPa、humidity 为相对湿度百分比。预聚合与原始来源见根目录资产说明。')

make('B', 7, '混合特征房价预测', '数字与类别混在一起，怎样预测房价？',
     '在明确标注的模拟房屋数据上比较线性回归与梯度提升。理解不同列的预处理、未见类别和验证选型，输出一份可复算的测试预测表。', [
S('## 1. 生成有可解释机制的房屋数据\n价格来自面积、房龄、区域及非线性面积效应，叠加噪声。真实房价不能由这个教学公式推断；我们使用它观察模型能否学到已知结构。缺失面积在生成价格后注入。', r'''
import numpy as np, pandas as pd, json
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
rng = np.random.default_rng(7); n = 800
houses = pd.DataFrame({'area': rng.uniform(35, 180, n), 'age': rng.integers(0, 45, n), 'district': rng.choice(['north','center','south'], n)})
prices = 80 + houses['area']*2.2 - houses['age']*1.2 + houses['district'].map({'north':20,'center':90,'south':0}) + 0.018*(houses['area']-100)**2 + rng.normal(0,12,n)
houses.loc[rng.choice(n, 35, replace=False), 'area'] = np.nan
houses.assign(price=prices).to_csv(DATA_DIR / 'simulated_houses.csv', index=False)
X_dev, X_test, y_dev, y_test = train_test_split(houses, prices, test_size=0.2, random_state=7)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25, random_state=7)
'''),
S('## 2. 让每种列经过适当处理\n数值缺失用训练中位数填充后标准化；类别变成独热列。`handle_unknown="ignore"` 让新区域不报错，但并不意味着模型已理解它。每次构造全新预处理器，避免两个模型共享可变状态。', r'''
def preprocessing():
    return ColumnTransformer([
        ('numeric', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), ['area','age']),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['district'])])
def make_models():
    return {'ridge': make_pipeline(preprocessing(), Ridge(alpha=1)),
            'boosting': make_pipeline(preprocessing(), HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15, random_state=7))}
models = make_models()
'''),
S('## 3. 先验证选型，再锁定测试\n同一行房屋用于公平比较；以验证 MAE 选推荐方案。两种方法均报告测试表现，但不能看到测试结果后重新选超参数。目标是学习有证据的选型，不保证复杂模型获胜。', r'''
val_scores = {}
for name, estimator in models.items():
    estimator.fit(X_train, y_train)
    val_scores[name] = float(mean_absolute_error(y_val, estimator.predict(X_val)))
selected = min(val_scores, key=val_scores.get)
predictions = pd.DataFrame({'actual': y_test.to_numpy()}, index=X_test.index)
for name, estimator in models.items():
    estimator.fit(X_dev, y_dev)
    predictions[name] = estimator.predict(X_test)
assert np.isfinite(predictions.to_numpy()).all()
'''),
S('## 4. 看误差，而不只看总分\n保存预测与原始特征，检查大面积、新区域是否更容易出错。金额是任意教学单位；图的对角线表示完美预测。', r'''
metrics = {'selected_by_validation': selected, 'validation_mae': val_scores, 'test_mae': {name: float(mean_absolute_error(y_test, predictions[name])) for name in models}}
X_test.join(predictions).to_csv(OUT_DIR / 'predictions.csv', index=False)
(OUT_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
plt.scatter(y_test, predictions[selected], s=15, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='grey')
plt.xlabel('真实价格 / 教学单位'); plt.ylabel('预测价格'); plt.tight_layout(); plt.savefig(OUT_DIR / 'price_predictions.png'); plt.close(); print(metrics)
''')], [
E('补全：独立复算误差', '对验证选中的模型重算测试 MAE，确认与 metrics 一致。', r'''
manual = float(np.abs(y_test.to_numpy() - predictions[selected].to_numpy()).mean())
assert np.isclose(manual, metrics['test_mae'][selected])
print(manual)
'''),
E('修改：预测未见区域', '创建一套 area 缺失、district=east 的房屋，验证流水线能返回有限值，并解释未知区域的不确定性。', r'''
new_house = pd.DataFrame({'area':[np.nan], 'age':[10], 'district':['east']})
value = models[selected].predict(new_house)
assert value.shape == (1,) and np.isfinite(value).all()
print('未见区域预测，仅作机制演示', value)
'''),
E('迁移：房租预测', '构造房租为价格的 0.004 倍加小噪声，在同一训练/验证索引上训练；不要混用房价指标单位。', r'''
rent = prices * 0.004 + np.random.default_rng(71).normal(0,0.05,n)
rent_model = make_models()['ridge']
rent_model.fit(houses.loc[X_train.index], rent.loc[X_train.index])
rent_mae = mean_absolute_error(rent.loc[X_val.index], rent_model.predict(houses.loc[X_val.index]))
assert np.isfinite(rent_mae) and rent_mae >= 0
print('房租验证 MAE / 教学单位', rent_mae)
''')], [('混合类型列转换器',SK+'auto_examples/compose/plot_column_transformer_mixed_types.html'),('梯度提升官方说明',SK+'modules/ensemble.html#histogram-based-gradient-boosting')],
prereqs='B06；学习列选择、独热编码、列转换器与验证选型。', pitfalls='模拟结构不代表真实市场。所有预处理在训练折内拟合；不能在全数据算中位数后再切分。')

make('B', 8, '识别垃圾短信', '怎样识别垃圾短信并分析误报？',
     '使用本地真实 SMS Spam Collection 数据，去除重复文本，再比较多数类基线与 TF-IDF+朴素贝叶斯。垃圾短信召回和正常短信误报分别评价。', [
S('## 1. 读入真实文本并处理重复\n同一短信的复制版本出现在训练和测试会虚高分数。先按原文检查冲突标签，再去重。只做保守空白清理，保留金额、链接等有用线索。', r'''
import pandas as pd, numpy as np, json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
messages = pd.read_csv(COURSE_ROOT / 'assets/data/sms.tsv', sep='\t')
messages['text'] = messages['text'].fillna('').str.replace(r'\s+', ' ', regex=True).str.strip()
assert messages['label'].isin(['ham','spam']).all()
conflicts = messages.groupby('text')['label'].nunique()
messages = messages.loc[~messages['text'].isin(conflicts[conflicts > 1].index)]
messages = messages.loc[messages['text'].ne('')].drop_duplicates('text').reset_index(drop=True)
y = (messages['label'] == 'spam').astype(int)
train_idx, test_idx = train_test_split(np.arange(len(messages)), test_size=0.25, stratify=y, random_state=8)
assert set(messages.loc[train_idx,'text']).isdisjoint(set(messages.loc[test_idx,'text']))
'''),
S('## 2. 文本转换只学习训练集词表\nTF-IDF 的每一列代表词，值表示词在某短信中的相对重要程度。Pipeline 保证测试文本不能影响词表与词频；稀疏矩阵无需展开成巨大的密集表。', r'''
model = make_pipeline(TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=2), MultinomialNB(alpha=0.5))
model.fit(messages.loc[train_idx,'text'], y.loc[train_idx])
print('词表大小', len(model.named_steps['tfidfvectorizer'].vocabulary_))
prediction = model.predict(messages.loc[test_idx,'text'])
probability = model.predict_proba(messages.loc[test_idx,'text'])[:,1]
'''),
S('## 3. 指标需要围绕误报与漏报\n“全部正常”可能准确率很高，但垃圾召回为零。混淆矩阵的行是真实类别、列是预测类别；precision 回答拦截短信中多少真是垃圾，recall 回答垃圾短信中拦住多少。', r'''
truth = y.loc[test_idx].to_numpy()
def classification_scores(truth, pred):
    return {'accuracy': float(np.mean(truth == pred)), 'precision': float(precision_score(truth,pred,zero_division=0)), 'recall': float(recall_score(truth,pred,zero_division=0)), 'f1': float(f1_score(truth,pred,zero_division=0))}
metrics = {'rows_after_deduplication': len(messages), 'baseline': classification_scores(truth,np.zeros_like(truth)), 'tfidf_nb': classification_scores(truth,prediction)}
matrix = confusion_matrix(truth, prediction, labels=[0,1])
'''),
S('## 4. 保存错误样例与矩阵\n错误列表用于理解，而不是回看测试集后不断修改模型。若要调阈值或平滑参数，应另留验证集。数据含真实短信文本，只在本地教学使用。', r'''
result = messages.loc[test_idx].copy(); result['prediction'] = prediction; result['spam_probability'] = probability
result.to_csv(OUT_DIR / 'predictions.csv', index=False, encoding='utf-8-sig')
result.loc[truth != prediction].to_csv(OUT_DIR / 'errors.csv', index=False, encoding='utf-8-sig')
(OUT_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
plt.imshow(matrix, cmap='Blues'); plt.xticks([0,1],['正常','垃圾']); plt.yticks([0,1],['正常','垃圾'])
for i in range(2):
    for j in range(2): plt.text(j,i,str(matrix[i,j]),ha='center',va='center')
plt.xlabel('预测'); plt.ylabel('真实'); plt.tight_layout(); plt.savefig(OUT_DIR / 'confusion.png'); plt.close(); print(metrics)
''')], [
E('补全：核算召回率', '从混淆矩阵提取 TP 与 FN，复算垃圾召回率。', r'''
tn, fp, fn, tp = matrix.ravel()
assert np.isclose(tp/(tp+fn), metrics['tfidf_nb']['recall'])
print('漏掉垃圾短信数', fn)
'''),
E('修改：查看新短信', '输入两条自写短信，输出预测概率；不要把这两条方便样例当成性能评测。', r'''
examples = ['Meeting at the library at five, see you there.', 'Congratulations! Claim your cash prize now by calling this number.']
scores = model.predict_proba(examples)[:,1]
assert scores.shape == (2,) and ((scores >= 0) & (scores <= 1)).all()
print(list(zip(examples,scores)))
'''),
E('迁移：用训练子集比较纯词模型', '从原训练集再切开发验证集，比较仅 unigram 的模型；不触碰原测试集做选择。', r'''
subtrain, subval = train_test_split(train_idx, test_size=0.2, stratify=y.loc[train_idx], random_state=81)
word_model = make_pipeline(TfidfVectorizer(min_df=2),MultinomialNB(alpha=0.5))
word_model.fit(messages.loc[subtrain,'text'],y.loc[subtrain])
scores = classification_scores(y.loc[subval].to_numpy(),word_model.predict(messages.loc[subval,'text']))
assert set(subval).isdisjoint(set(test_idx))
print('训练内部验证结果', scores)
''')], [('UCI SMS Spam Collection','https://archive.ics.uci.edu/dataset/228/sms+spam+collection'),('scikit-learn 稀疏特征文本分类示例',SK+'auto_examples/text/plot_document_classification_20newsgroups.html')],
prereqs='B06—B07；稀疏表示、朴素贝叶斯与分类指标。', pitfalls='英文旧短信数据不能代表当前中文诈骗；真实部署前需要新数据评估与误报处理。此例未把概率作严格校准概率使用。',
data_note='真实 SMS Spam Collection 本地副本 assets/data/sms.tsv，label 为 ham/spam，text 为英文短信。原始来源、许可和下载记录见根目录资产说明。')

make('B', 9, '少数故障与阈值', '故障很少时，怎样避免模型只预测正常？',
     '在模拟设备数据上区分模型评分、概率校准与行动阈值。校准只用训练数据的交叉验证，阈值只用验证集，以错误成本选型后锁定测试。', [
S('## 1. 制造有少数故障的可检查数据\n随机生成 8 个传感器特征，故障约占 8%。真实设备可能存在时间与设备分组，此处是假设独立样本的教学实验。用分层划分保证各段都有故障。', r'''
import numpy as np, pandas as pd, json
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import confusion_matrix, precision_score, recall_score, brier_score_loss
X, y = make_classification(n_samples=1800, n_features=8, n_informative=5, weights=[0.92,0.08], class_sep=1.3, flip_y=0.01, random_state=9)
X_train, X_hold, y_train, y_hold = train_test_split(X,y,test_size=0.4,stratify=y,random_state=9)
X_val, X_test, y_val, y_test = train_test_split(X_hold,y_hold,test_size=0.5,stratify=y_hold,random_state=9)
pd.DataFrame(X,columns=[f'sensor_{i}' for i in range(8)]).assign(failure=y).to_csv(DATA_DIR/'simulated_devices.csv',index=False)
'''),
S('## 2. 训练与概率校准分开理解\n带类别权重的模型更关注少数类，但原始概率不应直接当真实故障率。用训练折内 sigmoid 校准得到候选概率，测试集始终不参与拟合。Brier 分数评价概率误差。', r'''
raw_model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced',max_iter=500,random_state=9))
raw_model.fit(X_train,y_train)
calibrated = CalibratedClassifierCV(make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced',max_iter=500,random_state=9)), method='sigmoid', cv=3)
calibrated.fit(X_train,y_train)
p_val = calibrated.predict_proba(X_val)[:,1]
p_test = calibrated.predict_proba(X_test)[:,1]
'''),
S('## 3. 阈值对应明确代价\n漏检一次成本假定 20，误报一次成本假定 1。扫描验证阈值，输出误报、漏报与总成本。阈值小通常更积极报警，但最终应看当前成本和容量条件。', r'''
def error_cost(truth, pred, missed=20, false_alarm=1):
    tn, fp, fn, tp = confusion_matrix(truth,pred,labels=[0,1]).ravel()
    return {'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'cost':float(missed*fn+false_alarm*fp)}
threshold_rows = []
for threshold in np.linspace(0.01,0.8,80):
    threshold_rows.append({'threshold':float(threshold), **error_cost(y_val,p_val>=threshold)})
threshold_table = pd.DataFrame(threshold_rows)
threshold = float(threshold_table.loc[threshold_table['cost'].idxmin(),'threshold'])
pred = p_test >= threshold
'''),
S('## 4. 在测试集报告代价与可靠性\n校准曲线每组的横轴是平均预测概率，纵轴是真实故障率；样本少时曲线会抖动。不能以准确率单独评判故障模型，也不能保证校准在分布变化后仍有效。', r'''
metrics = {'selected_threshold':threshold,'all_normal':error_cost(y_test,np.zeros_like(y_test)), 'threshold_0_5':error_cost(y_test,p_test>=0.5), 'selected':error_cost(y_test,pred), 'precision':float(precision_score(y_test,pred,zero_division=0)), 'recall':float(recall_score(y_test,pred,zero_division=0)), 'raw_brier':float(brier_score_loss(y_test,raw_model.predict_proba(X_test)[:,1])), 'calibrated_brier':float(brier_score_loss(y_test,p_test))}
threshold_table.to_csv(OUT_DIR/'threshold_validation.csv',index=False)
pd.DataFrame({'actual':y_test,'probability':p_test,'alert':pred}).to_csv(OUT_DIR/'predictions.csv',index=False)
(OUT_DIR/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
fraction, average = calibration_curve(y_test,p_test,n_bins=6,strategy='quantile')
plt.plot(average,fraction,'o-',label='校准模型'); plt.plot([0,1],[0,1],'--'); plt.xlabel('平均预测概率'); plt.ylabel('实际故障比例'); plt.legend(); plt.tight_layout(); plt.savefig(OUT_DIR/'calibration.png'); plt.close(); print(metrics)
''')], [
E('补全：误报漏报成本', '用 selected 中的 fp/fn 复算代价。', r'''
computed = metrics['selected']['fp'] + 20*metrics['selected']['fn']
assert computed == metrics['selected']['cost']
print(computed)
'''),
E('修改：漏报成本减半', '仅在验证集用漏报成本 10 重新选阈值，报告新阈值而不在测试集选。', r'''
new_rows = [(t,error_cost(y_val,p_val>=t,missed=10)['cost']) for t in np.linspace(0.01,0.8,80)]
new_threshold, cost = min(new_rows,key=lambda row:row[1])
assert 0 < new_threshold < 1
print('成本改变后的验证阈值',new_threshold,'成本',cost)
'''),
E('迁移：报警容量只有 15 台', '在测试概率上按风险排序最多报警 15 台；用真实标签事后核算，不把标签用于排序。', r'''
limited = np.zeros(len(p_test),dtype=bool)
limited[np.argsort(-p_test)[:15]] = True
assert limited.sum() == 15
print(error_cost(y_test,limited))
''')], [('概率校准官方指南',SK+'modules/calibration.html'),('阈值调优',SK+'modules/classification_threshold.html')],
prereqs='B08；混淆矩阵、概率、阈值与验证集。', pitfalls='模拟故障机制与成本均为教学假设。校准和阈值选择不共享测试集；现实中同设备多次记录必须分组或按时间切分。')

make('B', 10, '公平比较分类模型', '怎样在同一预算下比较分类模型？',
     '使用 sklearn 自带乳腺癌数值数据练习模型比较，仅作算法教学。逻辑回归与梯度提升各搜索 3 组参数、使用同样 3 折验证，再一次性评价留出测试集。', [
S('## 1. 先锁定独立测试集\n本例使用公开内置数据。正类明确设为恶性，指标用 balanced accuracy 避免类别比例影响。它不是临床工具；公开小数据表现不能用于诊断。', r'''
import numpy as np, pandas as pd, json, time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
dataset = load_breast_cancer(as_frame=True)
X = dataset.data; y = (dataset.target == 0).astype(int)
X_dev,X_test,y_dev,y_test = train_test_split(X,y,test_size=0.25,stratify=y,random_state=10)
folds = list(StratifiedKFold(n_splits=3,shuffle=True,random_state=10).split(X_dev,y_dev))
assert set(X_dev.index).isdisjoint(X_test.index)
'''),
S('## 2. 预算意味着候选数与验证次数都一致\n标准化放在 Pipeline 内部，每个折只学习本折训练数据。逻辑回归测试 3 个 C，树模型测试 3 个叶子数；每种方法都是 9 次折内拟合。相同搜索次数不等于相同耗时，所以还记录时间。', r'''
specs = {
 'logistic':(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=10)),{'logisticregression__C':[0.1,1,10]}),
 'boosting':(HistGradientBoostingClassifier(max_iter=70,random_state=10),{'max_leaf_nodes':[3,7,15]})}
searches, comparison = {}, []
for name,(estimator,grid) in specs.items():
    start = time.perf_counter()
    search = GridSearchCV(estimator,grid,cv=folds,scoring='balanced_accuracy',n_jobs=1,refit=True)
    search.fit(X_dev,y_dev)
    searches[name] = search
    pd.DataFrame(search.cv_results_).to_csv(OUT_DIR/f'{name}_cv.csv',index=False)
    comparison.append({'model':name,'cv_score':float(search.best_score_),'seconds':time.perf_counter()-start,'best_parameters':search.best_params_})
'''),
S('## 3. 用交叉验证选模型，测试只作结算\n选中的模型已在完整开发集上重拟合。测试结果可能改变模型排名，但本轮推荐仍由预先的 CV 规则决定，不能反复拿测试集挑赢家。', r'''
selected = max(comparison,key=lambda row:row['cv_score'])['model']
result = pd.DataFrame({'actual':y_test.to_numpy()},index=X_test.index)
for row in comparison:
    name = row['model']; result[name] = searches[name].predict(X_test)
    row['test_balanced_accuracy'] = float(balanced_accuracy_score(y_test,result[name]))
metrics = {'selected_before_test':selected,'folds':3,'candidates_per_model':3,'comparison':comparison}
assert all(len(search.cv_results_['params']) == 3 for search in searches.values())
'''),
S('## 4. 保存全部候选，而非只保存最佳分数\n交付参数表、运行时间和测试预测。图显示 CV 与测试的差异，不能从少量折推断严格泛化置信区间。若真实应用存在患者分组，须改分组验证。', r'''
result.to_csv(OUT_DIR/'test_predictions.csv',index=False)
(OUT_DIR/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
pd.DataFrame(comparison).set_index('model')[['cv_score','test_balanced_accuracy']].plot.bar(ylim=(0,1))
plt.ylabel('balanced accuracy'); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(OUT_DIR/'comparison.png'); plt.close(); print(metrics)
''')], [
E('补全：验证折没有交集', '每个折的训练和验证索引应互斥，而且合计覆盖开发集。', r'''
for train_ids,val_ids in folds:
    assert set(train_ids).isdisjoint(val_ids)
    assert len(train_ids)+len(val_ids) == len(X_dev)
print('3 折划分检查通过')
'''),
E('修改：查看被选方案的混淆矩阵', '在测试集按固定选型结果计算混淆矩阵，并核对样本量。', r'''
matrix = confusion_matrix(y_test,result[selected],labels=[0,1])
assert matrix.sum() == len(y_test)
print(matrix)
'''),
E('迁移：另一份多分类数据', '用 sklearn 内置 wine，独立留出 25%，仅比较固定配置的逻辑回归；保证预测类别合法。', r'''
from sklearn.datasets import load_wine
wine = load_wine()
wx,tx,wy,ty = train_test_split(wine.data,wine.target,test_size=0.25,stratify=wine.target,random_state=101)
transfer = make_pipeline(StandardScaler(),LogisticRegression(max_iter=500)).fit(wx,wy)
pred = transfer.predict(tx)
assert set(pred).issubset({0,1,2})
print('wine balanced accuracy',balanced_accuracy_score(ty,pred))
''')], [('GridSearchCV 官方说明',SK+'modules/grid_search.html'),('scikit-learn 内置数据说明',SK+'datasets/toy_dataset.html')],
prereqs='B07—B09；交叉验证、参数字典、折内拟合与计时。', pitfalls='内置医学数据只用于算法教学，不输出个体诊断。统一候选次数不保证严格统一计算量；耗时单独记录。',
data_note='使用 sklearn 包内自带的 Wisconsin Diagnostic Breast Cancer 数据，离线读取；练习使用 wine 数据。数据出处及许可见 sklearn 官方数据说明。')
make('B', 11, '发现客户分群', '没有标签，怎样发现行为相近的客户群？',
     '模拟客户的购买频次、客单价与最近购买间隔，标准化后聚类。用轮廓系数辅助选择，再回原单位解释各群，不把聚类编号当真实人格标签。', [
S('## 1. 定义行为特征而非臆测客户身份\n三个特征单位不同：次数、金额、天数。生成机制包含三种购买习惯，但拟合阶段不提供潜在类别。聚类的输出只是相似性分组，不是因果解释。', r'''
import numpy as np,pandas as pd,json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
rng = np.random.default_rng(11)
centers = np.array([[2,60,40],[12,45,5],[4,260,18]])
parts = [rng.normal(center,[1,22,4],size=(90,3)) for center in centers]
customers = pd.DataFrame(np.maximum(np.vstack(parts),[0.1,5,0]),columns=['visits','basket','days_since'])
customers.to_csv(DATA_DIR/'simulated_customers.csv',index=False)
scaler = StandardScaler(); X_scaled = scaler.fit_transform(customers)
'''),
S('## 2. 尺度改变距离\nK-means 最小化群内平方距离；不标准化时金额可能压过次数。固定 n_init 与随机种子，避免把初始化偶然结果误认成稳定结构。先试 k=2—5，而不是直接写死三群。', r'''
cluster_models, scores = {}, []
for k in [2,3,4,5]:
    estimator = KMeans(n_clusters=k,n_init=10,random_state=11).fit(X_scaled)
    cluster_models[k] = estimator
    scores.append({'k':k,'silhouette':float(silhouette_score(X_scaled,estimator.labels_)),'inertia':float(estimator.inertia_)})
score_table = pd.DataFrame(scores)
selected_k = int(score_table.loc[score_table['silhouette'].idxmax(),'k'])
model = cluster_models[selected_k]
'''),
S('## 3. 返回原单位解释\n群 0/1/2 没有天然大小含义。按原始金额、间隔等描述，检查每群样本量；很小群可能是异常或样本不足，不能急着设计客户政策。', r'''
customers['cluster'] = model.labels_
profiles = customers.groupby('cluster')[['visits','basket','days_since']].mean()
profiles['size'] = customers.groupby('cluster').size()
assert profiles['size'].sum() == len(customers)
print(profiles)
'''),
S('## 4. 保存选择依据与分群\n二维散点仅显示三维空间的一部分。轮廓系数是内部指标，不能证明分群有商业价值；后续需要稳定性与实际行动收益验证。', r'''
customers.to_csv(OUT_DIR/'assignments.csv',index=False)
profiles.to_csv(OUT_DIR/'profiles.csv'); score_table.to_csv(OUT_DIR/'k_comparison.csv',index=False)
metrics = {'selected_k':selected_k,'silhouette':float(score_table.set_index('k').loc[selected_k,'silhouette']),'cluster_sizes':profiles['size'].to_dict()}
(OUT_DIR/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
plt.scatter(customers['visits'],customers['basket'],c=customers['cluster'],cmap='tab10',s=20)
plt.xlabel('购买次数'); plt.ylabel('客单价 / 教学单位'); plt.tight_layout(); plt.savefig(OUT_DIR/'clusters.png'); plt.close()
''')], [
E('补全：找到最大群', '从 profiles 中找人数最多的群，核对输出人数。', r'''
largest = profiles['size'].idxmax()
assert int((customers['cluster']==largest).sum()) == int(profiles.loc[largest,'size'])
print(profiles.loc[largest])
'''),
E('修改：不标准化会怎样', '对原单位特征运行相同 k，比较两种标签的调整兰德指数，不能直接逐个比较群编号。', r'''
from sklearn.metrics import adjusted_rand_score
unscaled = KMeans(n_clusters=selected_k,n_init=10,random_state=11).fit(customers[['visits','basket','days_since']])
agreement = adjusted_rand_score(model.labels_,unscaled.labels_)
assert -1 <= agreement <= 1
print('两种分群的一致性 ARI',agreement)
'''),
E('迁移：给新客户分组', '用已拟合标准化器处理新客户，不能对一个新客户重新 fit。', r'''
new = pd.DataFrame({'visits':[3,11],'basket':[250,50],'days_since':[15,4]})
labels = model.predict(scaler.transform(new))
assert labels.shape == (2,) and set(labels).issubset(set(model.labels_))
print(new.assign(cluster=labels))
''')], [('K-means 官方说明',SK+'modules/clustering.html#k-means'),('轮廓系数选择聚类数',SK+'auto_examples/cluster/plot_kmeans_silhouette_analysis.html')],
prereqs='B07；均值、距离、标准化、无标签分析。', pitfalls='潜在组是模拟机制。内部聚类指标不保证真实世界群体有效，群编号不是等级。')

make('B', 12, '无标签异常记录', '缺少故障标签时怎样发现异常传感器记录？',
     '用多数正常的历史数据建立分布，再比较单变量标准分数与 Isolation Forest。异常分数只产生复核队列，不能直接等同故障诊断。', [
S('## 1. 历史参考与未来复核集分开\n训练历史不使用故障标签，测试标签只在教学模拟中用于事后评分。三个传感器分别代表温度、振动、电流，异常机制包括高振动与高电流。', r'''
import numpy as np,pandas as pd,json
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score,recall_score
rng = np.random.default_rng(12)
train = rng.normal([40,2,10],[3,0.3,1],size=(450,3))
normal = rng.normal([40,2,10],[3,0.3,1],size=(180,3))
abnormal = rng.normal([43,3.1,13],[4,0.4,1],size=(20,3))
test = np.vstack([normal,abnormal]); truth = np.r_[np.zeros(180,dtype=int),np.ones(20,dtype=int)]
order = rng.permutation(len(test)); test,truth = test[order],truth[order]
pd.DataFrame(train,columns=['temperature','vibration','current']).to_csv(DATA_DIR/'reference_history.csv',index=False)
'''),
S('## 2. 简单基线：最大绝对标准分数\n每列用历史均值和标准差标准化，任一列极端就报警。用历史分数的 95% 分位数设定候选复核比例，不拿测试标签调阈值。', r'''
mean,scale = train.mean(axis=0),train.std(axis=0)
z_train = np.max(np.abs((train-mean)/scale),axis=1)
z_test = np.max(np.abs((test-mean)/scale),axis=1)
z_threshold = float(np.quantile(z_train,0.95))
z_alert = z_test > z_threshold
'''),
S('## 3. Isolation Forest：以隔离难度产生分数\n随机切分较容易隔离的点可能异常。score_samples 越小越异常，所以取负号统一为高分更可疑。仍用历史分位数作阈值，模型并没有学会故障类别。', r'''
model = IsolationForest(n_estimators=100,random_state=12,n_jobs=1).fit(train)
train_scores = -model.score_samples(train)
test_scores = -model.score_samples(test)
threshold = float(np.quantile(train_scores,0.95))
alert = test_scores > threshold
assert len(test_scores) == len(test)
'''),
S('## 4. 输出排序后的复核队列\n评价报警数量、精确率和召回率，允许简单方法胜出。真实无标签场景需要人工抽查被报警与未报警记录，才知道是否漏掉重要异常。', r'''
def score_alert(pred):
    return {'alerts':int(pred.sum()),'precision':float(precision_score(truth,pred,zero_division=0)),'recall':float(recall_score(truth,pred,zero_division=0))}
metrics = {'zscore':score_alert(z_alert),'isolation_forest':score_alert(alert),'threshold':threshold}
result = pd.DataFrame(test,columns=['temperature','vibration','current']).assign(score=test_scores,alert=alert,simulated_truth=truth)
result.sort_values('score',ascending=False).to_csv(OUT_DIR/'review_queue.csv',index=False)
(OUT_DIR/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
plt.scatter(test[:,1],test[:,2],c=test_scores,cmap='magma'); plt.colorbar(label='异常分数'); plt.xlabel('振动'); plt.ylabel('电流'); plt.tight_layout(); plt.savefig(OUT_DIR/'anomalies.png'); plt.close(); print(metrics)
''')], [
E('补全：复核队列排序', '找到分数最高的前 5 条，检查分数为非递增。', r'''
top = result.nlargest(5,'score')
assert len(top)==5 and np.all(np.diff(top['score'])<=0)
print(top)
'''),
E('修改：复核容量减半', '将历史阈值设为 97.5% 分位数，验证报警数量不多于原方案。', r'''
strict = test_scores > np.quantile(train_scores,0.975)
assert strict.sum() <= alert.sum()
print(score_alert(strict))
'''),
E('迁移：整体温度升高不等于设备故障', '把测试温度整体提高 8°C，保持其他特征不变；比较报警数，说明环境变化需要复核分布。', r'''
shifted = test.copy(); shifted[:,0] += 8
shifted_alert = -model.score_samples(shifted) > threshold
assert shifted_alert.shape == alert.shape
print({'原报警数':int(alert.sum()),'升温后报警数':int(shifted_alert.sum())})
''')], [('异常检测官方指南',SK+'modules/outlier_detection.html')],
prereqs='B09、B11；异常分数、阈值与人工复核。', pitfalls='历史正常性是教学假设。分布变化、罕见但正常行为也会触发报警；无监督异常不等于已识别故障。')

make('B', 13, '压缩多维特征', '特征很多时怎样压缩并观察结构？',
     '使用 sklearn 自带葡萄酒数值特征，训练集拟合标准化与 PCA，独立测试集检查重构误差。图中颜色仅用于事后观察，不参与无监督拟合。', [
S('## 1. 理解矩阵形状\n一行是一个样本，一列是一个测量值。不同单位先在训练集标准化。学习 shape、axis 与二维索引，避免把样本轴和特征轴颠倒。', r'''
import numpy as np,pandas as pd,json
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
dataset = load_wine(as_frame=True)
train_idx,test_idx = train_test_split(np.arange(len(dataset.data)),test_size=0.25,random_state=13,stratify=dataset.target)
scaler = StandardScaler().fit(dataset.data.iloc[train_idx])
train = scaler.transform(dataset.data.iloc[train_idx]); test = scaler.transform(dataset.data.iloc[test_idx])
assert train.shape[1] == 13
print('训练矩阵',train.shape,'测试矩阵',test.shape)
'''),
S('## 2. 用训练集方差决定保留维数\nPCA 寻找线性变化最大的方向，不寻找最有预测价值的方向。保留累计至少 80% 的训练方差；这个比例是预先设定的教学目标，不是用测试效果挑出来的。', r'''
full = PCA().fit(train)
cumulative = np.cumsum(full.explained_variance_ratio_)
k = int(np.searchsorted(cumulative,0.8)+1)
pca = PCA(n_components=k).fit(train)
compressed = pca.transform(test)
assert compressed.shape == (len(test),k)
'''),
S('## 3. 逆变换检查丢失的信息\ninverse_transform 返回标准化空间的近似原始点。与始终预测训练均值的基线比较重构误差；低重构误差不保证分类效果。先还原再查看一个样本的特征误差。', r'''
reconstructed = pca.inverse_transform(compressed)
baseline = np.tile(train.mean(axis=0),(len(test),1))
reconstruction_mse = float(np.mean((test-reconstructed)**2))
baseline_mse = float(np.mean((test-baseline)**2))
comparison = pd.DataFrame({'feature':dataset.feature_names,'standardized_mse':np.mean((test-reconstructed)**2,axis=0)})
assert reconstruction_mse <= baseline_mse + 1e-10
'''),
S('## 4. 保存方差曲线与二维投影\n二维图只供观察，不能把重叠样本强行解释成明显分群。将原始标签仅作为颜色加入，表明这一步是诊断展示而非训练过程。', r'''
metrics = {'n_features':13,'retained_components':k,'training_variance_retained':float(cumulative[k-1]),'test_reconstruction_mse':reconstruction_mse,'mean_baseline_mse':baseline_mse}
comparison.to_csv(OUT_DIR/'feature_errors.csv',index=False)
pd.DataFrame(compressed).to_csv(OUT_DIR/'compressed_test.csv',index=False)
(OUT_DIR/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
fig,axes = plt.subplots(1,2,figsize=(10,4)); axes[0].plot(np.arange(1,14),cumulative,'o-'); axes[0].axhline(0.8,ls='--'); axes[0].set(xlabel='维数',ylabel='累计解释方差')
projection = full.transform(test); axes[1].scatter(projection[:,0],projection[:,1],c=dataset.target.iloc[test_idx],cmap='tab10'); axes[1].set(xlabel='PC1',ylabel='PC2')
fig.tight_layout(); fig.savefig(OUT_DIR/'pca.png'); plt.close(fig); print(metrics)
''')], [
E('补全：逐特征重构误差', '平均各特征误差，核对它与整体测试重构 MSE 相同。', r'''
assert np.isclose(comparison['standardized_mse'].mean(),reconstruction_mse)
print(comparison.sort_values('standardized_mse',ascending=False).head())
'''),
E('修改：只保留 2 维', '训练集重新拟合 2 维 PCA，检查测试误差不能优于保留更多同一 PCA 方向的结果。', r'''
two = PCA(n_components=2).fit(train)
two_mse = float(np.mean((test-two.inverse_transform(two.transform(test)))**2))
assert two_mse >= reconstruction_mse - 1e-9
print('2 维测试重构 MSE',two_mse)
'''),
E('迁移：新观测恢复到原单位', '对一个新观测执行标准化→降维→重构→还原单位，验证形状。', r'''
new = dataset.data.iloc[[test_idx[0]]]
recovered = scaler.inverse_transform(pca.inverse_transform(pca.transform(scaler.transform(new))))
assert recovered.shape == (1,13) and np.isfinite(recovered).all()
print(pd.DataFrame({'原值':new.iloc[0],'重构值':recovered[0]}))
''')], [('PCA 官方指南',SK+'modules/decomposition.html#pca'),('wine 内置数据',SK+'datasets/toy_dataset.html#wine-recognition-dataset')],
prereqs='B07、B11；NumPy 广播、矩阵形状、方差、线性投影。', pitfalls='PCA 方差大不等于预测有用；标准化与 PCA 都只拟合训练集。', data_note='真实 sklearn wine 内置数值数据，离线加载。类别标签仅用于分层划分和事后可视化。')

make('B', 14, '少量评分推荐电影', '用户只评价过少量电影，怎样推荐？',
     '在模拟评分中按用户留出后续评价，构造用户—电影矩阵并做物品协同过滤。与平均评分推荐比较，检查已看过过滤、冷启动与评测偏差。', [
S('## 1. 评分是部分观察，不是完整偏好\n模拟 60 位用户对 15 部电影的部分评分，三类口味构成隐含结构。每人 8 次评分，最后两次留出。未评分不是零分；0 只能用作矩阵缺失占位。', r'''
import numpy as np,pandas as pd,json
from sklearn.metrics.pairwise import cosine_similarity
rng = np.random.default_rng(14); n_users,n_items = 60,15
records=[]
for user in range(n_users):
    taste = user%3
    chosen = rng.choice(n_items,8,replace=False)
    for event,item in enumerate(chosen):
        rating = np.clip(2.8 + 1.8*(item%3==taste) + rng.normal(0,0.45),1,5)
        records.append((user,int(item),event,float(rating)))
ratings = pd.DataFrame(records,columns=['user','item','event','rating'])
train = ratings.loc[ratings['event']<6].copy(); test = ratings.loc[ratings['event']>=6].copy()
matrix = train.pivot(index='user',columns='item',values='rating').reindex(index=range(n_users),columns=range(n_items))
ratings.to_csv(DATA_DIR/'simulated_ratings.csv',index=False)
'''),
S('## 2. 从训练评分学习电影相似度\n先减去每位用户的均分，再把缺失填 0。对转置后的电影向量计算余弦相似度，并去掉自相似；只保留正相关使加权解释简单。不能使用留出评分算相似度。', r'''
user_means = matrix.mean(axis=1)
residual = matrix.sub(user_means,axis=0).fillna(0)
similarity = np.maximum(cosine_similarity(residual.T),0)
np.fill_diagonal(similarity,0)
popularity = matrix.mean(axis=0).fillna(train['rating'].mean()).to_numpy()
'''),
S('## 3. 为未看过的电影排序\n候选电影分数来自已评分电影的相似度加权残差；无相似邻居时回退均分。已看过项目设为负无穷。新用户没有协同信号，只能先用全局候选。', r'''
def recommend(user, k=3, collaborative=True):
    if user not in matrix.index:
        return np.argsort(-popularity)[:k]
    seen = matrix.loc[user].notna().to_numpy()
    if collaborative:
        numerator = similarity @ residual.loc[user].to_numpy()
        denominator = similarity @ seen.astype(float)
        scores = float(user_means.loc[user]) + np.divide(numerator,denominator,out=np.zeros(n_items),where=denominator>0)
    else:
        scores = popularity.copy()
    scores[seen] = -np.inf
    return np.argsort(-scores)[:k]
assert not set(recommend(0)).intersection(set(train.loc[train['user']==0,'item']))
'''),
S('## 4. 留出正反馈命中率\n仅对至少有一条留出评分≥4 的用户计算 Hit@3。未评分电影不应被当成确定负例；这是一份受观察偏差影响的离线评测，不能直接等同点击率提升。', r'''
evaluation=[]
for user,part in test.groupby('user'):
    positives=set(part.loc[part['rating']>=4,'item'])
    if not positives: continue
    cf=recommend(user); popular=recommend(user,collaborative=False)
    evaluation.append({'user':int(user),'cf_hit':int(bool(set(cf)&positives)),'popular_hit':int(bool(set(popular)&positives)),'recommendations':','.join(map(str,cf))})
evaluation=pd.DataFrame(evaluation)
metrics={'eligible_users':len(evaluation),'cf_hit_at_3':float(evaluation['cf_hit'].mean()),'popular_hit_at_3':float(evaluation['popular_hit'].mean())}
evaluation.to_csv(OUT_DIR/'recommendations.csv',index=False)
(OUT_DIR/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
plt.imshow(similarity,cmap='Blues'); plt.colorbar(label='电影相似度'); plt.xlabel('电影编号'); plt.ylabel('电影编号'); plt.tight_layout(); plt.savefig(OUT_DIR/'item_similarity.png'); plt.close(); print(metrics)
''')], [
E('补全：确认无训练测试评分交叉', '把用户、电影组成二元组，检查两份评分集合没有交集。', r'''
train_pairs=set(zip(train['user'],train['item']))
test_pairs=set(zip(test['user'],test['item']))
assert train_pairs.isdisjoint(test_pairs)
print(len(train_pairs),len(test_pairs))
'''),
E('修改：推荐 5 部', '对用户 4 产生 5 个不同且未看过的推荐。', r'''
items=recommend(4,k=5)
assert len(set(items))==5
assert set(items).isdisjoint(train.loc[train['user']==4,'item'])
print(items)
'''),
E('迁移：没有历史的新用户', '调用新用户 999 的回退方案，并核对与全局均分排序一致。', r'''
items=recommend(999,k=3)
assert np.array_equal(items,np.argsort(-popularity)[:3])
print('冷启动候选',items)
''')], [('Amazon 原始 item-to-item 协同过滤论文：本例仅借鉴物品相似思路，不复刻其生产系统','https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf'),('scikit-learn 余弦相似度',SK+'modules/generated/sklearn.metrics.pairwise.cosine_similarity.html')],
prereqs='B11—B13；矩阵、余弦相似度、缺失占位、排序与离线评测。', pitfalls='模拟评分不是真实电影偏好。用户留出法没有模拟全平台统一时间，生产评估应进一步按真实事件时间回放；此处只演示防止同一用户—电影泄漏。')

make('B', 15, '预测一周用电', '怎样预测未来一周用电量并滚动评价？',
     '用标明机制的模拟每日用电量，训练同时输出未来 7 天的随机森林。分开区间校准窗口与最终回测窗口，比较同星期基线和直接多步预测。', [
S('## 1. 模拟日历与负荷机制\n日用电由周节律、季节、缓慢趋势及噪声组成。单位 kWh 是教学单位，数据不代表实际城市。多步预测时不能把未来真实用电填进滞后特征。', r'''
import numpy as np,pandas as pd,json
from sklearn.ensemble import RandomForestRegressor
rng=np.random.default_rng(15); dates=pd.date_range('2022-01-01',periods=740,freq='D'); t=np.arange(len(dates))
energy=120+18*np.sin(2*np.pi*t/365)+20*(dates.dayofweek.to_numpy()<5)+0.025*t+rng.normal(0,7,len(t))
series=pd.Series(energy,index=dates,name='kwh')
frame=pd.DataFrame({'weekday':dates.dayofweek,'annual_sin':np.sin(2*np.pi*t/365),'annual_cos':np.cos(2*np.pi*t/365)},index=dates)
for lag in range(14): frame[f'lag_{lag}']=series.shift(lag)
features=list(frame.columns)
for horizon in range(1,8): frame[f'y_{horizon}']=series.shift(-horizon)
targets=[f'y_{h}' for h in range(1,8)]
series.to_csv(DATA_DIR/'simulated_daily_energy.csv',index_label='date')
'''),
S('## 2. 每个预测起点只训练已经获得完整标签的行\n若今天索引为 origin，训练行的未来第 7 天必须不晚于今天。因此只取到 origin−7。森林一次输出 7 个数，直接多步预测避免递归把自己的误差再喂回去。', r'''
def forecast_at(origin):
    history=frame.iloc[:origin-6].dropna(subset=features+targets)
    assert history.index[-1]+pd.Timedelta(days=7)<=dates[origin]
    estimator=RandomForestRegressor(n_estimators=40,max_depth=8,min_samples_leaf=3,random_state=15,n_jobs=1)
    estimator.fit(history[features],history[targets])
    prediction=estimator.predict(frame.iloc[[origin]][features])[0]
    baseline=series.iloc[origin-6:origin+1].to_numpy()
    actual=series.iloc[origin+1:origin+8].to_numpy()
    return prediction,baseline,actual
calibration_origins=[390,420,450]
test_origins=[510,540,570,600,630,660,690,720]
'''),
S('## 3. 区间只用早期回测误差校准\n用早期窗口绝对误差的 90% 分位数扩展点预测。这是时序经验区间，没有直接套用独立同分布 conformal 保证。校准后固定宽度，在更晚窗口检查覆盖与误差。', r'''
calibration_errors=[]
for origin in calibration_origins:
    pred,base,actual=forecast_at(origin)
    calibration_errors.extend(abs(actual-pred))
width=float(np.quantile(calibration_errors,0.9,method='higher'))
rows=[]
for origin in test_origins:
    pred,base,actual=forecast_at(origin)
    for h in range(7):
        rows.append({'origin':dates[origin],'date':dates[origin+h+1],'horizon':h+1,'actual':actual[h],'forest':pred[h],'seasonal':base[h],'lower':pred[h]-width,'upper':pred[h]+width})
result=pd.DataFrame(rows)
'''),
S('## 4. 按预测距离检查，而非混成一个漂亮数字\n分别报告第 1—7 天误差，整体 MAE、区间覆盖和宽度。测试窗口从未用于调宽度；覆盖不足是需要报告的结果，不能看完测试再把区间加宽。', r'''
result['absolute_error']=abs(result['actual']-result['forest'])
coverage=float(((result['actual']>=result['lower'])&(result['actual']<=result['upper'])).mean())
metrics={'forecast_windows':len(test_origins),'forest_mae':float(result['absolute_error'].mean()),'seasonal_mae':float(abs(result['actual']-result['seasonal']).mean()),'empirical_coverage':coverage,'interval_width':2*width,'nominal_target':0.9,'iid_guarantee_claimed':False}
result.to_csv(OUT_DIR/'rolling_predictions.csv',index=False)
result.groupby('horizon')['absolute_error'].mean().to_csv(OUT_DIR/'horizon_mae.csv')
(OUT_DIR/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
last=result.tail(7); plt.plot(last['date'],last['actual'],'o-',label='实际'); plt.plot(last['date'],last['forest'],'o-',label='森林'); plt.plot(last['date'],last['seasonal'],'--',label='同星期基线'); plt.fill_between(last['date'],last['lower'],last['upper'],alpha=0.2)
plt.ylabel('每日用电 / kWh'); plt.xticks(rotation=25); plt.legend(); plt.tight_layout(); plt.savefig(OUT_DIR/'weekly_forecast.png'); plt.close(); print(metrics)
''')], [
E('补全：分预测距离 MAE', '计算各 horizon 的 MAE，并核对 7 个距离都有 8 个预测。', r'''
by_horizon=result.groupby('horizon')['absolute_error'].agg(['mean','count'])
assert len(by_horizon)==7 and by_horizon['count'].eq(8).all()
print(by_horizon)
'''),
E('修改：只改校准目标', '用原校准误差的 95% 分位数构造区间，核对宽度不缩小；这是预设目标对照，不按测试选宽度。', r'''
wide=float(np.quantile(calibration_errors,0.95,method='higher'))
assert wide>=width
new_coverage=np.mean(abs(result['actual']-result['forest'])<=wide)
print({'区间宽度':2*wide,'测试经验覆盖':float(new_coverage)})
'''),
E('迁移：第 731 天的新预测起点', '在 origin=730 生成未来 7 天，核对模型输出和基线形状，并计算这段已知真值的误差。', r'''
pred,base,actual=forecast_at(730)
assert pred.shape==base.shape==actual.shape==(7,)
print({'模型MAE':float(np.mean(abs(pred-actual))),'基线MAE':float(np.mean(abs(base-actual)))})
''')], [('scikit-learn 滞后特征时间序列案例',SK+'auto_examples/applications/plot_time_series_lagged_features.html'),('随机森林多输出回归',SK+'modules/ensemble.html#forest')],
prereqs='B06、B07、B10；多输出回归、滚动起点、经验预测区间。', pitfalls='数据是模拟用电，不能把模型分数当真实能源预测成绩。校准残差有时间相关性，这里的区间只是经验评估；预训练候选另见拓展。')
make('A', 11, '客流与人员配置', '预测到客流后，应该安排多少工作人员？',
     '把客流点预测转为排班人数，再比较风险缓冲与固定人数。使用独立模拟样本学习 split conformal 校准，分别报告预测误差、覆盖率和决策成本。', [
S('## 1. 定义目标与四份独立数据\n每行是一个独立活动时段，输入为小时、周末标志与活动强度；本例不把相邻时间当独立观测。训练拟合模型，校准构造区间，验证选择配置策略，测试结算成本。每人每小时可服务 20 人，工资 30，缺口每人次损失 8，均为假设。', r'''
import numpy as np,pandas as pd,json,math
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
rng=np.random.default_rng(111)
def generate_slots(n):
    x=pd.DataFrame({'hour':rng.integers(9,20,n),'weekend':rng.integers(0,2,n),'intensity':rng.uniform(0,1,n)})
    mean=35+45*np.exp(-((x['hour']-14)/3)**2)+20*x['weekend']+35*x['intensity']
    return x,rng.poisson(mean).astype(float)
X_train,y_train=generate_slots(500); X_cal,y_cal=generate_slots(200)
X_val,y_val=generate_slots(200); X_test,y_test=generate_slots(250)
capacity,wage,shortage_cost=20,30,8
model=HistGradientBoostingRegressor(max_iter=90,max_leaf_nodes=12,random_state=111).fit(X_train,y_train)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
linear_model=make_pipeline(StandardScaler(),Ridge(alpha=1)).fit(X_train,y_train)
'''),
S('## 2. 从点预测到校准区间\n把独立校准集绝对残差排序，取 ceil((n+1)×0.9) 对应顺序统计量。代码把有限样本分位数写出来，避免神秘 API。独立可交换样本下是边际覆盖保证，不保证每个小时、每个子群或每批有限测试数据恰好 90%；实测覆盖不足仍须如实报告，不能看测试后改区间。', r'''
alpha=0.1
calibration_residuals=np.abs(y_cal-model.predict(X_cal))
rank=min(len(calibration_residuals),math.ceil((len(calibration_residuals)+1)*(1-alpha)))
q=float(np.sort(calibration_residuals)[rank-1])
def staffing_predictions(x):
    point=np.maximum(model.predict(x),0)
    return {'fixed':np.full(len(x),math.ceil(y_train.mean()/capacity),dtype=int),
            'linear_mean':np.ceil(np.maximum(linear_model.predict(x),0)/capacity).astype(int),
            'mean':np.ceil(point/capacity).astype(int),
            'buffered':np.ceil((point+q)/capacity).astype(int)}
def evaluate_staff(staff,actual):
    gap=np.maximum(actual-staff*capacity,0)
    return {'cost':float(np.mean(staff*wage+gap*shortage_cost)),'mean_staff':float(np.mean(staff)),'shortage_rate':float(np.mean(gap>0)),'mean_shortage':float(gap.mean())}
'''),
S('## 3. 用验证成本选择策略\n覆盖率高不代表排班经济：多安排人有成本。对固定、线性预测、树模型预测、树模型加缓冲四种策略，在完全相同客流上比较成本。另列两种模型的 MAE 与同一人数转换规则下的成本，检查预测分数排序是否等于行动成本排序；排序可能一致，也可能不同。选择只使用验证集。', r'''
validation_scores={name:evaluate_staff(staff,y_val) for name,staff in staffing_predictions(X_val).items()}
selected=min(validation_scores,key=lambda name:validation_scores[name]['cost'])
test_staff=staffing_predictions(X_test)
test_scores={name:evaluate_staff(staff,y_test) for name,staff in test_staff.items()}
point=model.predict(X_test); lower=np.maximum(point-q,0); upper=point+q
coverage=float(np.mean((y_test>=lower)&(y_test<=upper)))
model_comparison=[]
for name,estimator,policy in [('ridge',linear_model,'linear_mean'),('hist_gradient_boosting',model,'mean')]:
    model_comparison.append({'model':name,'validation_mae':float(mean_absolute_error(y_val,estimator.predict(X_val))),
                            'test_mae':float(mean_absolute_error(y_test,estimator.predict(X_test))),
                            'validation_decision_cost':validation_scores[policy]['cost'],'test_decision_cost':test_scores[policy]['cost']})
pd.DataFrame(model_comparison).to_csv(OUT_DIR/'prediction_vs_decision.csv',index=False)
'''),
S('## 4. 给出有条件的人员建议\n交付每个时段人数、缺口和政策比较，并把建议写成可复核文本。实际运营还需轮班连续性与技能约束；本例只解决时段人数，不假装已完成完整排班。', r'''
results=X_test.assign(actual=y_test,point=point,lower=lower,upper=upper)
for name,staff in test_staff.items(): results[name+'_staff']=staff
results.to_csv(OUT_DIR/'staffing.csv',index=False)
metrics={'selected_on_validation':selected,'validation':validation_scores,'test':test_scores,'model_comparison':model_comparison,'test_mae':float(mean_absolute_error(y_test,point)),'interval_coverage':coverage,'mean_interval_width':float(np.mean(upper-lower)),'nominal_coverage':0.9}
(OUT_DIR/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
advice=f'验证集选择 {selected} 策略。独立测试平均成本 {test_scores[selected]["cost"]:.2f}，缺口时段比例 {test_scores[selected]["shortage_rate"]:.1%}；固定人数成本 {test_scores["fixed"]["cost"]:.2f}。经验区间覆盖 {coverage:.1%}。建议仅适用于每人容量 {capacity}、工资 {wage}、缺口代价 {shortage_cost} 的假设；容量、客流分布或工资改变后重新评估。'
if coverage < 0.9:
    advice += ' 本次有限测试集覆盖低于 90% 名义目标，须继续收集独立校准与验证数据并检查子群覆盖；不能把边际保证解读为当前每批数据必达标。'
(OUT_DIR/'决策建议.md').write_text('# 人员配置建议\n\n'+advice,encoding='utf-8')
pd.DataFrame(test_scores).T['cost'].plot.bar(); plt.ylabel('每时段平均总成本'); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(OUT_DIR/'staffing_cost.png'); plt.close(); print(metrics)
''')], [
E('补全：容量与缺口', '手工计算选中方案的平均缺口，并与输出指标核对。', r'''
gap=np.maximum(y_test-test_staff[selected]*capacity,0)
assert np.isclose(gap.mean(),test_scores[selected]['mean_shortage'])
print(gap.mean())
'''),
E('修改：缺口成本翻倍', '用验证集客流把缺口成本改为 16，重新比较策略；不需要重新训练预测模型。', r'''
new_cost={name:float(np.mean(staff*wage+np.maximum(y_val-staff*capacity,0)*16)) for name,staff in staffing_predictions(X_val).items()}
assert all(np.isfinite(list(new_cost.values())))
print(new_cost,'新方案',min(new_cost,key=new_cost.get))
'''),
E('迁移：人员最多 6 人', '给测试时段配置加入 6 人上限，输出被截断时段数与新缺口率。', r'''
capped=np.minimum(test_staff[selected],6)
assert capped.max()<=6
print({'截断时段':int(np.sum(test_staff[selected]>6)),**evaluate_staff(capped,y_test)})
''')], [('MAPIE 独立校准与测试数据的作用','https://mapie.readthedocs.io/en/stable/content/conformal-prediction/split-cross-conformal/'),('梯度提升回归',SK+'modules/ensemble.html#histogram-based-gradient-boosting')],
prereqs='B07、B15；先区分预测区间与平均误差，再学习成本函数。',
pitfalls='独立模拟时段满足教学假设，真实相邻时段可能相关；不能直接照搬交换性覆盖保证。区间不是排班目标，人数还需要结合成本与硬限制。',
decision='运行后查看 outputs/决策建议.md、staffing.csv 和 metrics.json。推荐方案由验证集成本选出，报告来自独立测试集。')

make('A', 12, '降雨风险与转场', '有降雨风险时，应该提前转场还是等待？',
     '把明确标注的降雨概率情景转成行动：户外、立即室内、等待一个有噪声信号再决定。使用贝叶斯更新与决策树计算预期损失，不把温度预测当降雨概率。', [
S('## 1. 列出天气状态与损失表\n仅两种状态：下雨与不下雨。户外遇雨损失 12000，立即转场成本 3000。概率 0.35 是假设，不是查询到的天气预报；每项成本也都是教学单位。', r'''
import numpy as np,pandas as pd,json
prior=0.35; rain_loss=12000; early_indoor=3000; late_indoor=3800; wait_fee=500
sensitivity=0.8; false_positive=0.2
loss_table=pd.DataFrame({'no_rain':[0,early_indoor],'rain':[rain_loss,early_indoor]},index=['outdoor','indoor_now'])
print(loss_table)
'''),
S('## 2. 等待带来信息也带来代价\n湿信号在下雨时出现概率为 sensitivity，在晴天时误报概率为 false_positive。先算信号概率，再用贝叶斯公式算后验；后验必须落在 0—1，不应直接把 0.8 当作下雨概率。', r'''
def signal_branches(p,sensitivity=0.8,false_positive=0.2):
    wet_probability=p*sensitivity+(1-p)*false_positive
    branches=[]
    for label,prob, rain_joint in [('wet',wet_probability,p*sensitivity),('dry',1-wet_probability,p*(1-sensitivity))]:
        posterior=rain_joint/prob if prob>0 else 0.0
        branches.append({'signal':label,'probability':prob,'posterior_rain':posterior})
    return branches
branches=signal_branches(prior,sensitivity,false_positive)
assert np.isclose(sum(b['probability']*b['posterior_rain'] for b in branches),prior)
'''),
S('## 3. 从决策树末端往前计算\n观察信号后，在户外与延迟转场之间选损失较低者，再按信号概率加权，最后加等待成本。立即转场阈值为 3000/12000=0.25，但加入等待选项后还要比较第三条曲线。', r'''
def expected_losses(p,waiting_fee=wait_fee):
    branch_rows=signal_branches(p,sensitivity,false_positive)
    expected_wait=waiting_fee
    for branch in branch_rows:
        outdoor=branch['posterior_rain']*rain_loss
        branch['action']='outdoor' if outdoor<=late_indoor else 'indoor_later'
        branch['loss_after_signal']=min(outdoor,late_indoor)
        expected_wait+=branch['probability']*branch['loss_after_signal']
    return {'outdoor':p*rain_loss,'indoor_now':early_indoor,'wait':expected_wait},branch_rows
losses,branch_decisions=expected_losses(prior)
selected=min(losses,key=losses.get)
curve=pd.DataFrame([{'rain_probability':p,**expected_losses(p)[0]} for p in np.linspace(0,1,101)])
'''),
S('## 4. 交付规则与改变建议的条件\n输出当前推荐、每个信号后的动作和全概率范围的决策曲线。风险厌恶、场地容量和极端损失尚未建模；如果等待会失去室内场地，应修改树而非继续使用旧公式。', r'''
metrics={'assumed_rain_probability':prior,'expected_losses':losses,'selected':selected,'immediate_indoor_threshold':early_indoor/rain_loss,'signal_sensitivity':sensitivity,'signal_false_positive':false_positive}
curve.to_csv(OUT_DIR/'decision_curve.csv',index=False)
pd.DataFrame(branch_decisions).to_csv(OUT_DIR/'signal_actions.csv',index=False)
(OUT_DIR/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
advice=f'在假设降雨概率 {prior:.0%} 下，选择 {selected}，预期损失 {losses[selected]:.2f}；户外 {losses["outdoor"]:.2f}，立即室内 {losses["indoor_now"]:.2f}，等待 {losses["wait"]:.2f}。等待信号灵敏度 {sensitivity:.0%}、误报率 {false_positive:.0%}，需在实际取得新信息后按 signal_actions.csv 执行。所有数值均为教学情景，若场地或成本变化须重算。'
(OUT_DIR/'决策建议.md').write_text('# 转场建议\n\n'+advice,encoding='utf-8')
curve.set_index('rain_probability')[['outdoor','indoor_now','wait']].plot(); plt.xlabel('假设降雨概率'); plt.ylabel('预期损失'); plt.tight_layout(); plt.savefig(OUT_DIR/'decision.png'); plt.close(); print(metrics)
''')], [
E('补全：检验边界概率', 'p=0 时户外期望损失为 0；p=1 时为 rain_loss。', r'''
assert expected_losses(0)[0]['outdoor']==0
assert expected_losses(1)[0]['outdoor']==rain_loss
assert np.isclose(early_indoor/rain_loss,0.25)
print('边界检查通过')
'''),
E('修改：等待成本上涨', '将等待成本提高到 2000，确认最优选择不再是等待。', r'''
expensive,_=expected_losses(prior,waiting_fee=2000)
choice=min(expensive,key=expensive.get)
assert choice!='wait'
print(expensive,choice)
'''),
E('迁移：不同可靠性的信号', '若一个设备检测的灵敏度为 0.9、误报率为 0.1、故障先验为 0.05，计算阳性后的故障概率。', r'''
diagnostic=signal_branches(0.05,sensitivity=0.9,false_positive=0.1)
posterior=diagnostic[0]['posterior_rain']
assert np.isclose(posterior,0.045/(0.045+0.095))
print('阳性后的故障概率',posterior,'不是0.9')
''')], [('AIMA 官方概率与决策网络实现：期望效用','https://github.com/aimacode/aima-python/blob/master/aima/probability.py')],
prereqs='B09；概率、条件概率、期望损失；不要求外部天气服务。',
pitfalls='所有降雨概率与成本是标明的假设，不能转述为当前城市真实预报。贝叶斯更新依赖信号质量已知，现实中需要校准资料。', decision='本例同时输出立即决策阈值与有噪声信息后的行动。实际计算建议保存在 outputs/决策建议.md。')

make('A', 13, '项目延期与缓冲', '工期不稳定时，怎样估计延期概率与安排缓冲？',
     '用带相关性的模拟任务历史估计工期分布，保留任务依赖。比较把两天缓冲放在搭建前或项目末尾；缓冲只有通过预约机制才可能改变延期风险，不凭空缩短工作。', [
S('## 1. 历史数据与依赖图\n六个任务：A方案；B采购和C审批都在A后；D搭建需B、C完成；E验收在D后；F宣传在A后。历史每行是一整个项目，同一项目的共同波动造成任务相关性。', r'''
import numpy as np,pandas as pd,json
rng=np.random.default_rng(113)
names=['A','B','C','D','E','F']; means=np.array([2,5,7,3,2,6])
def new_projects(n,random):
    shared=random.normal(0,0.14,(n,1))
    individual=random.normal(0,0.12,(n,6))
    return means*np.exp(shared+individual-(0.14**2+0.12**2)/2)
history=new_projects(100,rng)
history_frame=pd.DataFrame(history,columns=names); history_frame.to_csv(DATA_DIR/'simulated_project_history.csv',index=False)
estimated_mean=history.mean(axis=0)
nominal_ready=estimated_mean[0]+max(estimated_mean[1],estimated_mean[2])
deadline=nominal_ready+estimated_mean[3]+estimated_mean[4]+2
'''),
S('## 2. 用预约机制明确缓冲的作用\n搭建队只有预约时点可立即开工，若准备晚于预约则额外等待 3 天重新预约。两种方案总预留均为 2 天：before 把两天放在搭建前，end 把两天留在最后。宣传并行，最终必须验收与宣传均完成。', r'''
def finish_times(durations,buffer_before):
    ready=durations[:,0]+np.maximum(durations[:,1],durations[:,2])
    slot=nominal_ready+buffer_before
    missed=ready>slot
    start_d=np.maximum(ready,slot)+3*missed
    finish_e=start_d+durations[:,3]+durations[:,4]
    finish_f=durations[:,0]+durations[:,5]
    return np.maximum(finish_e,finish_f),missed
tiny=np.array([[2,5,7,3,2,6]],dtype=float)
tiny_finish,_=finish_times(tiny,0)
assert tiny_finish[0]>=14
'''),
S('## 3. 用整行重抽样保留任务相关性\n从历史项目中重抽整行，不能分别抽每列后声称保留了相关性。用 2000 个开发情景选择缓冲位置，固定准时目标，不在最终新项目上挑方案。', r'''
development=history[rng.integers(0,len(history),2000)]
policies={'end':0.0,'before':2.0}
development_results={}
for name,buffer_before in policies.items():
    finish,missed=finish_times(development,buffer_before)
    development_results[name]={'late_probability':float(np.mean(finish>deadline)),'mean_finish':float(finish.mean()),'missed_booking_rate':float(missed.mean())}
selected=min(development_results,key=lambda name:development_results[name]['late_probability'])
'''),
S('## 4. 独立未来情景验证并说明局限\n最终 2000 个情景从同一已知模拟机制重新生成，不复用历史重抽样。报告不同缓冲方案的完工分布、延期率和重新预约率；对同一项目情景的延期指示作配对差值，用其标准误构造近似蒙特卡洛区间，判断微小差别是否只是模拟噪声。该区间不涵盖模型偏差；真实任务还要检验分布、依赖与预约规则。', r'''
future=new_projects(2000,np.random.default_rng(114))
future_results={}; rows=[]; finish_by_policy={}
for name,buffer_before in policies.items():
    finish,missed=finish_times(future,buffer_before); finish_by_policy[name]=finish
    future_results[name]={'late_probability':float(np.mean(finish>deadline)),'mean_finish':float(finish.mean()),'p90_finish':float(np.quantile(finish,0.9)),'missed_booking_rate':float(missed.mean())}
    rows.extend({'policy':name,'finish':float(f),'late':bool(f>deadline)} for f in finish)
metrics={'deadline':float(deadline),'selected_on_development':selected,'development':development_results,'independent_simulation':future_results,'simulation_only':True}
paired_difference=(finish_by_policy['before']>deadline).astype(float)-(finish_by_policy['end']>deadline).astype(float)
delta=float(paired_difference.mean()); monte_carlo_se=float(paired_difference.std(ddof=1)/np.sqrt(len(paired_difference)))
difference_interval=[delta-1.96*monte_carlo_se,delta+1.96*monte_carlo_se]
metrics['before_minus_end_late_probability']={'estimate':delta,'approx_95_monte_carlo_interval':difference_interval,'excludes_model_uncertainty':True}
pd.DataFrame(rows).to_csv(OUT_DIR/'finish_distribution.csv',index=False)
(OUT_DIR/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
advice=f'开发情景选择缓冲方案 {selected}。共同承诺工期 {deadline:.2f} 天；独立模拟延期率 {future_results[selected]["late_probability"]:.1%}，90% 完工分位 {future_results[selected]["p90_finish"]:.2f} 天。前置缓冲会推迟预约，但降低错过搭建队的风险；若不存在 3 天重约等待，本结论需重新计算。历史仅为模拟，不代表真实项目可靠性。'
advice += f' 前置减末端方案的延期率差为 {delta:.2%}，共同随机情景下的近似 95% 蒙特卡洛区间为 [{difference_interval[0]:.2%}, {difference_interval[1]:.2%}]；该区间不涵盖建模偏差。'
if difference_interval[0]<=0<=difference_interval[1]:
    advice += ' 目前差异不足以说明哪个方案稳定降低延期率；需结合预约失败后果、成本与更多真实记录决定。'
(OUT_DIR/'决策建议.md').write_text('# 工期建议\n\n'+advice,encoding='utf-8')
for name,finish in finish_by_policy.items(): plt.hist(finish,bins=35,alpha=0.5,label=name,density=True)
plt.axvline(deadline,color='black',ls='--',label='承诺日期'); plt.legend(); plt.xlabel('完工天数'); plt.ylabel('密度'); plt.tight_layout(); plt.savefig(OUT_DIR/'completion.png'); plt.close(); print(metrics)
''')], [
E('补全：核对延期率', '从 finish_by_policy 重算选中方案延期比例，并与指标核对。', r'''
rate=float(np.mean(finish_by_policy[selected]>deadline))
assert np.isclose(rate,future_results[selected]['late_probability'])
print(rate)
'''),
E('修改：承诺日期晚两天', '保持模拟结果不变，仅将交付承诺推迟两天；延期率不应上升。', r'''
later={name:float(np.mean(finish>deadline+2)) for name,finish in finish_by_policy.items()}
assert all(later[name]<=future_results[name]['late_probability'] for name in policies)
print(later)
'''),
E('迁移：没有预约限制的小项目', '去掉预约时点与重新预约等待，按纯依赖计算完工时间，验证不晚于受预约约束的结果。', r'''
ready=future[:,0]+np.maximum(future[:,1],future[:,2])
unconstrained=np.maximum(ready+future[:,3]+future[:,4],future[:,0]+future[:,5])
assert np.all(unconstrained<=finish_by_policy['end']+1e-10)
print({'无预约约束平均工期':float(unconstrained.mean()),'延期率':float(np.mean(unconstrained>deadline))})
''')], [('NumPy 随机抽样官方 API','https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html'),('NetworkX 有向无环图算法','https://networkx.org/documentation/stable/reference/algorithms/dag.html')],
prereqs='B27、A04；概率分布、任务依赖、整行 bootstrap 与独立情景。',
pitfalls='缓冲不是能自动缩短真实工时的资源。本例效应来自明确的预约失败机制；改规则就应重跑。2000 次模拟降低蒙特卡洛误差，不能消除建模偏差。', decision='推荐缓冲位置取自开发情景；独立模拟结果与完整适用条件写入 outputs/决策建议.md。')

make('A', 14, '故障概率与维护决策', '设备可能出问题时，什么时候值得停机检查？',
     '将故障概率转换为预期维护收益，受最多检查 40 台的容量约束。比较不检查、固定 0.5 阈值与按净收益排序；概率校准与决策收益分别核验。', [
S('## 1. 明确维修作用与成本\n模拟传感器特征与故障结果；故障标签表示不干预时的潜在故障。检查每台成本 40，故障损失 600，检查可消除 85% 风险。这个干预有效性是教学假设，真实应用需独立证据。', r'''
import numpy as np,pandas as pd,json
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
X,y=make_classification(n_samples=1600,n_features=7,n_informative=5,weights=[0.88,0.12],class_sep=1.3,random_state=114)
X_train,X_hold,y_train,y_hold=train_test_split(X,y,test_size=0.5,stratify=y,random_state=114)
X_val,X_test,y_val,y_test=train_test_split(X_hold,y_hold,test_size=0.5,stratify=y_hold,random_state=114)
inspection_cost,failure_loss,effectiveness,capacity=40,600,0.85,40
raw=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=500)).fit(X_train,y_train)
calibrated=CalibratedClassifierCV(make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=500)),method='sigmoid',cv=3).fit(X_train,y_train)
'''),
S('## 2. 先用验证概率质量选择概率模型\n误报和漏报成本不能直接修复概率偏差；先用验证 Brier 分数比较原始与校准概率，锁定候选后再看测试。校准不一定获胜，因此由实际验证结果选择。', r'''
candidates={'raw':raw,'calibrated':calibrated}
validation_brier={name:float(brier_score_loss(y_val,estimator.predict_proba(X_val)[:,1])) for name,estimator in candidates.items()}
probability_model=min(validation_brier,key=validation_brier.get)
p=candidates[probability_model].predict_proba(X_test)[:,1]
threshold=inspection_cost/(failure_loss*effectiveness)
benefit=p*failure_loss*effectiveness-inspection_cost
'''),
S('## 3. 正净收益且容量允许才检查\n检查收益是可避免故障损失减去检查成本。设备检查耗时相同时，按正净收益从大到小取前 40 台；若检查时长不同应改背包或约束优化，而不能继续简单排序。', r'''
def choose_inspections(probability,minimum_threshold,max_checks):
    eligible=np.flatnonzero(probability>=minimum_threshold)
    ordered=eligible[np.argsort(-probability[eligible])]
    mask=np.zeros(len(probability),dtype=bool); mask[ordered[:max_checks]]=True
    return mask
policies={'none':np.zeros(len(p),dtype=bool),'threshold_0_5':choose_inspections(p,0.5,capacity),'net_benefit':choose_inspections(p,threshold,capacity)}
def policy_cost(mask,truth):
    costs=inspection_cost*mask+failure_loss*truth*(1-effectiveness*mask)
    return {'total_cost':float(costs.sum()),'inspections':int(mask.sum()),'uninspected_failures':int(np.sum(truth & ~mask))}
scores={name:policy_cost(mask,y_test.astype(bool)) for name,mask in policies.items()}
assert all(mask.sum()<=capacity for mask in policies.values())
'''),
S('## 4. 决策报告区分预测、假设与实际情景结算\n测试标签不参与检查排序。检查后的残余故障成本按 15% 期望值结算，属于模拟干预假设下的评价，不是观察到的真实维修效果。报告被容量挡住的正收益设备。', r'''
selected=policies['net_benefit']
queue=pd.DataFrame({'device':np.arange(len(p)),'probability':p,'expected_net_benefit':benefit,'inspect':selected,'simulated_failure_without_action':y_test})
queue.sort_values('expected_net_benefit',ascending=False).to_csv(OUT_DIR/'maintenance_queue.csv',index=False)
metrics={'probability_model':probability_model,'validation_brier':validation_brier,'test_brier':float(brier_score_loss(y_test,p)),'action_threshold':threshold,'capacity':capacity,'costs_under_assumed_effectiveness':scores,'positive_benefit_without_capacity':int(np.sum((benefit>0)&~selected))}
(OUT_DIR/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
advice=f'在检查成本 {inspection_cost}、故障损失 {failure_loss}、风险消除比例 {effectiveness:.0%} 的假设下，理论阈值 {threshold:.3f}；容量限定后检查 {int(selected.sum())} 台。测试情景期望结算成本 {scores["net_benefit"]["total_cost"]:.2f}，不检查为 {scores["none"]["total_cost"]:.2f}。概率模型由验证集选择为 {probability_model}。维修效果是外加假设，若效果下降、检查时长不一致或数据变化，应重新建模。'
(OUT_DIR/'决策建议.md').write_text('# 维护建议\n\n'+advice,encoding='utf-8')
pd.DataFrame(scores).T['total_cost'].plot.bar(); plt.ylabel('模拟期望成本'); plt.xticks(rotation=15); plt.tight_layout(); plt.savefig(OUT_DIR/'maintenance_cost.png'); plt.close(); print(metrics)
''')], [
E('补全：盈亏平衡', '在理论阈值处验证预期避免损失恰等于检查成本。', r'''
assert np.isclose(threshold*failure_loss*effectiveness,inspection_cost)
assert (queue.loc[queue['inspect'],'expected_net_benefit']>=-1e-10).all()
print(threshold)
'''),
E('修改：容量减半', '把容量改为 20，核对检查不超过 20 台，并输出情景成本。', r'''
small=choose_inspections(p,threshold,20)
assert small.sum()<=20
assert np.all(~small|selected)
print(policy_cost(small,y_test.astype(bool)))
'''),
E('迁移：检查效果只剩 50%', '仅重新计算理论阈值与检查名单，说明为什么阈值升高；不要继续使用旧效果结算函数。', r'''
weaker=0.5
new_threshold=inspection_cost/(failure_loss*weaker)
new_mask=choose_inspections(p,new_threshold,capacity)
new_cost=float(np.sum(inspection_cost*new_mask+failure_loss*y_test*(1-weaker*new_mask)))
assert new_threshold>threshold
print({'新阈值':new_threshold,'检查数':int(new_mask.sum()),'新效果假设成本':new_cost})
''')], [('概率校准',SK+'modules/calibration.html'),('基于代价的阈值选择',SK+'auto_examples/model_selection/plot_cost_sensitive_learning.html')],
prereqs='B09、B29；概率校准、期望损失、排序与容量约束。', pitfalls='有故障风险并不自动意味着检查有效。此处潜在结果与维护效果来自教学模拟，真实观察标签不能直接识别维修因果效果。', decision='outputs/maintenance_queue.csv 给检查清单，outputs/决策建议.md 将收益、容量和效果假设一起交付。')

make('A', 15, '识别措施因果效果', '措施实施后结果变好，真的是措施的作用吗？',
     '用已知机制的模拟培训数据比较直接均值差、控制混杂后的估计与随机 A/B。使用 DoWhy 明确建模、识别、估计和安慰剂反驳；检验通过不等于证明因果。', [
S('## 1. 明确目标效应与混杂机制\nW 是原有经验，T 是是否参加培训，Y 是结果。经验同时影响参训与结果，直接比较培训组会混入经验差异。模拟真实平均效应设为 2，只用于核验教学代码；真实数据通常不知道这个答案。', r'''
import numpy as np,pandas as pd,json,warnings
from scipy.special import expit
from dowhy import CausalModel
rng=np.random.default_rng(115); n=1200
W=rng.normal(0,1,n); probability=expit(0.9*W)
T=rng.binomial(1,probability).astype(int)
Y=10+2*T+3*W+rng.normal(0,1,n)
data=pd.DataFrame({'experience':W,'training':T,'outcome':Y})
data.to_csv(DATA_DIR/'simulated_training.csv',index=False)
naive=float(data.loc[T==1,'outcome'].mean()-data.loc[T==0,'outcome'].mean())
graph='digraph { experience -> training; experience -> outcome; training -> outcome; }'
assumptions='经验是所有共同原因；无未测混杂；个体间无干扰；培训含义一致；每种经验附近都可能参训或不参训。'
'''),
S('## 2. 先识别，再估计\n图来自机制假设，不由数据自动证明。DoWhy 先给出后门调整表达式，再调用线性回归估计。本例已知模拟图没有未测共同原因；另构造有未观测 U 的假设图，展示后门调整失败时停止估计。proceed_when_unidentifiable=True 只避免交互提示，不是允许强行得出因果结果。', r'''
causal_model=CausalModel(data=data,treatment='training',outcome='outcome',graph=graph)
estimand=causal_model.identify_effect(proceed_when_unidentifiable=True)
adjustment=estimand.get_backdoor_variables()
assert 'experience' in adjustment
estimate=causal_model.estimate_effect(estimand,method_name='backdoor.linear_regression')
adjusted=float(estimate.value)
(OUT_DIR/'identified_estimand.txt').write_text(str(estimand)+'\n\n假设：'+assumptions,encoding='utf-8')
unobserved_graph='digraph { experience -> training; experience -> outcome; training -> outcome; U -> training; U -> outcome; }'
unobserved_model=CausalModel(data=data,treatment='training',outcome='outcome',graph=unobserved_graph)
unobserved_estimand=unobserved_model.identify_effect(proceed_when_unidentifiable=True)
unobserved_backdoor_unidentified=unobserved_estimand.estimands.get('backdoor') is None
assert unobserved_backdoor_unidentified
(OUT_DIR/'unidentified_scenario.txt').write_text(str(unobserved_estimand)+'\n\nU 未被测量，当前图下不能用已观测变量完成后门调整；不继续调用估计器，应收集共同原因或设计随机试验。',encoding='utf-8')
print('直接均值差',naive,'后门调整',adjusted)
'''),
S('## 3. 安慰剂、抽样不确定性与随机 A/B\n打乱培训标签后应不再保留原来的效应，这是可反驳检查而非因果证明。为控制练习时长，只运行 10 次安慰剂和 250 次 bootstrap；次数仅适合演示，不能把小规模检验结果当充分证据。对独立行重抽样重算系数，区间只描述抽样不确定性；另生成随机参训数据，展示随机化减少系统性混杂。', r'''
refutation=causal_model.refute_estimate(estimand,estimate,method_name='placebo_treatment_refuter',placebo_type='permute',num_simulations=10,random_seed=115,n_jobs=1,show_progress_bar=False)
placebo=float(np.asarray(refutation.new_effect).mean())
def adjusted_coefficient(frame):
    design=np.column_stack([np.ones(len(frame)),frame['training'],frame['experience']])
    return float(np.linalg.lstsq(design,frame['outcome'],rcond=None)[0][1])
bootstrap=[]
boot_rng=np.random.default_rng(116)
for _ in range(250):
    bootstrap.append(adjusted_coefficient(data.iloc[boot_rng.integers(0,n,n)]))
interval=np.quantile(bootstrap,[0.025,0.975])
ab_t=rng.binomial(1,0.5,n)
ab_y=10+2*ab_t+3*W+rng.normal(0,1,n)
ab_effect=float(ab_y[ab_t==1].mean()-ab_y[ab_t==0].mean())
(OUT_DIR/'refutation.txt').write_text(str(refutation)+'\n\n通过反驳检验不证明识别假设成立。',encoding='utf-8')
'''),
S('## 4. 用偏差情景说明未测混杂的风险\n假设遗漏变量在两组条件均值差为 delta，对结果作用为 gamma，则线性加性情景下偏差为 gamma×delta。扫描假设强度，观察结论何时翻转；不是从数据估出了隐藏变量。若真实图还有未测共同原因且无其他识别设计，应停止因果估计并提出随机试验。', r'''
sensitivity_rows=[{'gamma':g,'delta':d,'assumed_bias':g*d,'bias_adjusted_effect':adjusted-g*d} for g in [0,1,2,3,4] for d in [0,0.5,1]]
sensitivity=pd.DataFrame(sensitivity_rows)
metrics={'true_effect_in_simulator':2,'naive_difference':naive,'adjusted_effect':adjusted,'bootstrap_interval_95':[float(v) for v in interval],'randomized_ab_difference':ab_effect,'placebo_effect':placebo,'adjustment_variables':adjustment,'assumptions':assumptions,'refutation_is_proof':False,'unobserved_scenario_backdoor_unidentified':unobserved_backdoor_unidentified}
sensitivity.to_csv(OUT_DIR/'unmeasured_bias_scenarios.csv',index=False)
(OUT_DIR/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
advice=f'直接比较得到 {naive:.3f}，按经验调整后为 {adjusted:.3f}，独立样本 bootstrap 95% 区间 [{interval[0]:.3f},{interval[1]:.3f}]；随机 A/B 对照为 {ab_effect:.3f}，安慰剂为 {placebo:.3f}。这些是已知模拟图的演示。采用因果解释前必须支持“{assumptions}”。反驳通过不等于证明，隐藏偏差达到估计量大小时正效应可能消失；真实项目应优先收集共同原因或安排随机试验。'
(OUT_DIR/'决策建议.md').write_text('# 因果分析建议\n\n'+advice,encoding='utf-8')
plt.bar(['直接均值差','调整估计','随机A/B','安慰剂'],[naive,adjusted,ab_effect,placebo]); plt.axhline(2,ls='--',color='grey',label='模拟真实效应'); plt.ylabel('效果 / 教学单位'); plt.legend(); plt.tight_layout(); plt.savefig(OUT_DIR/'causal_comparison.png'); plt.close(); print(metrics)
''')], [
E('补全：手写回归与 DoWhy 一致', '使用 adjusted_coefficient 在原数据上重算，核对与 DoWhy 结果一致。', r'''
manual=adjusted_coefficient(data)
assert np.isclose(manual,adjusted,atol=1e-6)
print(manual)
'''),
E('修改：计算结论翻转所需偏差', '若遗漏变量两组均值差 delta=0.5，计算使正效应消失所需的 gamma。', r'''
delta=0.5
tipping_gamma=adjusted/delta
assert np.isclose(adjusted-tipping_gamma*delta,0)
print('条件性翻转阈值 gamma',tipping_gamma,'不是数据估出的隐藏因素强度')
'''),
E('迁移：随机试验的均值差与区间', '对随机 A/B 数据重抽样 200 次，计算均值差分布与 95% 区间，避免把观察性 bootstrap 当随机化。', r'''
ab_rng=np.random.default_rng(117); effects=[]
for _ in range(200):
    idx=ab_rng.integers(0,n,n)
    sample_t,sample_y=ab_t[idx],ab_y[idx]
    effects.append(sample_y[sample_t==1].mean()-sample_y[sample_t==0].mean())
ab_interval=np.quantile(effects,[0.025,0.975])
assert ab_interval.shape==(2,) and ab_interval[0]<ab_interval[1]
print('随机试验抽样区间',ab_interval)
''')], [('DoWhy 官方基本因果效应示例','https://www.pywhy.org/dowhy/v0.13/example_notebooks/dowhy_simple_example.html')],
prereqs='B10、B27；相关与因果、因果图、线性估计、bootstrap。',
pitfalls='identify 返回表达式只说明在输入图与假设下可识别；图可能错误。反驳检验通过不能验证没有未测混杂，bootstrap 也不涵盖识别假设错误。', decision='输出同时保留识别表达式、反驳结果、未测混杂偏差情景与决策建议；不把统计显著或安慰剂通过写成因果证明。')

print('B01–B15 与 A11–A15 已生成。')
