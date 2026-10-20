"""整理入口、许可证、案例清单；不覆盖用户学习记录。"""
from pathlib import Path
import json,csv,html,urllib.parse,datetime
ROOT=Path(__file__).resolve().parents[1]
cases=[]
online_topics={
    'B07':'另行获取 TabPFN 权重并比较，需要在线确认访问权限与下载；本课回归主线不需要。',
    'B10':'新增 TabPFN 模型对照，需要在线确认访问权限与下载；本课同预算选型主线不需要。',
    'B15':'下载其他 Chronos/TimesFM 权重或新天气数据需要联网；预先下载的 Chronos-Bolt-tiny 可离线运行。',
    'B21':'改用云端生成服务或下载新的预训练模型，需要联网；预先下载的 FLAN-T5-small 可离线运行。',
    'B22':'把文本模型接入在线服务或获取新训练数据，需要联网；本课已有文本材料可离线运行。',
    'B23':'在线搜索、新增语料或下载其他语义模型，需要联网；预先下载的 MiniLM 和笔记可离线运行。',
    'B24':'改用云端大模型、在线搜索或新的模型权重，需要联网；预先下载的 Qwen 与检索材料可离线运行。',
    'A09':'调用云模型生成新的规划草稿需要联网；本地合法/错误草稿及独立检查器可离线学习。',
    'A30':'实时外部资料与云模型调用需要联网；本地综合规划、仿真和重规划主线可离线运行。',
}
for track in ['01_AI基础课','02_AI进阶课']:
    for file in sorted((ROOT/track).glob('*/case.json')):
        meta=json.loads(file.read_text(encoding='utf-8'));meta['directory']=str(file.parent.relative_to(ROOT)).replace('\\','/');cases.append(meta)
        readme=file.parent/'README.md';text=readme.read_text(encoding='utf-8')
        banner='> **核心案例：离线可学。** 首次先按[安装指南](../../安装指南.md)配置环境、下载所需模型。准备完成后，输入均从本课程目录读取。外部参考链接不影响本地运行。需要联网的延伸活动见[联网拓展清单](../../联网拓展清单.md)，可延后学习。\n\n'
        text = '\n'.join(line for line in text.split('\n') if not line.startswith('> **核心案例：离线可学。**'))
        readme.write_text(text,encoding='utf-8')
        if '**核心案例：离线可学。**' not in text:
            first,rest=text.split('\n',1);readme.write_text(first+'\n\n'+banner+rest.lstrip(),encoding='utf-8')
        text=readme.read_text(encoding='utf-8')
        if meta['id'] in online_topics and '**联网选修：需要联网，可延后学习。**' not in text:
            first,rest=text.split('\n',1)
            readme.write_text(first+'\n\n> **联网选修：需要联网，可延后学习。** '+online_topics[meta['id']]+'\n'+rest,encoding='utf-8')
        if (file.parent/'extension_chronos.py').exists():
            tail='\n\n## 已附本地拓展\n见 [Chronos离线对照](拓展_Chronos离线对照.md)，使用预先下载的小模型，验收状态见总报告。\n'
            if '## 已附本地拓展' not in readme.read_text(encoding='utf-8'):readme.write_text(readme.read_text(encoding='utf-8')+tail,encoding='utf-8')
        if (file.parent/'extension_dqn.py').exists():
            tail='\n\n## 已附本地拓展\n见 [DQN离线实验](拓展_DQN.md)，无需下载模型，验收状态见总报告。\n'
            if '## 已附本地拓展' not in readme.read_text(encoding='utf-8'):readme.write_text(readme.read_text(encoding='utf-8')+tail,encoding='utf-8')
report_files=[ROOT/'reports'/n for n in ['data_validation.json','model_validation.json','planning_validation.json','strategy_validation.json']]
results=[]
for f in report_files:
    if f.exists():results+=json.loads(f.read_text(encoding='utf-8'))['results']
historical = ROOT/'validation/original_windows_validation.json'
if not results and historical.exists():results=json.loads(historical.read_text(encoding='utf-8'))['results']
status={}
for case in cases:
    runs=[r for r in results if r['case']==case['id']]
    status[case['id']]='原版Windows已验证' if len(runs)==4 and all(r['passed'] for r in runs) else '待验证'
cards=[];markdown=['# 60个案例学习目录\n','从B01开始，先基础后进阶。首次安装见[安装指南](安装指南.md)。状态指原版Windows运行记录，公开版检查范围见[验收报告](验收报告.md)。\n','| 案例 | 问题 | 状态 |\n|---|---|---|']
for case in cases:
    folder=urllib.parse.quote(case['directory'],safe='/')
    name=html.escape(case['id']+' '+case['title']);group='基础课' if case['track']=='B' else '进阶课'
    cards.append(f'<article data-track="{case["track"]}"><small>{group} · {status[case["id"]]} · 离线核心</small><h3>{name}</h3><p><a href="{folder}/教学讲解.html">阅读讲解与结果</a> · <a href="{folder}/参考答案.html">参考答案</a></p><p><a href="{folder}/README.md">任务说明</a> · <a href="{folder}/教学讲解.ipynb">Notebook</a> · <a href="{folder}/solution.py">完整代码</a></p></article>')
    markdown.append(f'| {case["id"]} | [{case["title"]}]({case["directory"]}/README.md) | {status[case["id"]]} |')
(ROOT/'案例目录.md').write_text('\n'.join(markdown),encoding='utf-8')
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>AI离线实践课程</title><style>body{margin:0;background:#f2f5f6;color:#173044;font:16px/1.7 "Microsoft YaHei",sans-serif}header,main,footer{max-width:1140px;margin:auto;padding:32px}header{padding-top:50px}h1{font-size:36px;line-height:1.3;margin:12px 0}h3{font-size:18px;margin:8px 0}p{margin:8px 0}small{color:#507074}a{color:#086e79;text-underline-offset:4px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:18px}article{background:white;border:1px solid #d5e1e5;border-radius:12px;padding:22px}input{width:min(95%,550px);padding:12px;font:inherit;border:1px solid #abc1c6;border-radius:8px}button{padding:10px 18px;margin:12px 8px 0 0;background:#176777;color:white;border:0;border-radius:8px;font:inherit}.note{border-left:4px solid #d29c42;padding-left:16px}.tag{letter-spacing:2px;color:#376d6d}footer{color:#526b76}</style><body><header><div class="tag">60 个问题 · Python 实现 · 分析与决策</div><h1>AI 离线实践课程</h1><p>先学会独立实现，再学习建模、规划、推演和策略选择。</p><p class="note">首次先按 <a href="安装指南.md">安装指南</a> 配置环境和模型，再双击「开始学习.cmd」。本页和讲解HTML可直接离线阅读。Notebook需要课程内核。先读 <a href="先读我_离线使用.md">使用说明</a>；遇到问题查 <a href="故障排查.md">故障排查</a>。</p><p><a href="学习路线与迁移挑战.md">学习路线与迁移挑战</a> · <a href="Python知识索引.md">Python知识索引</a> · <a href="验收报告.md">验证范围与记录</a> · <a href="联网拓展清单.md">可延后学习的联网拓展</a></p><input id="query" placeholder="搜索问题，例如：天气、排班、因果、检索"><div><button onclick="choose('')">全部60例</button><button onclick="choose('B')">基础课30例</button><button onclick="choose('A')">进阶课30例</button></div></header><main><div class="grid">'''+''.join(cards)+'''</div></main><footer>教学数据与方法都有适用条件。指标来自案例实际运行，不代表真实部署效果。仓库包含代码、讲解、练习、答案和小数据；环境、权重与离线安装材料需按指南预先准备。</footer><script>let track='';function choose(t){track=t;filter()}function filter(){const q=document.getElementById('query').value.toLowerCase();document.querySelectorAll('article').forEach(x=>x.hidden=!!((track&&x.dataset.track!==track)||!x.textContent.toLowerCase().includes(q)))}document.getElementById('query').addEventListener('input',filter)</script></body></html>'''
(ROOT/'课程首页.html').write_text(page,encoding='utf-8')
page=page.replace('<a href="Python知识索引.md">Python知识索引</a>', '<a href="Python知识索引.md">Python知识索引</a> · <a href="Python离线救急手册.md">Python救急手册</a>')
(ROOT/'课程首页.html').write_text(page,encoding='utf-8')
(ROOT/'course_manifest.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
if not (ROOT/'学习记录.csv').exists():
    with (ROOT/'学习记录.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['案例','问题','首次运行日期','独立完成日期','难点或错误','迁移尝试'])
        for c in cases:writer.writerow([c['id'],c['title'],'','','',''])
print('目录已生成：',len(cases))
