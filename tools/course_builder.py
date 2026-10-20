"""课程制作工具；只负责文档/Notebook 包装，不隐藏任何解题算法。"""
from pathlib import Path
import json
import re
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

def write_case(track, number, slug, title, intro, sections, exercises, sources,
               prereqs='', pitfalls='', decision='', data_note='教学模拟数据；生成机制、参数和随机种子见本案例代码。', extra_files=None):
    """sections=[(中文解释, Python代码)], exercises=[dict(title,prompt,starter,answer)]"""
    directory = ROOT / ('01_AI基础课' if track == 'B' else '02_AI进阶课') / f'{number:02d}_{slug}'
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'data').mkdir(exist_ok=True)
    (directory/'outputs').mkdir(exist_ok=True)
    header = '''from pathlib import Path
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
'''
    code = header + '\n\n' + '\n\n'.join('# %%\n# '+md.splitlines()[0]+'\n'+src for md,src in sections)
    (directory/'solution.py').write_text(code, encoding='utf-8')
    cells=[nbf.v4.new_markdown_cell(f'# {track}{number:02d} {title}\n\n{intro}\n\n先修：{prereqs}\n\n首次使用请按根目录安装指南准备环境；模型案例需预先下载权重，备妥后离线运行。\n\n**环境准备：** 下一格设置本地路径、离线模式和图表字体，直接运行即可，暂时不需要掌握所有设置。真正的解题步骤从后面的中文小节开始。'),nbf.v4.new_code_cell(header)]
    for md,src in sections:
        cells += [nbf.v4.new_markdown_cell(md), nbf.v4.new_code_cell(src)]
    cells += [nbf.v4.new_markdown_cell('## 结果图与检查\n下面展示本次计算保存的图。数值以单元格和 outputs 中的实际输出为准。'),nbf.v4.new_code_cell('from IPython.display import display, Image\nfor figure in sorted(OUT_DIR.glob("*.png")):\n    display(Image(filename=str(figure)))'),nbf.v4.new_markdown_cell(f'## 适用边界与常见错误\n{pitfalls}\n\n## 独立练习\n打开「动手练习.ipynb」，完成补全、修改、迁移，再查看「参考答案.ipynb」。')]
    metadata={'kernelspec':{'display_name':'AI离线课程 Python 3.12','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12.13'}}
    for name,cs in [('教学讲解.ipynb',cells)]:
        notebook=nbf.v4.new_notebook(cells=cs,metadata=metadata);nbf.validate(notebook);nbf.write(notebook,directory/name)
    exercise_setup='from pathlib import Path\nCASE_DIR = Path.cwd()\n# 载入本例的可修改函数和数据，所有算法都在同目录 solution.py。\nexec(compile((CASE_DIR / "solution.py").read_text(encoding="utf-8"), str(CASE_DIR / "solution.py"), "exec"))'
    for answered,name in [(False,'动手练习.ipynb'),(True,'参考答案.ipynb')]:
        cs=[nbf.v4.new_markdown_cell(f'# {title}：'+('参考答案' if answered else '动手练习')+'\n\n每道题先写预期结果再运行。练习中的 TODO 需要自行完成；能顺序运行不代表已完成练习。'),nbf.v4.new_code_cell(exercise_setup)]
        for i,exercise in enumerate(exercises,1):
            cs += [nbf.v4.new_markdown_cell(f'## {i}. {exercise["title"]}\n{exercise["prompt"]}'),nbf.v4.new_code_cell(exercise['answer'] if answered else exercise['starter'])]
        notebook=nbf.v4.new_notebook(cells=cs,metadata=metadata);nbf.validate(notebook);nbf.write(notebook,directory/name)
    source_text='\n'.join(f'- {title}: {url}' for title,url in sources)
    (directory/'来源与延伸阅读.md').write_text(f'# 来源、数据与改编\n\n{source_text}\n\n核验/整理日期：2026-10-20。以下是按公开方法重新编写的教学实现，不代表真实企业部署案例。\n\n{data_note}\n\n本地讲解和代码是主要学习资料，链接仅供有网络时延伸阅读。第三方代码、权重和数据许可见 assets 与课程根目录 THIRD_PARTY_NOTICES.md；原创教学代码采用 MIT，讲解采用 CC BY 4.0。\n',encoding='utf-8')
    (directory/'data'/'README.md').write_text(data_note+'\n\n本案例不依赖完成前面案例。需要的本地公共素材位于课程根目录 assets。',encoding='utf-8')
    (directory/'README.md').write_text(f'# {track}{number:02d} {title}\n\n{intro}\n\n## 先修与运行\n首次使用先完成根目录[安装指南](../../安装指南.md)；模型案例需预先下载权重。\n\n{prereqs}\n\n在课程根目录双击「开始学习.cmd」，进入本文件夹并打开教学讲解。也可以双击本目录「运行案例.cmd」。\n\n## 学习顺序\n1. 阅读任务，写下输入、输出和你预计的结果。\n2. 打开教学讲解，逐格运行，观察变量和图表。\n3. 对照 solution.py 临摹关键步骤。\n4. 完成动手练习的补全、修改与迁移，再看参考答案。\n5. 检查 outputs，解释方案的适用范围。\n\n## 方法与检查\n'+ '\n\n'.join(md for md,_ in sections)+f'\n\n## 常见错误与边界\n{pitfalls}\n\n## 数据\n{data_note}\n\n来源及改编说明见同目录「来源与延伸阅读.md」。完整代码、讲解与答案均在本地。\n',encoding='utf-8')
    (directory/'运行案例.cmd').write_text('@echo off\r\nchcp 65001 >nul\r\nset PYTHONUTF8=1\r\ncd /d "%~dp0"\r\n"..\\..\\.venv\\Scripts\\python.exe" "..\\..\\tools\\run_case.py" "%~dp0solution.py"\r\nset "COURSE_EXIT=%ERRORLEVEL%"\r\nif not "%COURSE_EXIT%"=="0" echo 运行失败，请查看错误信息和根目录故障排查说明。\r\npause\r\nexit /b %COURSE_EXIT%\r\n',encoding='utf-8',newline='')
    if decision:(directory/'决策说明.md').write_text('# 决策说明\n\n'+decision,encoding='utf-8')
    for name,content in (extra_files or {}).items():
        dest=directory/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(content,encoding='utf-8')
    (directory/'case.json').write_text(json.dumps({'id':f'{track}{number:02d}','title':title,'track':track,'number':number,'sources':sources,'prereqs':prereqs},ensure_ascii=False,indent=2),encoding='utf-8')
    return directory
