from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
kernel=ROOT/'.venv'/'share'/'jupyter'/'kernels'/'python3'
kernel.mkdir(parents=True,exist_ok=True)
data={'argv':[str(ROOT/'.venv'/'Scripts'/'python.exe'),'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'AI离线课程 Python 3.12','language':'python','metadata':{'debugger':True}}
(kernel/'kernel.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print('课程内核已绑定：'+data['argv'][0])
