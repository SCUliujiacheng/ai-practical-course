"""用户入口；只绑定127.0.0.1，浏览器显示的是本机服务。"""
import os,sys,subprocess
from pathlib import Path
from run_case import ROOT,environment
env=environment()
env['COURSE_NETWORK_LOG']=str(ROOT/'reports'/'learning_network_attempts.jsonl')
print('正在启动本机 Jupyter。没有互联网也可以使用浏览器中的 127.0.0.1 页面。',flush=True)
print('结束学习时在此窗口按 Ctrl+C。',flush=True)
command=[sys.executable,'-m','jupyterlab','--ip=127.0.0.1','--port=8888','--port-retries=20','--ServerApp.root_dir='+str(ROOT),'--LabApp.check_for_updates_class=jupyterlab.NeverCheckForUpdate','--LabApp.extension_manager=readonly']
raise SystemExit(subprocess.call(command,cwd=ROOT,env=env))
