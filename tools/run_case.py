"""启动一个案例；通过子进程继承课程专用环境与离线检查。"""
from pathlib import Path
import os,sys,subprocess
ROOT=Path(__file__).resolve().parents[1]
def environment():
    env=os.environ.copy()
    env.update(COURSE_OFFLINE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUTF8='1',PYTHONIOENCODING='utf-8',MPLBACKEND='Agg')
    env['PYTHONPATH']=str(ROOT/'tools'/'offline_guard')
    env['HF_HOME']=str(ROOT/'reports'/'empty_hf_cache')
    env['MPLCONFIGDIR']=str(ROOT/'runtime'/'matplotlib_cache')
    env['JUPYTER_CONFIG_DIR']=str(ROOT/'runtime'/'jupyter_config')
    env['JUPYTER_DATA_DIR']=str(ROOT/'runtime'/'jupyter_data')
    env['JUPYTER_RUNTIME_DIR']=str(ROOT/'runtime'/'jupyter_runtime')
    for key in ['HF_HOME','MPLCONFIGDIR','JUPYTER_CONFIG_DIR','JUPYTER_DATA_DIR','JUPYTER_RUNTIME_DIR']:Path(env[key]).mkdir(parents=True,exist_ok=True)
    return env
if __name__=='__main__':
    target=Path(sys.argv[1]).resolve()
    if not target.is_file():raise SystemExit('案例脚本不存在：'+str(target))
    env=environment();env['COURSE_NETWORK_LOG']=str(ROOT/'reports'/'network_attempts.jsonl')
    result=subprocess.run([sys.executable,str(target),*sys.argv[2:]],cwd=target.parent,env=env)
    raise SystemExit(result.returncode)
