"""对选修拓展做同样的离线限制与运行记录；不会安装或下载。"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time
from run_case import ROOT, environment

parser = argparse.ArgumentParser()
parser.add_argument('--ids', default='B15,A27')
args = parser.parse_args()
targets = {
    'B15': next((ROOT / '01_AI基础课').glob('15_*/extension_chronos.py')),
    'A27': next((ROOT / '02_AI进阶课').glob('27_*/extension_dqn.py')),
}
report_path = ROOT / 'reports' / 'extension_validation.json'
rows = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else []
for case_id in args.ids.split(','):
    target = targets[case_id]
    network = ROOT / 'reports' / (case_id + '_extension_network.jsonl')
    network.write_text('', encoding='utf-8')
    env = environment()
    env['COURSE_NETWORK_LOG'] = str(network)
    start = time.monotonic()
    result = subprocess.run([sys.executable, str(target)], cwd=target.parent, env=env,
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=1200)
    log = ROOT / 'reports' / (case_id + '_extension.log')
    log.write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
    attempted = bool(network.read_text(encoding='utf-8').strip())
    row = {'case': case_id, 'path': target.relative_to(ROOT).as_posix(),
           'source_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
           'passed': result.returncode == 0 and not attempted,
           'external_network_attempt': attempted, 'returncode': result.returncode,
           'seconds': round(time.monotonic() - start, 2),
           'tested_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'log': log.relative_to(ROOT).as_posix()}
    rows = [r for r in rows if r['case'] != case_id] + [row]
    report_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(row, ensure_ascii=False), flush=True)
    if not row['passed']:
        print((result.stdout + result.stderr)[-4000:])
raise SystemExit(0 if all(r['passed'] for r in rows) else 1)
