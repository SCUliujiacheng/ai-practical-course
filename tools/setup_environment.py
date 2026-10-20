"""Install the locked Windows x64 / Python 3.12 environment without moving an existing venv."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYPI = 'https://pypi.org/simple'
TORCH_INDEX = 'https://download.pytorch.org/whl/cpu'
PYTHON_HELP = '请先从 https://www.python.org/downloads/windows/ 安装 Python 3.12 x64（含 pip 与 venv），再用 py -3.12 运行本脚本。其他系统/架构未经本课程验证。'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require_windows():
    if platform.system() != 'Windows' or platform.machine().lower() not in {'amd64', 'x86_64'}:
        raise RuntimeError('本安装器仅支持已验证的 Windows x64。' + PYTHON_HELP)


def python_info(python):
    code = "import sys,platform,struct,json;print(json.dumps({'version':list(sys.version_info[:3]),'bits':struct.calcsize('P')*8,'system':platform.system(),'base_prefix':sys.base_prefix,'executable':sys.executable}))"
    run = subprocess.run([str(python), '-I', '-c', code], check=True, capture_output=True, text=True, encoding='utf-8')
    info = json.loads(run.stdout)
    if info['version'][:2] != [3, 12] or info['bits'] != 64 or info['system'] != 'Windows':
        raise RuntimeError('要求 Windows 的 Python 3.12 x64；检测到：' + str(info) + '\n' + PYTHON_HELP)
    return info


def choose_python(explicit=None):
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        python_info(candidate)
        return candidate
    candidates = [ROOT / 'runtime/python/python.exe', Path(sys.executable)]
    launcher = shutil.which('py')
    if launcher:
        run = subprocess.run([launcher, '-3.12', '-c', 'import sys;print(sys.executable)'], capture_output=True, text=True)
        if run.returncode == 0:
            candidates.append(Path(run.stdout.strip()))
    for candidate in candidates:
        if candidate.is_file():
            try:
                python_info(candidate)
                return candidate.resolve()
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError):
                pass
    raise RuntimeError(PYTHON_HELP)


def locked_requirements(root=ROOT):
    rows = []
    for line in (root / 'requirements-lock.txt').read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s;]+)', line)
        if not match:
            raise ValueError('锁文件必须逐项使用 name==version：' + line)
        rows.append(match.groups())
    if len(rows) != 190 or len({name.lower().replace('_', '-') for name, _ in rows}) != 190:
        raise ValueError('本发布版应有190项不重复的锁定依赖；请检查 requirements-lock.txt。')
    return rows


def pip_command(python, action='install', wheelhouse=None, destination=None):
    command = [str(python), '-m', 'pip', '--isolated', '--disable-pip-version-check', action,
               '--no-cache-dir', '--only-binary=:all:']
    if wheelhouse is not None:
        command += ['--no-index', '--find-links', str(wheelhouse)]
    else:
        command += ['--index-url', PYPI, '--extra-index-url', TORCH_INDEX]
    if action == 'download':
        command += ['--no-deps', '--dest', str(destination)]
    command += ['-r', str(ROOT / 'requirements-lock.txt')]
    return command


def child_environment(offline=False, network_log=None):
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PIP_CONFIG_FILE=os.devnull,
               PIP_DISABLE_PIP_VERSION_CHECK='1', HF_HUB_DISABLE_TELEMETRY='1')
    # The learning launcher installs an offline socket guard. Explicit online preparation
    # must not accidentally inherit that guard from a notebook or terminal.
    env.pop('PYTHONHOME', None)
    env.pop('PYTHONPATH', None)
    for key in ['COURSE_OFFLINE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'COURSE_NETWORK_LOG']:
        env.pop(key, None)
    if offline:
        env.update(COURSE_OFFLINE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   PYTHONPATH=str(ROOT / 'tools/offline_guard'))
        if network_log:
            env['COURSE_NETWORK_LOG'] = str(network_log)
    return env


def run_command(command, env=None, dry_run=False):
    print(subprocess.list2cmdline([str(x) for x in command]), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def install_environment(base_python, wheelhouse=None, dry_run=False):
    target = ROOT / '.venv'
    python = target / 'Scripts/python.exe'
    if target.resolve().parent != ROOT.resolve() or target.is_symlink():
        raise RuntimeError('.venv 必须位于课程根目录内，不能指向外部环境。')
    if target.exists():
        if not python.is_file():
            raise RuntimeError('现有 .venv 不完整；安装器不会移动或删除它。请使用 tools/repair_environment.py --rebuild。')
        python_info(python)
        print('复用现有 .venv，按锁定版本补齐依赖；不移动环境。', flush=True)
    else:
        run_command([str(base_python), '-m', 'venv', str(target)], dry_run=dry_run)
    reports = ROOT / 'reports'
    network = reports / 'setup_network.jsonl'
    if not dry_run:
        reports.mkdir(parents=True, exist_ok=True)
        network.write_text('', encoding='utf-8')
    env = child_environment(wheelhouse is not None, network)
    run_command(pip_command(python, wheelhouse=wheelhouse), env=env, dry_run=dry_run)
    run_command([str(python), '-m', 'pip', '--disable-pip-version-check', 'check'], env=env, dry_run=dry_run)
    run_command([str(python), str(ROOT / 'tools/configure_kernel.py')], env=env, dry_run=dry_run)
    if not dry_run and wheelhouse is not None and network.read_text(encoding='utf-8').strip():
        raise RuntimeError('离线安装记录到外部网络尝试，请检查 reports/setup_network.jsonl。')


def main():
    parser = argparse.ArgumentParser(description='首次安装/幂等补齐190项依赖。默认联网；--wheelhouse 仅从指定本地目录安装。')
    parser.add_argument('--python', help='Python 3.12 x64 的 python.exe；缺省查找随包运行时、当前 Python 和 py -3.12')
    parser.add_argument('--wheelhouse', type=Path, help='包含完整190个 wheel 的本地目录；启用 --no-index 和离线网络防护')
    parser.add_argument('--dry-run', action='store_true', help='仅校验前置条件并打印命令，不创建或安装环境')
    args = parser.parse_args()
    started = time.monotonic()
    report = {'passed': False, 'offline': args.wheelhouse is not None}
    try:
        require_windows()
        locked_requirements()
        base = choose_python(args.python)
        if args.wheelhouse is not None:
            args.wheelhouse = args.wheelhouse.expanduser().resolve()
            if not args.wheelhouse.is_dir() or not any(args.wheelhouse.glob('*.whl')):
                raise RuntimeError('本地 wheelhouse 不存在或不含 wheel；先执行在线准备，或者指定完整的外部 wheelhouse。')
        report.update(python=python_info(base), requirements_sha256=sha256(ROOT / 'requirements-lock.txt'))
        install_environment(base, args.wheelhouse, args.dry_run)
        report['passed'] = True
        print('命令预览完成。' if args.dry_run else '环境安装完成。请运行 .venv\\Scripts\\python.exe tools\\self_check.py --core，再按需准备模型。')
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        report['error'] = str(exc)
        print('安装未完成：' + str(exc), file=sys.stderr)
        return 1
    finally:
        if not args.dry_run:
            (ROOT / 'reports').mkdir(parents=True, exist_ok=True)
            report['seconds'] = round(time.monotonic() - started, 2)
            (ROOT / 'reports/setup_environment.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
