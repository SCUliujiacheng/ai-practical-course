"""Prepare a Windows-only recovery bundle while leaving the working .venv intact."""
from pathlib import Path
from email.parser import Parser
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from setup_environment import (ROOT, require_windows, python_info, locked_requirements,
                               pip_command, child_environment, run_command, sha256)


def normalize(name):
    return name.lower().replace('_', '-').replace('.', '-')


def select_wheels(folder, requirements):
    wanted = {normalize(name): version for name, version in requirements}
    selected = {}
    for wheel in sorted(folder.glob('*.whl')):
        with zipfile.ZipFile(wheel) as archive:
            metadata_paths = [name for name in archive.namelist() if name.count('/') == 1 and name.endswith('.dist-info/METADATA')]
            if len(metadata_paths) != 1:
                raise ValueError('wheel元数据无效：' + str(wheel))
            metadata = Parser().parsestr(archive.read(metadata_paths[0]).decode('utf-8'))
        name = normalize(metadata['Name'])
        if wanted.get(name) == metadata['Version']:
            if name in selected:
                raise ValueError('同一个锁定依赖有多个wheel，无法唯一选择：' + name)
            selected[name] = wheel
    missing = sorted(set(wanted) - set(selected))
    if missing:
        raise ValueError('wheelhouse缺少锁定版本：' + ', '.join(missing))
    return list(selected.values())


def copy_checked(source, target):
    source, target = Path(source), Path(target)
    source_hash = sha256(source)
    if target.is_file() and sha256(target) == source_hash:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.copying', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copy2(source, temporary)
        if sha256(temporary) != source_hash:
            raise ValueError('复制后哈希不符：' + str(target))
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def copy_runtime(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if source == destination:
        python_info(destination / 'python.exe')
        return
    def ignored(folder, names):
        skip = {name for name in names if name == '__pycache__' or name.endswith(('.pyc', '.pyo'))}
        if Path(folder).resolve() == (source / 'Lib').resolve():
            skip.add('site-packages')
        return skip
    if destination.exists():
        for folder, dirs, filenames in os.walk(source):
            skip = ignored(folder, dirs + filenames)
            dirs[:] = [name for name in dirs if name not in skip]
            for filename in filenames:
                if filename in skip:
                    continue
                original = Path(folder) / filename
                copied = destination / original.relative_to(source)
                if not copied.is_file() or sha256(copied) != sha256(original):
                    raise RuntimeError('已有runtime与当前基础运行时不一致，保留目录并停止：' + str(copied))
        python_info(destination / 'python.exe')
        print('已有runtime与基础运行时逐文件一致，直接复用。', flush=True)
        return
    if destination.is_relative_to(source):
        raise RuntimeError('课程目录位于Python安装目录内部，拒绝递归复制。请把课程放到独立目录。')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='python-preparing-', dir=destination.parent))
    # The temporary directory is new, confined to runtime/, and never contains the venv.
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True, ignore=ignored)
        python_info(temporary / 'python.exe')
        subprocess.run([str(temporary / 'python.exe'), '-I', '-c', 'import venv, ensurepip, ssl'], check=True)
        temporary.rename(destination)
    except Exception:
        print('运行时复制未完成，工作环境未改动；诊断文件保留在：' + str(temporary), flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description='先在线准备离线恢复包：复制Python运行时，并准备190个锁定wheel；不修改当前.venv。')
    parser.add_argument('--wheelhouse-source', type=Path, help='从已有外部wheelhouse复制190项；指定后不访问包索引')
    parser.add_argument('--dry-run', action='store_true', help='只校验前置条件与打印计划，不复制、不下载')
    args = parser.parse_args()
    started = time.monotonic()
    report = {'passed': False, 'working_venv_untouched': True}
    try:
        require_windows()
        if Path(sys.prefix).resolve() != (ROOT / '.venv').resolve():
            raise RuntimeError('请先完成安装，再用课程虚拟环境运行：.venv\\Scripts\\python.exe tools\\prepare_offline.py。系统/基础Python可能受PEP668管理；本工具不修改系统环境，也不需要break-system-packages。')
        info = python_info(sys.executable)
        requirements = locked_requirements()
        base = Path(info['base_prefix'])
        runtime = ROOT / 'runtime/python'
        wheelhouse = ROOT / 'wheelhouse'
        if not runtime.resolve().is_relative_to(ROOT.resolve()) or not wheelhouse.resolve().is_relative_to(ROOT.resolve()):
            raise RuntimeError('runtime与wheelhouse必须位于课程目录内，不能指向外部目录。')
        source = args.wheelhouse_source.expanduser().resolve() if args.wheelhouse_source else None
        wheels = select_wheels(source, requirements) if source else None
        print('准备恢复运行时：' + str(base) + ' → ' + str(runtime), flush=True)
        if args.dry_run:
            if source:
                print('复制已检查的190项wheel：' + str(source) + ' → ' + str(wheelhouse))
            else:
                run_command(pip_command(sys.executable, action='download', destination=wheelhouse), dry_run=True)
            print('随后将以 --dry-run --ignore-installed --no-index 检查wheel兼容性；当前.venv保持不动。')
            return 0
        (ROOT / 'reports').mkdir(parents=True, exist_ok=True)
        wheelhouse.mkdir(parents=True, exist_ok=True)
        copy_runtime(base, runtime)
        if wheels is not None:
            for wheel in wheels:
                copy_checked(wheel, wheelhouse / wheel.name)
        else:
            run_command(pip_command(sys.executable, action='download', destination=wheelhouse), env=child_environment())
        selected = select_wheels(wheelhouse, requirements)
        # Resolving against all packages even in a populated venv checks both wheel
        # compatibility and dependency completeness without installing anything.
        command = pip_command(sys.executable, wheelhouse=wheelhouse)
        command += ['--dry-run', '--ignore-installed']
        network = ROOT / 'reports/offline_preparation_network.jsonl'
        network.write_text('', encoding='utf-8')
        run_command(command, env=child_environment(True, network))
        if network.read_text(encoding='utf-8').strip():
            raise RuntimeError('本地wheel检查时记录到外部网络尝试。')
        files = [p for p in runtime.rglob('*') if p.is_file()] + selected
        manifest = [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in sorted(files)]
        bundle = {'platform': 'Windows x64', 'python_version': info['version'],
                  'requirements_sha256': sha256(ROOT / 'requirements-lock.txt'), 'wheels': len(selected), 'files': manifest}
        (ROOT / 'runtime/offline_manifest.json').write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding='utf-8')
        report.update(passed=True, wheel_count=len(selected), runtime_files=len(files) - len(selected),
                      manifest='runtime/offline_manifest.json')
        print('Python与190项wheel已准备并检查。模型需另运行 prepare_models.py --all；全部完成后才能称为完整离线包。')
        return 0
    except Exception as exc:
        report['error'] = str(exc)
        print('离线准备未完成：' + str(exc), file=sys.stderr)
        return 1
    finally:
        if not args.dry_run:
            (ROOT / 'reports').mkdir(parents=True, exist_ok=True)
            report['seconds'] = round(time.monotonic() - started, 2)
            (ROOT / 'reports/offline_preparation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
