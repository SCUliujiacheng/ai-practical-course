"""Explicit, reversible offline venv rebuild. A prepared recovery bundle is required."""
from pathlib import Path
import argparse
import datetime
import json
import subprocess
import sys
import time
from setup_environment import ROOT, require_windows, python_info, sha256, install_environment


def verify_bundle():
    manifest_path = ROOT / 'runtime/offline_manifest.json'
    if not manifest_path.is_file():
        raise RuntimeError('缺少离线恢复清单。先在联网环境运行 prepare_offline.py；当前环境不会移动。')
    data = json.loads(manifest_path.read_text(encoding='utf-8'))
    if data.get('wheels') != 190 or data['requirements_sha256'] != sha256(ROOT / 'requirements-lock.txt'):
        raise RuntimeError('恢复包与190项锁文件不一致。')
    for row in data['files']:
        path = (ROOT / row['path']).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not any(path.is_relative_to(p.resolve()) for p in [ROOT / 'runtime/python', ROOT / 'wheelhouse']):
            raise ValueError('恢复清单包含越界路径：' + row['path'])
        if not path.is_file() or path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise ValueError('恢复包缺失或哈希不符：' + row['path'])
    python_info(ROOT / 'runtime/python/python.exe')


def main():
    parser = argparse.ArgumentParser(description='显式 --rebuild 才备份现有.venv并离线重建；失败回滚，不删除原环境。先关闭Jupyter/案例进程。')
    parser.add_argument('--rebuild', action='store_true', help='确认本次操作需要备份并重建.venv')
    parser.add_argument('--dry-run', action='store_true', help='校验恢复包并说明计划，不移动或安装环境')
    args = parser.parse_args()
    if not args.rebuild and not args.dry_run:
        parser.error('请先关闭课程进程，再加 --rebuild；首次安装请使用 setup_environment.py。')
    started = time.monotonic()
    report = {'passed': False, 'offline': True}
    venv = ROOT / '.venv'
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup = ROOT / ('.venv_backup_' + stamp)
    moved = False
    try:
        require_windows()
        verify_bundle()  # all expensive validation happens before moving the working venv
        if venv.resolve().parent != ROOT.resolve() or venv.is_symlink():
            raise RuntimeError('.venv 必须是课程根目录内的真实目录，不能为外部链接。')
        print('将保留原环境为 ' + backup.name + '，使用已校验运行时和wheelhouse离线创建新环境。')
        if args.dry_run:
            return 0
        (ROOT / 'reports').mkdir(parents=True, exist_ok=True)
        if venv.exists():
            venv.rename(backup)
            moved = True
        install_environment(ROOT / 'runtime/python/python.exe', ROOT / 'wheelhouse')
        report.update(passed=True, backup=backup.name if moved else None)
        print('离线重建完成。原环境保留；课程/模型文件未移动。')
        return 0
    except Exception as exc:
        report['error'] = str(exc)
        # Only roll back when this invocation actually moved an existing environment.
        if moved:
            failed = ROOT / ('.venv_failed_' + stamp)
            try:
                if venv.exists():
                    venv.rename(failed)
                backup.rename(venv)
                report['rolled_back'] = True
            except OSError as rollback_error:
                report['rollback_error'] = str(rollback_error)
        print('重建未完成：' + str(exc), file=sys.stderr)
        return 1
    finally:
        if not args.dry_run:
            (ROOT / 'reports').mkdir(parents=True, exist_ok=True)
            report['seconds'] = round(time.monotonic() - started, 2)
            (ROOT / 'reports/repair_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
