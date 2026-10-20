"""Check the published course; --core works before optional model downloads."""
from pathlib import Path
import argparse
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from setup_environment import ROOT, locked_requirements, python_info, sha256
from prepare_models import model_plan, checked_path, matches

DATA_HASHES = {
    'bike_day.csv': 'a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be',
    'bike_hour.csv': 'e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f',
    'bike_README.txt': 'b92c8628622948bf0828c43a1b315d1517c3d7fd63654730d0dea379b8e88175',
    'jena_daily.csv': 'ba74bd0d6f0c7f5a372c4de4f667313107b40d28d8189ce6952da1b29ee4e410',
    'sms_README.txt': '8753cd2d3cab68f80c8257851b8c2037778c267f55accb6fd1b5a2e32d36a84e',
    'sms.tsv': 'c33272435ad0f516dbc836f4d1b0d734ffb975182173fa0a35353676a421ace6',
}
MODULES = ['numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'torch', 'torchvision',
           'transformers', 'sentence_transformers', 'jupyterlab', 'nbformat', 'nbclient',
           'ortools', 'networkx', 'simpy', 'nashpy', 'dowhy', 'mapie', 'ax', 'gymnasium',
           'stable_baselines3', 'chronos', 'fastapi', 'httpx']


def asset_failures(root=ROOT, core=False, full_hash=False):
    failures = []
    for filename, expected in DATA_HASHES.items():
        path = root / 'assets/data' / filename
        if not path.is_file():
            failures.append('缺少随仓库提供的数据：' + filename)
        elif full_hash and sha256(path) != expected:
            failures.append('数据哈希不符：' + filename)
    for track in ['01_AI基础课', '02_AI进阶课']:
        count = len(list((root / track).glob('*/case.json')))
        if count != 30:
            failures.append(f'{track}案例数为{count}，预期30。')
    if not core:
        try:
            for row in model_plan(root):
                path = checked_path(root, row['path'])
                good = matches(path, row) if full_hash else path.is_file() and path.stat().st_size == row['bytes']
                if not good:
                    failures.append('模型缺失或校验不符：' + row['path'])
        except Exception as exc:
            failures.append('模型清单无效：' + str(exc))
    return failures


def main():
    parser = argparse.ArgumentParser(description='检查依赖、60案例和素材；不会安装或下载。首次安装后使用 --core。')
    parser.add_argument('--core', action='store_true', help='检查完整依赖与小数据，不要求尚未下载的五个模型')
    parser.add_argument('--full-hash', action='store_true', help='对模型和小数据做SHA256校验；与--core组合时只校验小数据')
    parser.add_argument('--assets-only', action='store_true', help='仅检查案例/素材，跳过依赖与Python运行环境；不能证明案例可运行')
    args = parser.parse_args()
    (ROOT / 'reports').mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                      OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MPLBACKEND='Agg')
    failures = asset_failures(core=args.core, full_hash=args.full_hash)
    records = []
    if not args.assets_only:
        try:
            python_info(sys.executable)
        except Exception as exc:
            failures.append(str(exc))
        try:
            for package, expected in locked_requirements():
                try:
                    actual = importlib.metadata.version(package)
                    if actual != expected:
                        failures.append(f'依赖版本不符：{package}={actual}，锁定为{expected}')
                except importlib.metadata.PackageNotFoundError:
                    failures.append('缺少依赖：' + package)
        except Exception as exc:
            failures.append('锁文件错误：' + str(exc))
        for module in MODULES:
            try:
                importlib.import_module(module)
                records.append({'module': module, 'ok': True})
            except Exception as exc:
                records.append({'module': module, 'ok': False, 'error': str(exc)})
                failures.append(f'{module}: {exc}')
        result = subprocess.run([sys.executable, '-m', 'pip', '--disable-pip-version-check', 'check'],
                                capture_output=True, text=True, encoding='utf-8', errors='replace')
        if result.returncode:
            failures.append('pip check：' + result.stdout + result.stderr)
    report = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'passed': not failures,
              'python': sys.version, 'executable': sys.executable, 'imports': records,
              'failures': failures, 'core': args.core, 'full_hash': args.full_hash,
              'assets_only': args.assets_only, 'models_checked': not args.core}
    path = ROOT / 'reports/self_check.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if failures:
        print('\n'.join(failures))
        if not args.core:
            print('尚未下载模型时先用 --core；准备完整模型后再做全量检查。')
    else:
        print('所选范围检查通过。' + ('仅素材检查，未验证依赖或运行环境。' if args.assets_only else '核心检查通过；可选模型未检查。' if args.core else '依赖、模型、小数据与60个案例齐全。'))
    print('报告：' + str(path))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
