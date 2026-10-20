"""Fetch only pinned, hash-verified model files. Uses only the Python standard library."""
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('minilm', 'flan-t5-small', 'qwen3-0.6b', 'chronos-bolt-tiny', 'resnet18')
RESNET_URL = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_relative_to((root / 'assets/models').resolve()):
        raise ValueError('模型清单路径超出 assets/models：' + relative)
    return path


def matches(path, row):
    return path.is_file() and path.stat().st_size == row['bytes'] and digest(path) == row['sha256']


def model_plan(root=ROOT, names=MODELS):
    manifest = json.loads((root / 'model_files.json').read_text(encoding='utf-8'))
    rows = {row['path']: row for row in manifest}
    if len(rows) != len(manifest):
        raise ValueError('模型清单含重复路径。')
    plans = []
    for name in names:
        if name not in MODELS:
            raise ValueError('未知模型：' + name)
        if name == 'resnet18':
            row = rows['assets/models/resnet18.pth']
            plans.append({**row, 'model': name, 'revision': row['sha256'], 'url': RESNET_URL, 'local_metadata': False})
            continue
        prefix = 'assets/models/' + name + '/'
        source_row = rows[prefix + 'SOURCE.json']
        source_path = checked_path(root, source_row['path'])
        if not matches(source_path, source_row):
            raise ValueError('本地 SOURCE.json 缺失或哈希不符；请从仓库恢复：' + str(source_path))
        source = json.loads(source_path.read_text(encoding='utf-8'))
        revision = source['revision']
        if not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('必须使用完整40位提交revision，不允许latest/main：' + name)
        expected = {prefix + f for f in source['files']} | {prefix + 'SOURCE.json'}
        actual = {p for p in rows if p.startswith(prefix)}
        if actual != expected:
            raise ValueError('SOURCE.files 与哈希清单不一致：' + name)
        for path in sorted(expected):
            row = rows[path]
            checked_path(root, path)
            filename = path[len(prefix):]
            local = filename == 'SOURCE.json'
            url = None if local else 'https://huggingface.co/' + quote(source['repository'], safe='/') + '/resolve/' + revision + '/' + quote(filename, safe='/')
            plans.append({**row, 'model': name, 'revision': revision, 'url': url, 'local_metadata': local})
    return plans


def download_verified(url, destination, row, opener=urlopen):
    """Never replace a destination until the complete temporary file passes its hash."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + '.', suffix='.part', delete=False) as output:
            temporary = Path(output.name)
            request = Request(url, headers={'User-Agent': 'ai-practical-course-pinned-downloader/1.0'})
            with opener(request, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        if not matches(temporary, row):
            raise ValueError('下载文件大小或SHA256不符，保留原文件：' + row['path'])
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description='在线下载固定revision模型，并验证model_files.json全部文件哈希；已有正确文件直接复用。')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--models', default='all', help='all，或逗号分隔：' + ', '.join(MODELS))
    selection.add_argument('--all', action='store_true', help='准备全部五个模型（也是缺省行为）')
    parser.add_argument('--list', action='store_true', help='仅列出固定版本、字节数，不下载')
    parser.add_argument('--verify-only', action='store_true', help='仅校验本地文件，不下载')
    parser.add_argument('--repair', action='store_true', help='允许重新下载已有但校验不符的远端文件；SOURCE元数据仍须从仓库恢复')
    args = parser.parse_args()
    names = MODELS if args.models == 'all' else tuple(dict.fromkeys(x.strip() for x in args.models.split(',') if x.strip()))
    started = time.monotonic()
    results = []
    error = None
    try:
        if not names:
            raise ValueError('至少选择一个模型。')
        plans = model_plan(ROOT, names)
        if args.list:
            for name in names:
                entries = [r for r in plans if r['model'] == name]
                print(name, entries[0]['revision'], sum(r['bytes'] for r in entries), 'bytes')
            return 0
        for row in plans:
            path = checked_path(ROOT, row['path'])
            if matches(path, row):
                status = 'verified_existing'
            elif row['local_metadata']:
                raise ValueError('本地SOURCE元数据校验失败：' + row['path'])
            elif args.verify_only:
                status = 'missing_or_hash_mismatch'
            elif path.exists() and not args.repair:
                raise ValueError('已有文件校验不符；保留文件。使用 --repair 才会下载校验后替换：' + row['path'])
            else:
                print('下载固定版本：' + row['path'], flush=True)
                download_verified(row['url'], path, row)
                status = 'downloaded_and_verified'
            results.append({'path': row['path'], 'status': status, 'sha256': row['sha256']})
            print(status + ': ' + row['path'], flush=True)
        if any(r['status'] == 'missing_or_hash_mismatch' for r in results):
            raise ValueError('选定模型有缺失或校验不符文件；联网时先运行准备命令。')
        return 0
    except Exception as exc:
        error = str(exc)
        print('模型准备未完成：' + error, file=sys.stderr)
        return 1
    finally:
        if not args.list:
            (ROOT / 'reports').mkdir(parents=True, exist_ok=True)
            report = {'passed': error is None, 'models': list(names), 'verify_only': args.verify_only,
                      'seconds': round(time.monotonic() - started, 2), 'files': results, 'error': error}
            (ROOT / 'reports/model_preparation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
