"""独立进程运行案例、Notebook与答案；产生机器可读的离线验收记录。"""
from pathlib import Path
import sys,os,json,time,subprocess,argparse,traceback,html,hashlib
from run_case import ROOT,environment

def cases():
    return sorted(list((ROOT/'01_AI基础课').glob('*/case.json'))+list((ROOT/'02_AI进阶课').glob('*/case.json')))
def execute_notebook(path):
    import nbformat
    from nbclient import NotebookClient
    nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
    NotebookClient(nb,timeout=900,kernel_name='python3',resources={'metadata':{'path':str(path.parent)}},allow_errors=False).execute()
    nbformat.write(nb,path)
    # 独立HTML只依赖内联CSS、base64图像，无CDN或在线MathJax。
    import mistune
    blocks=[]
    for cell in nb.cells:
        if cell.cell_type=='markdown':blocks.append(mistune.html(cell.source))
        elif cell.cell_type=='code':
            blocks.append('<pre class="code">'+html.escape(cell.source)+'</pre>')
            for output in cell.get('outputs',[]):
                if output.output_type=='stream':blocks.append('<pre class="output">'+html.escape(output.text)+'</pre>')
                elif output.output_type in ('display_data','execute_result'):
                    data=output.data
                    if 'image/png' in data:blocks.append('<img src="data:image/png;base64,'+data['image/png']+'">')
                    elif 'text/html' in data:blocks.append(data['text/html'])
                    elif 'text/plain' in data:blocks.append('<pre class="output">'+html.escape(data['text/plain'])+'</pre>')
    template='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+html.escape(path.parent.name+' '+path.stem)+'</title><style>body{max-width:1000px;margin:32px auto;padding:0 24px;font:17px/1.8 "Microsoft YaHei",sans-serif;color:#233044;background:#fafbfe}h1,h2,h3{line-height:1.4}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 Consolas,monospace;border-radius:8px;padding:16px}.code{background:#eaf0f8}.output{background:#fff;border-left:4px solid #278477}img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #ccd4dd;padding:6px 10px}a{color:#176792}</style><body><p>离线阅读版：代码与图表来自实际执行。可编辑版本为同目录 Notebook。</p>'+''.join(blocks)+'</body></html>'
    path.with_suffix('.html').write_text(template,encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ids',default='');parser.add_argument('--mode',choices=['scripts','notebooks','all'],default='all');parser.add_argument('--notebook-worker');parser.add_argument('--report',default='offline_validation.json');args=parser.parse_args()
    if args.notebook_worker:execute_notebook(Path(args.notebook_worker));return
    selected=set(args.ids.split(',')) if args.ids else None;reports=[]
    manifests=cases()
    actual_ids={json.loads(p.read_text(encoding='utf-8'))['id'] for p in manifests}
    if selected and selected-actual_ids:raise SystemExit('请求的案例不存在：'+','.join(sorted(selected-actual_ids)))
    report_path=ROOT/'reports'/args.report;report_path.parent.mkdir(exist_ok=True)
    if report_path.exists():reports=json.loads(report_path.read_text(encoding='utf-8')).get('results',[])
    for manifest in manifests:
        meta=json.loads(manifest.read_text(encoding='utf-8'));case_id=meta['id']
        if selected is not None and case_id not in selected:continue
        folder=manifest.parent
        targets=[]
        if args.mode in ['scripts','all']:targets.append(folder/'solution.py')
        if args.mode in ['notebooks','all']:targets += [folder/'教学讲解.ipynb',folder/'参考答案.ipynb',folder/'动手练习.ipynb']
        for target in targets:
            if target.suffix=='.ipynb':
                source=json.dumps([(c['cell_type'],c['source']) for c in json.loads(target.read_text(encoding='utf-8'))['cells']],ensure_ascii=False,sort_keys=True)
            else:source=target.read_text(encoding='utf-8')
            source_hash=hashlib.sha256(source.encode('utf-8')).hexdigest()
            env=environment();network_log=ROOT/'reports'/f'{case_id}_{target.stem}_network.jsonl'
            if network_log.exists():network_log.write_text('',encoding='utf-8')
            env['COURSE_NETWORK_LOG']=str(network_log)
            command=[sys.executable,str(target)] if target.suffix=='.py' else [sys.executable,str(Path(__file__).resolve()),'--notebook-worker',str(target)]
            started=time.monotonic();error=None
            try:
                result=subprocess.run(command,cwd=folder,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=1500)
                output=result.stdout+'\n'+result.stderr
                if result.returncode:error=f'exit {result.returncode}'
            except subprocess.TimeoutExpired as exc:output=str(exc);error='timeout'
            log=ROOT/'reports'/f'{case_id}_{target.stem}.log';log.write_text(output,encoding='utf-8')
            attempted=network_log.exists() and bool(network_log.read_text(encoding='utf-8').strip())
            if attempted:error=(error or '')+' external_network_attempt'
            row={'case':case_id,'file':target.name,'passed':error is None,'error':error,'seconds':round(time.monotonic()-started,2),'log':str(log.relative_to(ROOT)),'external_network_attempt':bool(attempted),'source_sha256':source_hash,'tested_at':time.strftime('%Y-%m-%d %H:%M:%S')}
            reports=[r for r in reports if (r['case'],r['file'])!=(case_id,target.name)]
            reports.append(row)
            report_path.write_text(json.dumps({'started_date':time.strftime('%Y-%m-%d'),'python':sys.version,'mode':args.mode,'network_policy':'Python socket guard; loopback allowed; HF offline; empty HF cache','results':reports},ensure_ascii=False,indent=2),encoding='utf-8')
            print(f'{case_id} {target.name} {"PASS" if row["passed"] else "FAIL"} {row["seconds"]}s',flush=True)
            if error:print(output[-1800:],flush=True)
    if not reports:raise SystemExit('没有找到待验收案例')
    raise SystemExit(0 if all(r['passed'] for r in reports) else 1)
if __name__=='__main__':main()
