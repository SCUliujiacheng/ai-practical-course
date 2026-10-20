"""将课程说明导出为本地浏览器可读HTML；不加载外部字体或脚本。"""
from pathlib import Path
import html
import re
import mistune
from run_case import ROOT

paths = list(ROOT.glob('*.md')) + list((ROOT/'assets').rglob('*.md')) + list((ROOT/'validation').glob('*.md'))
for track in ['01_AI基础课', '02_AI进阶课']:
    paths += list((ROOT / track).rglob('*.md'))
for path in paths:
    body = mistune.html(path.read_text(encoding='utf-8'))
    # Only rewrite local Markdown links whose target is also exported.
    def link(match):
        target = match.group(1)
        if '://' not in target and (path.parent / target.split('#')[0]).is_file():
            return 'href="' + target.replace('.md', '.html') + '"'
        return match.group(0)
    body = re.sub(r'href="([^"#]+\.md(?:#[^"]*)?)"', link, body)
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>' + html.escape(path.stem) + '</title><style>body{max-width:1100px;margin:35px auto;padding:0 24px;font:17px/1.8 "Microsoft YaHei",sans-serif;color:#203445}table{border-collapse:collapse;font-size:15px}td,th{border:1px solid #ccd5dc;padding:8px}pre{background:#eef3f6;white-space:pre-wrap;padding:18px;overflow-wrap:anywhere}a{color:#076d7c}h1,h2,h3{line-height:1.4}blockquote{border-left:4px solid #238080;margin-left:0;padding-left:18px}</style><body>' + body + '</body></html>'
    path.with_suffix('.html').write_text(page, encoding='utf-8')
# Point the navigation home at the rendered instructions too.
home = ROOT / '课程首页.html'
text = home.read_text(encoding='utf-8')
text = re.sub(r'href="([^"#]+\.md)"', lambda m: 'href="' + m.group(1)[:-3] + '.html"', text)
home.write_text(text, encoding='utf-8')
print(f'已导出 {len(paths)} 份中文说明网页。')
