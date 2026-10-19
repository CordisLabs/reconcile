#!/usr/bin/env python3
"""Generate docs/index.html from SYLLABUS.md.

The page is derived, never hand-written, so it cannot drift from the syllabus it lists.
Run: python3 scripts/build_site.py
"""

import html
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "https://github.com/CordisLabs/reconcile"

SLICES = {
    "01": "rust/m01_mdl", "02": "rust/m02_mdl_tutorial", "03": "rust/m03_complexodynamics",
    "04": "rust/m04_coffee", "05": "rust/m05_huffman", "06": "rust/m06_msi",
    "07": "python/07_char_rnn", "08": "python/08_lstm", "09": "python/09_rnn_reg",
    "10": "python/10_alexnet", "11": "python/11_resnet", "12": "python/12_resnet_v2",
    "13": "python/13_dilated", "14": "python/14_cs231n", "15": "python/15_bahdanau",
    "16": "python/16_attention", "17": "python/17_annotated_transformer",
    "18": "python/18_order_matters", "19": "python/19_pointer",
    "20": "python/20_relational", "21": "python/21_relational_rnn",
    "22": "python/22_mpnn", "23": "python/23_ntm", "24": "python/24_ctc",
    "25": "python/25_vlae", "26": "python/26_gpipe", "27": "python/27_scaling_laws",
    "28": "python/28_ppo", "29": "python/29_rlhf",
}


def rows():
    out = []
    with open(os.path.join(ROOT, "SYLLABUS.md"), encoding="utf-8") as fh:
        for line in fh:
            m = re.match(r"^\| (\d\d) \| (.+?) \| ([^|]*) \| (.+?) *\|$", line.rstrip())
            if m:
                num, title, year, metric = m.groups()
                out.append((num, title.strip(), year.strip(), metric.strip()))
    return out


def main():
    items = rows()
    body = []
    for num, title, year, metric in items:
        path = SLICES.get(num, "")
        done = "✅" in metric
        metric = metric.replace("|", "\\|").strip()
        body.append(
            f'<tr><td class="n">{num}</td>'
            f'<td><a href="{REPO}/tree/master/{path}">{html.escape(title)}</a>'
            f'<div class="y">{html.escape(year)}</div></td>'
            f'<td>{html.escape(metric)}</td>'
            f'<td class="s">{"完成" if done else "—"}</td></tr>'
        )
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>reconcile · 对账</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font: 16px/1.6 -apple-system, "Segoe UI", "Noto Sans SC", sans-serif;
       max-width: 60rem; margin: 3rem auto; padding: 0 1.2rem; }}
h1 {{ font-size: 1.8rem; margin-bottom: .2rem; }}
p.lead {{ color: #666; margin-top: 0; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 1.6rem; font-size: .94rem; }}
td, th {{ border-bottom: 1px solid #8884; padding: .5rem .4rem; vertical-align: top; }}
th {{ text-align: left; font-weight: 600; }}
td.n {{ font-variant-numeric: tabular-nums; color: #888; width: 2.4rem; }}
td.s {{ white-space: nowrap; }}
td .y {{ color: #888; font-size: .82rem; }}
code {{ font-size: .9em; }}
footer {{ margin-top: 2.5rem; color: #888; font-size: .88rem; }}
</style>
</head>
<body>
<h1>reconcile · 对账</h1>
<p class="lead">29 篇论文的独立实现。每篇一个目录，各自带实现、测试，和一份说明
「论文报告 / 本仓实测 / 差异归因」的 README。</p>
<p><a href="{REPO}">源码</a> · <a href="{REPO}/blob/master/SYLLABUS.md">SYLLABUS</a> ·
<a href="{REPO}/blob/master/START-HERE.md">维护说明</a> · MIT</p>
<table>
<thead><tr><th>#</th><th>论文</th><th>可验证的指标</th><th>状态</th></tr></thead>
<tbody>
{chr(10).join(body)}
</tbody>
</table>
<footer>本页由 <code>scripts/build_site.py</code> 从 <code>SYLLABUS.md</code> 生成。</footer>
</body>
</html>
"""
    out = os.path.join(ROOT, "docs", "index.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"wrote {out} ({len(items)} rows)")


if __name__ == "__main__":
    main()
