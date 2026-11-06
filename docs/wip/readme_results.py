"""The README's Results tables from a full `cuda,cpu` results file: one table per device, a row per type, the four core
operations as lib / number / qiyas / x (x = lib / qiyas). Run from the repository root:
    python docs/wip/readme_results.py results-qiyas-cuda-cpu.md
and paste the output over the two tables in README.md."""
import sys, collections
sys.stdout.reconfigure(encoding='utf-8')
path = sys.argv[1]
rows = collections.OrderedDict()
types = []
for line in open(path, encoding='utf-8'):
    if not line.startswith('| c'): continue
    cells = [c.strip() for c in line.strip().strip('|').split('|')]
    device, op, typ, shape = cells[:4]
    rest = cells[4:]
    subj = {}
    for i, name in enumerate(['lib', 'number', 'qiyas']):
        ms, x, lane, err = rest[4 * i: 4 * i + 4]
        subj[name] = ms
    rows[(device, typ, op)] = subj
    if typ not in types: types.append(typ)
ops = ['add', 'matmul', 'softmax', 'rms_norm']
titles = {'cuda': 'CUDA (2048³ product, 16M-value add, 4096×4096 rows)', 'cpu': 'CPU (1024³ product, 16M-value add, 4096×4096 rows)'}
for device in ('cuda', 'cpu'):
    print(f"**{titles[device]}** — lib / number / **qiyas** / × (× = lib ÷ qiyas)\n")
    print("| type | add | matmul | softmax | rms_norm |")
    print("|---|---|---|---|---|")
    for typ in types:
        cells = []
        for op in ops:
            r = rows.get((device, typ, op))
            if r is None: cells.append(''); continue
            lib, num, q = r['lib'], r['number'], r['qiyas']
            try: x = f"{float(lib) / float(q):.2f}×"
            except ValueError: x = '—'
            cells.append(f"{lib} / {num} / **{q}** / {x}")
        print(f"| {typ} | " + " | ".join(cells) + " |")
    print()
