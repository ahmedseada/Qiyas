"""The README's Results tables from a full `cuda,cpu` results file: one table per device, a row per type, the four core
operations as lib / number / qiyas / x (x = lib / qiyas). Run from the repository root:
    python docs/wip/readme_results.py results-qiyas-cuda-cpu.md
and paste the output over the two tables in README.md; the per-family summary table follows them (every row of an
--ops all run)."""
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

# The per-family summary of every row of the run (--ops all): rows measured per device, and against the lib where it has
# the row: the geometric mean of lib ÷ qiyas and on how many rows qiyas is ahead.
import math
def family(op):
    if op in ('add', 'matmul', 'softmax', 'rms_norm'): return 'core'
    if op.startswith('map:'): return 'map (element-wise formulas)'
    if op.startswith(('sum:', 'sumsq:', 'max:', 'min:', 'argmax:', 'maprow:')) or op in ('transpose', 'gemv', 'gemv_t'): return 'reductions, broadcasts, transpose, gemv'
    if op.startswith(('row:', 'norm_params_bwd:')): return 'rows'
    if op.startswith('layout:'): return 'layout and indexing'
    if op.startswith('matmul_'): return 'products on views'
    if op.startswith('attention'): return 'attention'
    if op in ('conv', 'conv_bwd'): return 'convolution'
    if op.startswith('optimizer:'): return 'optimizers'
    if op.startswith(('lstm_', 'gru_')): return 'recurrent'
    return 'routing, losses'
order = []
stats = collections.defaultdict(lambda: {'rows': 0, 'ops': set(), 'logs': [], 'ahead': 0})
for (device, typ, op), r in rows.items():
    f = family(op)
    if f not in order: order.append(f)
    s = stats[(f, device)]
    try: q = float(r['qiyas'])
    except ValueError: continue
    s['rows'] += 1; s['ops'].add(op)
    try: lib = float(r['lib'])
    except ValueError: continue
    if lib > 0 and q > 0:
        s['logs'].append(math.log(lib / q))
        if lib > q: s['ahead'] += 1
def vs(s):
    n = len(s['logs'])
    if n == 0: return '—'
    return f"{math.exp(sum(s['logs']) / n):.2f}× (ahead on {s['ahead']} of {n})"
print("**Every family** — rows measured (operation × type) and, where the lib has the row, the geometric mean of lib ÷ qiyas\n")
print("| family | operations | CUDA rows | CUDA vs lib | CPU rows | CPU vs lib |")
print("|---|---:|---:|---|---:|---|")
for f in order:
    c, p = stats[(f, 'cuda')], stats[(f, 'cpu')]
    print(f"| {f} | {len(c['ops'] | p['ops'])} | {c['rows']} | {vs(c)} | {p['rows']} | {vs(p)} |")
print()
