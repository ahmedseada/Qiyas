"""The README's Results tables from a full `cuda,cpu` results file of the subjects adad and number: one table per device,
a row per type, the four core operations as number / adad / x (x = number / adad); then the rows measured per family.
Run from the repository root:
    python docs/wip/readme_results.py results-qiyas-cuda-cpu.md
and paste the output over the README's Results tables."""
import sys, collections, math
sys.stdout.reconfigure(encoding='utf-8')
path = sys.argv[1]
rows = collections.OrderedDict()
types = []
at = None
for line in open(path, encoding='utf-8'):
    cells = [c.strip() for c in line.strip().strip('|').split('|')]
    # Each subject's ms column by the header's names; the library's is "adad" ("qiyas" in files from before it was Adad).
    if line.startswith('| device'):
        at = {c[:-3]: i for i, c in enumerate(cells) if c.endswith(' ms')}
        extra = {c: i for i, c in enumerate(cells) if c.endswith(' err') or c.endswith(' lane')}
        continue
    if not line.startswith('| c') or at is None: continue
    device, op, typ, shape = cells[:4]
    subj = {name: cells[i] for name, i in at.items()}
    if 'adad' not in subj and 'qiyas' in subj: subj['adad'] = subj['qiyas']
    subj.setdefault('number', '—')
    for name, i in extra.items(): subj[name] = cells[i]
    rows[(device, typ, op)] = subj
    if typ not in types: types.append(typ)
ops = ['add', 'matmul', 'softmax', 'rms_norm']
titles = {'cuda': 'CUDA (2048³ product, 16M-value add, 4096×4096 rows)', 'cpu': 'CPU (1024³ product, 16M-value add, 4096×4096 rows)'}
for device in ('cuda', 'cpu'):
    print(f"**{titles[device]}** — number / **adad** / × (× = number ÷ adad)\n")
    print("| type | add | matmul | softmax | rms_norm |")
    print("|---|---|---|---|---|")
    for typ in types:
        cells = []
        for op in ops:
            r = rows.get((device, typ, op))
            if r is None: cells.append(''); continue
            num, q = r['number'], r['adad']
            try: x = f"{float(num) / float(q):.2f}×"
            except ValueError: x = '—'
            # A float64 cell where number's error is a float32 lane's (above 1e-12 and 100 times Adad's): each subject's
            # error beside its time, so the reader sees why number is faster there.
            def err(name):
                try: return float(r.get(name + ' err', ''))
                except ValueError: return None
            ne, ae = err('number'), err('adad')
            if typ == 'float64' and ne is not None and ae is not None and ne > 1e-12 and ne > 100 * ae:
                cells.append(f"{num} (err {r.get('number err', '?')}) / **{q}** (err {r.get('adad err', '?')}) / {x}")
            else:
                cells.append(f"{num} / **{q}** / {x}")
        print(f"| {typ} | " + " | ".join(cells) + " |")
    print()

# The per-family summary of every row of the run (--ops all): rows measured per device, and against number where it has
# the row: the geometric mean of number ÷ adad and on how many rows adad is ahead.
def family(op):
    if op in ('add', 'matmul', 'softmax', 'rms_norm'): return 'core'
    if op.startswith('map:'): return 'map (element-wise formulas)'
    if op.startswith(('sum:', 'sumsq:', 'max:', 'min:', 'argmax:', 'maprow:')) or op in ('transpose', 'gemv', 'gemv_t'): return 'reductions, broadcasts, transpose, gemv'
    if op.startswith(('row:', 'norm_params_bwd:')): return 'rows'
    if op.startswith('layout:') or op == 'resample': return 'layout and indexing'
    if op.startswith('matmul_'): return 'products on views'
    if op in ('attention_cached', 'cache_write', 'quantize_rows', 'decoder_mask'): return 'caches'
    if op.startswith('attention') or op == 'rope_table': return 'attention'
    if op in ('conv', 'conv_bwd'): return 'convolution'
    if op.startswith('optimizer'): return 'optimizers'
    if op.startswith(('lstm_', 'gru_')): return 'recurrent'
    if op in ('penalize', 'sample_filtered', 'sample_plain', 'sample_rows', 'dropout', 'dropout_residual'): return 'draws and decoding'
    if op.startswith('memory:'): return 'memory'
    if op in ('route_topk', 'expert_matmul', 'expert_gated'): return 'experts'
    if op.startswith('box_loss'): return 'box losses'
    return 'routing, losses'
order = []
stats = collections.defaultdict(lambda: {'rows': 0, 'ops': set(), 'logs': [], 'ahead': 0})
for (device, typ, op), r in rows.items():
    f = family(op)
    if f not in order: order.append(f)
    s = stats[(f, device)]
    try: q = float(r['adad'])
    except ValueError: continue
    s['rows'] += 1; s['ops'].add(op)
    try: num = float(r['number'])
    except ValueError: continue
    if num > 0 and q > 0:
        s['logs'].append(math.log(num / q))
        if num > q: s['ahead'] += 1
def vs(s):
    n = len(s['logs'])
    if n == 0: return '—'
    return f"{math.exp(sum(s['logs']) / n):.2f}× (ahead on {s['ahead']} of {n})"
print("**Every family** — rows measured (operation × type) and, where number has the row, the geometric mean of number ÷ adad\n")
print("| family | operations | CUDA rows | CUDA vs number | CPU rows | CPU vs number |")
print("|---|---:|---:|---|---:|---|")
for f in order:
    c, p = stats[(f, 'cuda')], stats[(f, 'cpu')]
    print(f"| {f} | {len(c['ops'] | p['ops'])} | {c['rows']} | {vs(c)} | {p['rows']} | {vs(p)} |")
print()
