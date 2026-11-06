import sys, re
base, new = sys.argv[1], sys.argv[2]

def parse(path):
    # Each subject's columns (ms, x, lane, err) are found by the header's names ("qiyas ms", ...), so a run of some
    # subjects only (--subjects qiyas) compares with a run of all of them.
    rows = {}
    at = None
    for line in open(path, encoding='utf-8'):
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if line.startswith('| device'):
            at = {c[:-3]: i for i, c in enumerate(cells) if c.endswith(' ms')}
            continue
        if not line.startswith('| c') or at is None: continue
        device, op, typ = cells[:3]
        rows[(device, op, typ)] = {name: tuple(cells[i:i + 4]) for name, i in at.items()}
    return rows

b, n = parse(base), parse(new)
print(f"{'cell':34} {'base ms':>9} {'new ms':>9} {'ratio':>7}  lanes  err base/new")
regress = []
for key in n:
    if key not in b: print("new row", key); continue
    bq, nq = b[key]['qiyas'], n[key]['qiyas']
    try:
        bm, nm = float(bq[0]), float(nq[0])
    except ValueError:
        continue
    ratio = nm / bm
    flag = ''
    if ratio > 1.05: flag = '  SLOWER'; regress.append(key)
    be, ne = bq[3], nq[3]
    try:
        if float(ne.replace('e+', 'e')) > float(be.replace('e+', 'e')) * 1.0001: flag += '  ERR UP'
    except ValueError:
        pass
    print(f"{' '.join(key):34} {bm:9.3f} {nm:9.3f} {ratio:7.2f}  {bq[2]}->{nq[2]}  {be}/{ne}{flag}")
print("slower cells:", len(regress))
