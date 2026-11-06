# Compares a full results file cell by cell with the qiyas column of the README Results tables (the 12:38 run of item 3);
# the third file supplies the error column of an earlier full run. Flags SLOWER (> 5 %) and ERR UP.
import sys,io,re
sys.stdout.reconfigure(encoding='utf-8')
readme,new,old=sys.argv[1:4]
base={}
dev=None
for line in io.open(readme,encoding='utf-8'):
    if line.startswith('**CUDA'): dev='cuda'
    elif line.startswith('**CPU'): dev='cpu'
    elif line.startswith('## Open'): break
    elif dev and line.startswith('| ') and not line.startswith('| type') :
        c=[x.strip() for x in line.strip().strip('|').split('|')]
        for op,cell in zip(['add','matmul','softmax','rms_norm'],c[1:]):
            m=re.search(r'\*\*([\d.]+)\*\*',cell)
            if m: base[(dev,op,c[0])]=float(m.group(1))
def rows(p):
    d={}
    for line in io.open(p,encoding='utf-8'):
        if line.startswith('| cpu') or line.startswith('| cuda'):
            c=[x.strip() for x in line.strip().strip('|').split('|')]
            d[(c[0],c[1],c[2])]=c
    return d
N=rows(new); O=rows(old)
print('README base cells',len(base),'new rows',len(N))
slow=0; errup=0
for k,b in sorted(base.items()):
    if k not in N: print('missing',k); continue
    n=float(N[k][12]); r=n/b
    flag=''
    if r>1.05: flag='SLOWER'; slow+=1
    e_new=N[k][15]; e_old=O[k][15] if k in O else '-'
    try:
        if float(e_new)>float(e_old)*1.0000001: flag+=' ERR UP'; errup+=1
    except: pass
    if flag: print(f'{k[0]:4} {k[1]:8} {k[2]:12} {b:9.3f} {n:9.3f} {r:5.2f} err {e_old}/{e_new} {flag}')
print('slower',slow,'err up',errup)
