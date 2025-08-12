import re, glob, os
import numpy as np

def parse_summary(text):
    rows = []
    pat = re.compile(r'^(\S+)\s+baseline=\s*([0-9\.E+ -]+)\s+A=\s*([0-9\.E+ -]+)\s+B=\s*([0-9\.E+ -]+)\s+best=(\S+)$')
    for line in text.splitlines():
        m = pat.match(line.strip())
        if m:
            name = m.group(1)
            base = float(m.group(2))
            A = float(m.group(3))
            B = float(m.group(4))
            best = m.group(5)
            rows.append((name, base, A, B, best))
    return rows

def read_rotmod_components(name):
    path = None
    for p in glob.glob(f"{name}_rotmod.dat"):
        path = p; break
    if path is None:
        return None
    arr = []
    with open(path,'r') as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in '#;': continue
            parts = s.replace(',',' ').split()
            try:
                vals = [float(x) for x in parts]
                arr.append(vals)
            except:
                pass
    if not arr: return None
    A = np.array(arr, float)
    r = A[:,0]; vobs=A[:,1]; dv=A[:,2]
    vgas = A[:,3] if A.shape[1] > 3 else None
    vdisk= A[:,4] if A.shape[1] > 4 else None
    vbulg= A[:,5] if A.shape[1] > 5 else None
    def quad(*vs):
        acc=None
        for v in vs:
            if v is None: continue
            vv=np.asarray(v); acc = vv**2 if acc is None else acc+vv**2
        return np.sqrt(acc) if acc is not None else None
    vstar = quad(vdisk, vbulg)
    vbary = quad(vgas, vdisk, vbulg)
    vmax = float(np.nanmax(vbary)) if vbary is not None else np.nan
    # Fractions at mid radii
    mid = slice(len(r)//3, 2*len(r)//3 or None)
    vg = np.nanmedian(vgas[mid]) if vgas is not None else 0.0
    vs = np.nanmedian(vstar[mid]) if vstar is not None else 0.0
    vb = np.nanmedian(vbary[mid]) if vbary is not None else 1.0
    frac_g = (vg**2)/(vb**2) if vb>0 else np.nan
    frac_s = (vs**2)/(vb**2) if vb>0 else np.nan
    return dict(vmax=vmax, frac_g=frac_g, frac_s=frac_s)

def classify(meta):
    vmax = meta['vmax']
    fg = meta['frac_g']; fs = meta['frac_s']
    if np.isnan(vmax): return 'Unknown'
    if vmax < 80: return 'Dwarf'
    # Gas-rich vs stellar-dominated
    if fg >= 0.6: return 'Gas-rich (LSB)'
    if fs >= 0.6: return 'Stellar-dominated (HSB)'
    return 'Mixed'

def summarize(rows):
    # rows: (name, base, A, B, best)
    out = {}
    for name, base, A, B, best in rows:
        meta = read_rotmod_components(name)
        cls = classify(meta) if meta else 'Unknown'
        ratio = (A/base) if base>0 else np.nan
        d = out.setdefault(cls, {'names':[], 'ratios':[], 'wins':0})
        d['names'].append(name)
        d['ratios'].append(ratio)
        d['wins'] += 1 if best=='A' else 0
    return out

def print_table(out):
    print("Class                         N   Win_A   Median(A/base)   Mean(A/base)")
    for cls, d in out.items():
        N = len(d['names'])
        wins = d['wins']
        ratios = np.array(d['ratios'], float)
        med = np.nanmedian(ratios)
        mean = np.nanmean(ratios)
        print(f"{cls:28s} {N:4d}  {wins:6d}     {med:8.3f}         {mean:8.3f}")

if __name__ == "__main__":
    # Paste your summary block between the triple quotes:
    summary_text = open("blind_summary.txt","r").read() if os.path.exists("blind_summary.txt") else ""
    if not summary_text:
        print("Put your summary lines into blind_summary.txt and rerun.")
    else:
        rows = parse_summary(summary_text)
        out = summarize(rows)
        print_table(out)