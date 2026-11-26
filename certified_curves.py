#!/usr/bin/env python3
"""Standard certification metrics from Cohen-style CERTIFY logs.

Usage:
  python certified_curves.py --sigma 0.25 --std std_seed1.tsv std_seed2.tsv ... \
      --pur pur_seed1.tsv pur_seed2.tsv ... --N 1000 --alpha 0.001 --out curves_sigma0.25

Each log is tab-separated with the header written by Cohen et al.'s certify.py:
  idx  label  predict  radius  correct  time
`predict == -1` means ABSTAIN. Files for the two methods must be given in the
same seed order and certified on the same images, so the paired test is valid.
Outputs: <out>.csv (a(r), ACR, abstention per seed and mean/std), <out>.png (curves),
and a paired McNemar test on certified correctness at r = 0 and at each radius in --radii.
"""
import argparse, csv, math
import numpy as np
from scipy.stats import norm, binomtest
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load(path):
    rows = list(csv.DictReader(open(path), delimiter='\t'))
    idx = np.array([int(r['idx']) for r in rows])
    pred = np.array([int(r['predict']) for r in rows])
    rad = np.array([float(r['radius']) for r in rows])
    cor = np.array([int(r['correct']) for r in rows])
    ok = (cor == 1) & (pred != -1)
    return idx, ok, np.where(ok, rad, 0.0), pred == -1

def acc_at(ok, rad, r):  # certified accuracy at radius r (a(r)); r = 0 counts every correct non-abstained image
    return float(np.mean(ok & (rad >= r))) if r > 0 else float(np.mean(ok))

def mcnemar(a, b):
    n10 = int(np.sum(a & ~b)); n01 = int(np.sum(~a & b))
    n = n10 + n01
    return n10, n01, (1.0 if n == 0 else binomtest(min(n10, n01), n, 0.5).pvalue)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sigma', type=float, required=True)
    ap.add_argument('--std', nargs='+', required=True)
    ap.add_argument('--pur', nargs='+', required=True)
    ap.add_argument('--N', type=int, default=1000)
    ap.add_argument('--alpha', type=float, default=0.001)
    ap.add_argument('--radii', type=float, nargs='+', default=[0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument('--out', default='curves')
    a = ap.parse_args()
    rmax = a.sigma * norm.ppf(a.alpha ** (1.0 / a.N))
    grid = np.linspace(0, max(rmax * 1.05, 1e-6), 200)
    res = {}
    for name, files in (('Standard RS', a.std), ('Diffusion-Purified', a.pur)):
        per = []
        for f in files:
            idx, ok, rad, abst = load(f)
            per.append(dict(idx=idx, ok=ok, rad=rad, abst=abst))
        res[name] = per
    # table
    with open(a.out + '.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['method', 'metric', 'mean', 'std'] + ['seed%d' % (i + 1) for i in range(len(a.std))])
        for name, per in res.items():
            for r in a.radii:
                v = [acc_at(p['ok'], p['rad'], r) for p in per]
                w.writerow([name, 'a(%.2f)' % r, np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0] + v)
            v = [float(np.mean(p['rad'])) for p in per]
            w.writerow([name, 'ACR', np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0] + v)
            v = [float(np.mean(p['abst'])) for p in per]
            w.writerow([name, 'abstention rate', np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0] + v)
        w.writerow([]); w.writerow(['R_max (N=%d, alpha=%g)' % (a.N, a.alpha), rmax])
    # paired McNemar per radius, pooled over seeds (same images per seed assumed)
    print('Paired McNemar test (Standard RS vs Diffusion-Purified), pooled over seeds')
    for r in a.radii:
        A = np.concatenate([acc_ok(p, r) for p in res['Standard RS']])
        B = np.concatenate([acc_ok(p, r) for p in res['Diffusion-Purified']])
        n10, n01, pv = mcnemar(A, B)
        print('  r=%.2f  std-only correct=%d  purified-only correct=%d  p=%.3g' % (r, n10, n01, pv))
    # curves
    plt.figure(figsize=(5, 3.6), dpi=200)
    for name, col in (('Standard RS', '#1f77b4'), ('Diffusion-Purified', '#d62728')):
        curves = np.array([[acc_at(p['ok'], p['rad'], r) for r in grid] for p in res[name]])
        m = curves.mean(0); s = curves.std(0, ddof=1) if len(curves) > 1 else np.zeros_like(m)
        plt.plot(grid, m, color=col, label=name); plt.fill_between(grid, m - s, m + s, color=col, alpha=.2)
    plt.axvline(rmax, ls='--', color='gray', lw=1); plt.text(rmax, 0.02, ' R_max', fontsize=7, color='gray')
    plt.xlabel('ℓ2 radius r'); plt.ylabel('Certified accuracy a(r)'); plt.title('σ = %.2f, N = %d' % (a.sigma, a.N), fontsize=9)
    plt.legend(fontsize=7); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(a.out + '.png')

def acc_ok(p, r):
    return p['ok'] & (p['rad'] >= r) if r > 0 else p['ok']

if __name__ == '__main__':
    main()
