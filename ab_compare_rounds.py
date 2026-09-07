"""容量 A/B — 逐轮对齐对比: 不选最优轮, 直接比 BC + DAgger 各轮同编号结果.

每轮载入 runs/gp_small_kamm533_d{n}.pt (3.7K, v3 实存) 与 runs/ab_gpmed_d{n}.pt (9K),
同一 200 随机批 (seed 1000+) 逐轮 eval → 表 + 曲线图.

用法: 需先完成 distill_ab_size.py (生成 ab_gpmed_d{n}.pt).
输出: runs/viz_ab_rounds.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P
from eval_ckpt import load_model

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
ROUNDS = ['BC'] + [f'D{i}' for i in range(1, 13)]
FILES = {
    'GP-Small 3.7K': lambda r: 'runs/gp_small_kamm533_bc.pt' if r == 'BC' else f'runs/gp_small_kamm533_{r.lower()}.pt',
    'GP-Medium 9K':  lambda r: 'runs/ab_gpmed_bc.pt'          if r == 'BC' else f'runs/ab_gpmed_{r.lower()}.pt',
}

def make_scenes():
    out = []
    for i in range(N):
        rng = np.random.RandomState(1000 + i)
        s0 = np.array([rng.uniform(P.M, P.FLD-P.M), rng.uniform(P.M, P.FLD-P.M),
                       rng.uniform(-np.pi, np.pi), rng.uniform(0.5, P.V_MAX),
                       rng.uniform(-P.DELTA_MAX, P.DELTA_MAX)], dtype=np.float32)
        tg = [np.array([rng.uniform(P.M, P.FLD-P.M), rng.uniform(P.M, P.FLD-P.M)], dtype=np.float32)
              for _ in range(3)]
        out.append((s0, tg))
    return out

def main():
    S = make_scenes()
    teacher, _ = load_model('runs/kamm533_teacher.pt'); teacher.eval()
    succ_t, time_t = P.eval_batch(teacher, False, S)
    ok = time_t[succ_t]
    print(f'{"round":>5s} | {"GP-Small 3.7K":^22s} | {"GP-Medium 9K":^22s}')
    print(f'{"":>5s} | {"succ":>6s} {"mean":>7s} {"p90":>6s} | {"succ":>6s} {"mean":>7s} {"p90":>6s}')
    print(f'{"teacher":>7s} | {int(succ_t.sum()):>6d} {ok.mean():>7.2f} {np.quantile(ok,.9):>6.2f} |')
    rows = {'GP-Small 3.7K': [], 'GP-Medium 9K': []}
    for r in ROUNDS:
        line = [f'{r:>7s}']
        for name, f in FILES.items():
            if not os.path.exists(f(r)):
                line += ['(skip)'] * 4
                rows[name].append(None); continue
            m, _ = load_model(f(r)); m.eval()
            succ, times = P.eval_batch(m, True, S)
            ok_ = times[succ]
            s_ok = int(succ.sum()); mean = ok_.mean() if len(ok_) else float('nan')
            p90 = np.quantile(ok_, .9) if len(ok_) else float('nan')
            rows[name].append((s_ok, mean, p90))
            line += [f'{s_ok:>6d} {mean:>7.2f} {p90:>6.2f}']
        print(' | '.join(line))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor='#0d0d0d')
    colors = {'GP-Small 3.7K': '#44ff44', 'GP-Medium 9K': '#66ccff'}
    xs = np.arange(len(ROUNDS))
    ax1.axhline(int(succ_t.sum()), color='#ff6666', ls='--', lw=1, label='teacher')
    for name in rows:
        v = [r[0] if r else np.nan for r in rows[name]]
        ax1.plot(xs, v, marker='o', color=colors[name], lw=1.6, label=name)
        best_i = int(np.nanargmax(v))
        ax1.annotate(f"best={v[best_i]}/{N}@{ROUNDS[best_i]}", (xs[best_i], v[best_i]),
                     textcoords='offset points', xytext=(6, -14), fontsize=8, color=colors[name])
    ax1.set_xticks(xs, ROUNDS, rotation=45, fontsize=7)
    ax1.set_ylim(150, 201); ax1.set_ylabel('succ / 200'); ax1.legend(fontsize=8)
    ax1.set_title('same-round succ (random 200)', color='white', fontsize=10)
    ax1.tick_params(colors='#999999'); ax1.set_facecolor('#1a1a1a')
    for s in ('top', 'right'): ax1.spines[s].set_visible(False)

    ax2.axhline(ok.mean(), color='#ff6666', ls='--', lw=1, label='teacher')
    for name in rows:
        v = [r[1] if r else np.nan for r in rows[name]]
        ax2.plot(xs, v, marker='o', color=colors[name], lw=1.6, label=name)
    ax2.set_xticks(xs, ROUNDS, rotation=45, fontsize=7)
    ax2.set_ylabel('mean time (s)'); ax2.legend(fontsize=8)
    ax2.set_title('same-round mean time (succ only)', color='white', fontsize=10)
    ax2.tick_params(colors='#999999'); ax2.set_facecolor('#1a1a1a')
    for s in ('top', 'right'): ax2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig('runs/viz_ab_rounds.png', dpi=130, facecolor='#0d0d0d')
    print('  -> runs/viz_ab_rounds.png')

if __name__ == '__main__':
    main()
