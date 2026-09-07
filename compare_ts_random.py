"""v3 教师 vs v3 学生 (D4): 随机场景行为对比.

200 固定随机场景 (seed 1000+, 与 eval_ckpt 同批) → eval_batch 得逐场景 succ/time;
汇总 + 挑行为差异代表场景 (学生失败 / 时间差大 / 学生反而更快) → run_traj 细跑 + 并排渲染.

输出: 控制台统计; runs/viz_ts_rand<idx>.png (轨迹 + 速度曲线并排)
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P
import viz
from eval_ckpt import load_model
from bench_loops import detect_loops

N = 200

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
    teacher, _ = load_model('runs/kamm533_teacher.pt'); teacher.eval()
    gp, _ = load_model('runs/gp_small_kamm533.pt'); gp.eval()
    S = make_scenes()
    st, tt = P.eval_batch(teacher, False, S)
    sg, tg = P.eval_batch(gp, True, S)

    for name, succ, times in (('teacher', st, tt), ('GP D4', sg, tg)):
        ok = times[succ]
        print(f'{name:>8s}: {int(succ.sum())}/{N} succ  '
              f'mean={ok.mean():.2f}s  p50={np.median(ok):.2f}s  p90={np.quantile(ok, .9):.2f}s  max={ok.max():.2f}s')

    both = st & sg
    dt = tg[both] - tt[both]
    print(f'\n双成功 {int(both.sum())}: Δt(学生-教师)  mean={dt.mean():+.2f}s  med={np.median(dt):+.2f}s  '
          f'worst={dt.max():+.2f}s  best={dt.min():+.2f}s')

    rows = [(i, int(st[i]), float(tt[i]), int(sg[i]), float(tg[i])) for i in range(N)]
    lose = [r for r in rows if r[1] and not r[3]]          # 学生失败 教师成功
    slow = sorted([r for r in rows if r[1] and r[3]], key=lambda r: r[4]-r[2], reverse=True)
    fast = sorted([r for r in rows if r[1] and r[3]], key=lambda r: r[4]-r[2])[:1]
    print(f'\n学生失败而教师成功: {len(lose)} 个 -> {[r[0] for r in lose]}')
    print('学生显著更慢 (双成功, top8):')
    for r in slow[:8]:
        print(f'  scene {r[0]:4d}: teacher {r[2]:5.1f}s | gp {r[4]:5.1f}s | Δ{r[4]-r[2]:+5.1f}s')

    picks = [r[0] for r in lose][:3] + [r[0] for r in slow[:3]] + [r[0] for r in fast]
    seen = set()
    for i in picks:
        if i in seen: continue
        seen.add(i)
        s0, tgts = S[i]
        viz.render_compare(f'random scene #{i} — teacher vs GP-D4', s0, tgts,
                           [('teacher', teacher, False, '#ff6666'), ('GP D4', gp, True, '#44ff44')],
                           f'runs/viz_ts_rand{i}.png')
        for tag, m, ug in (('teacher', teacher, False), ('GP', gp, True)):
            ta, _, gi = viz.run_traj(m, ug, s0.copy(), tgts)
            ev, _ = detect_loops(ta, tgts)
            print(f'  scene {i} {tag}: {gi}/3 {(len(ta)-1)*P.DT:5.1f}s  loops={ev if ev else "-"}')
    print('Done')

if __name__ == '__main__':
    main()
