"""长距探针: 验证"远端目标不退化" (v4 删长腿带的对冲检查).

三类远端构型 (v0∈[3,5] 全速起步, g1 3-6m):
  A 直行远: g1 正前 3-6m, g2/g3 续直 — 全速逼近+终点刹车
  B 远后急弯: g1 远, g2 大转角短距 — 提前为过门后的急弯减速
  C 远侧向: g1 3-6m 大侧角 — 全速下先减速再转向
指标: succ / mean time / v@d1=1m (越过 1m 圈时的速度 = 提前减速程度) / path ratio.

用法: python probe_long.py [checkpoint ...]    # 默认 v4 教师 + v1 参照 (若存在)
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P
import viz
from eval_ckpt import load_model
from bench_loops import detect_loops

C, Y = P.CAR_X, P.CAR_Y
N_PER = 40

def make_scenes(kind, i):
    rng = np.random.RandomState(9000 + kind*100 + i)
    th0 = rng.uniform(-np.pi, np.pi)
    v0 = rng.uniform(3.0, P.V_MAX)
    dl = rng.uniform(3.0, 6.0)
    s0 = np.array([C, Y, th0, v0, rng.uniform(-P.DELTA_MAX, P.DELTA_MAX)], dtype=np.float32)
    g1 = np.array([C + dl*np.cos(th0), Y + dl*np.sin(th0)], dtype=np.float32)
    if kind == 0:      # 直行远
        g2 = g1 + np.array([0.8*np.cos(th0), 0.8*np.sin(th0)], dtype=np.float32)
        g3 = g2 + np.array([0.8*np.cos(th0), 0.8*np.sin(th0)], dtype=np.float32)
    elif kind == 1:    # 远后急弯
        side = 1 if rng.rand() < 0.5 else -1
        th2 = th0 + side*rng.uniform(np.pi/3, 2*np.pi/3)
        g2 = g1 + rng.uniform(0.5, 1.5)*np.array([np.cos(th2), np.sin(th2)], dtype=np.float32)
        th3 = th2 + side*rng.uniform(0.3, 1.0)
        g3 = g2 + rng.uniform(0.5, 1.5)*np.array([np.cos(th3), np.sin(th3)], dtype=np.float32)
    else:              # 远侧向
        side = 1 if rng.rand() < 0.5 else -1
        th1 = th0 + side*rng.uniform(np.pi/6, np.pi/2)
        g1 = np.array([C + dl*np.cos(th1), Y + dl*np.sin(th1)], dtype=np.float32)
        g2 = g1 + np.array([0.8*np.cos(th1), 0.8*np.sin(th1)], dtype=np.float32)
        g3 = g2 + np.array([0.8*np.cos(th1), 0.8*np.sin(th1)], dtype=np.float32)
    for g in (g1, g2, g3):
        g[0] = np.clip(g[0], P.M, P.FLD-P.M); g[1] = np.clip(g[1], P.M, P.FLD-P.M)
    return s0, [g1, g2, g3]

def v_at_1m(traj, tg):
    """越过当前目标 1m 圈时的速度 (越早减速值越小). 未越过 (失败/超时) → nan."""
    g = tg[0]
    for i in range(len(traj)-1):
        d0 = np.hypot(traj[i,0]-g[0], traj[i,1]-g[1])
        d1 = np.hypot(traj[i+1,0]-g[0], traj[i+1,1]-g[1])
        if d0 >= 1.0 and d1 < 1.0:
            return float(traj[i+1, 3])
    return float('nan')

def run(m, use_gp):
    names = ('A 直行远', 'B 远后急弯', 'C 远侧向')
    for kind in range(3):
        scenes = [make_scenes(kind, i) for i in range(N_PER)]
        succ, times = P.eval_batch(m, use_gp, scenes)
        vs = []; ratios = []
        for s0, tg in scenes:
            ta, _, gi = viz.run_traj(m, use_gp, s0.copy(), tg)
            vs.append(v_at_1m(ta, tg))
            straight = np.hypot(tg[0][0]-s0[0], tg[0][1]-s0[1])
            plen = sum(np.hypot(*(ta[i+1,:2]-ta[i,:2])) for i in range(len(ta)-1))
            ratios.append(plen/straight)
        ok_v = np.array([v for v in vs if not np.isnan(v)])
        print(f'  {names[kind]:>8s}: {int(succ.sum()):2d}/{N_PER}  {times[succ].mean():5.2f}s  '
              f'v@1m={ok_v.mean():.2f} (n={len(ok_v)})  path={np.mean(ratios):.2f}x')

def main():
    paths = sys.argv[1:] or ['runs/kamm533_teacher.pt', 'runs/backup_v3/kamm533_teacher.pt',
                             'runs/backup_pre_fix/kamm533_teacher.pt']
    for p in paths:
        if not os.path.exists(p):
            print(f'skip {p}'); continue
        m, use_gp = load_model(p); m.eval()
        print(f'--- {p} ({ "GP" if use_gp else "teacher" }) ---')
        run(m, use_gp)

if __name__ == '__main__':
    main()
