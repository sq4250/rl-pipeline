"""最终评测电池: 任意 checkpoint 列表全面对比.

维度: std4 / random200 (seed 1000+) / 反打探针 40 / 长距探针 3×40 / 多绕圈 6 展示 / 多目标 3-8.
用法: python final_battery.py path1 [path2 ...]
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P
import viz
from eval_ckpt import load_model
from viz_multitarget import showcases
from bench_loops import detect_loops
import probe_long

C, Y = P.CAR_X, P.CAR_Y

def std4(m, use_gp):
    scenes = [
        (np.array([0., 0., np.pi/4, 0., 0.], dtype=np.float32),
         [np.array([C+1.5, Y+1.5]), np.array([C+0.5, Y+3.0]), np.array([C-1.0, Y+1.5])]),
        (np.array([0., 0., 0., 0., 0.], dtype=np.float32),
         [np.array([C+2.0, Y+0.0]), np.array([C+2.5, Y+2.5]), np.array([C+0.5, Y+2.0])]),
        (np.array([C, Y, 0., 2.0, 0.], dtype=np.float32),
         [np.array([C-1.0, Y+1.5]), np.array([C+1.0, Y+4.0]), np.array([C+3.0, Y+2.0])]),
        (np.array([C, Y, 0., 0., 0.], dtype=np.float32),
         [np.array([C+2.0, Y+0.1]), np.array([C+4.0, Y-0.1]), np.array([C+6.0, Y+0.0])]),
    ]
    scenes = [(s, t) for s, t in scenes for _ in range(4)]
    succ, times = P.eval_batch(m, use_gp, scenes)
    return int(succ.sum()), float(times[succ].mean())

def rand200(m, use_gp):
    scenes = []
    for i in range(200):
        rng = np.random.RandomState(1000 + i)
        s0 = np.array([rng.uniform(P.M, P.FLD-P.M), rng.uniform(P.M, P.FLD-P.M), rng.uniform(-np.pi, np.pi),
                       rng.uniform(0.5, P.V_MAX), rng.uniform(-P.DELTA_MAX, P.DELTA_MAX)], dtype=np.float32)
        tg = [np.array([rng.uniform(P.M, P.FLD-P.M), rng.uniform(P.M, P.FLD-P.M)], dtype=np.float32) for _ in range(3)]
        scenes.append((s0, tg))
    succ, times = P.eval_batch(m, use_gp, scenes)
    ok = times[succ]
    return int(succ.sum()), float(ok.mean()), float(np.quantile(ok, .9)), float(ok.max())

def probe_cs(m, use_gp, n=40):
    sc = [P.gen_trigger_scene(np.random.RandomState(3000+i)) for i in range(n)]
    succ, times = P.eval_batch(m, use_gp, sc)
    cs_total = 0; n_steps = 0
    for s0, tg in sc[:12]:
        traj, gi = P.rollout_traj(m, use_gp, s0, tg)
        if len(traj) < 3: continue
        dth = np.diff(traj[:, 2]) / P.DT
        dth = np.arctan2(np.sin(dth), np.cos(dth))
        delta = traj[:-1, 4]
        cs = (np.abs(delta) > 0.1) & (np.abs(dth) > 0.15) & (np.sign(delta) != np.sign(dth))
        cs_total += int(cs.sum()); n_steps += len(delta)
    return int(succ.sum()), float(times.mean()), cs_total

def long3(m, use_gp):
    """长距探针三类的聚合: (succ/120, mean_time, mean_v@1m)."""
    out = []
    for kind in range(3):
        scenes = [probe_long.make_scenes(kind, i) for i in range(probe_long.N_PER)]
        succ, times = P.eval_batch(m, use_gp, scenes)
        vs = []
        for s0, tg in scenes:
            ta, _, gi = viz.run_traj(m, use_gp, s0.copy(), tg)
            v = probe_long.v_at_1m(ta, tg)
            if not np.isnan(v): vs.append(v)
        out.append((int(succ.sum()), float(times[succ].mean()), float(np.mean(vs)) if vs else float('nan')))
    return out

def loops6(m, use_gp):
    tot_t = 0.0; tot_ev = 0; ratios = []
    for name, s0, tgts in showcases():
        ta, _, gi = viz.run_traj(m, use_gp, s0.copy(), tgts)
        ev, _ = detect_loops(ta, tgts)
        tot_t += (len(ta)-1)*P.DT; tot_ev += len(ev)
        straight = np.hypot(tgts[0][0]-s0[0], tgts[0][1]-s0[1]) + sum(
            np.hypot(tgts[k+1][0]-tgts[k][0], tgts[k+1][1]-tgts[k][1]) for k in range(len(tgts)-1))
        plen = sum(np.hypot(*(ta[i+1,:2]-ta[i,:2])) for i in range(len(ta)-1))
        ratios.append(plen/straight)
    return tot_t, tot_ev, float(np.mean(ratios))

def multitarget(m, use_gp, ns=(3, 4, 5, 6, 8), n_scenes=100):
    old = P.MAX_EVAL_STEPS; P.MAX_EVAL_STEPS = 1200
    tot_succ = 0; tot_t = 0.0; tot_n = 0
    for N in ns:
        scenes = []
        for i in range(n_scenes):
            rng = np.random.RandomState(700 + N*100 + i)
            s0 = np.array([rng.uniform(P.M, P.FLD-P.M), rng.uniform(P.M, P.FLD-P.M), rng.uniform(-np.pi, np.pi),
                           rng.uniform(0.5, P.V_MAX), rng.uniform(-P.DELTA_MAX, P.DELTA_MAX)], dtype=np.float32)
            tg = [np.array([rng.uniform(P.M, P.FLD-P.M), rng.uniform(P.M, P.FLD-P.M)], dtype=np.float32) for _ in range(N)]
            scenes.append((s0, tg))
        succ, times = P.eval_batch(m, use_gp, scenes)
        tot_succ += int(succ.sum()); tot_t += float(times.sum()); tot_n += len(scenes)
    P.MAX_EVAL_STEPS = old
    return tot_succ, tot_t/tot_n

def main():
    paths = sys.argv[1:] or ['runs/kamm533_teacher.pt', 'runs/gp_small_kamm533.pt']
    for p in paths:
        if not os.path.exists(p):
            print(f'skip {p}'); continue
        m, use_gp = load_model(p); m.eval()
        kind = 'GP' if use_gp else 'Teacher'
        s4, t4 = std4(m, use_gp)
        s200, t200, p90, mx = rand200(m, use_gp)
        sp, tp, cs = probe_cs(m, use_gp)
        L = long3(m, use_gp)
        lt = sum(r[0] for r in L); lt_mean = float(np.mean([r[1] for r in L])); lv = float(np.mean([r[2] for r in L if not np.isnan(r[2])]))
        tl, lev, pr = loops6(m, use_gp)
        smt, tmt = multitarget(m, use_gp)
        print(f'== {p} ({kind}) ==')
        print(f'  std4:      {s4}/16  {t4:.2f}s')
        print(f'  random200: {s200}/200  mean={t200:.2f}s  p90={p90:.2f}s  max={mx:.2f}s')
        print(f'  反打探针:  {sp}/40  {tp:.2f}s  cs_steps={cs}')
        print(f'  长距3类:   succ={lt}/120  mean={lt_mean:.2f}s  v@1m={lv:.2f}')
        print(f'    逐类:    ' + ' | '.join(f'{r[0]:2d}/40 {r[1]:.2f}s v@1m={r[2]:.2f}' for r in L))
        print(f'  多绕圈:    t={tl:.1f}s  events={lev}  path={pr:.2f}x')
        print(f'  多目标:    {smt}/500  {tmt:.2f}s')
        print()

if __name__ == '__main__':
    main()
