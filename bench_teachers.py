"""教师 checkpoint 全面对比 (视频选角用).

维度: 4 标准场景 / random 100 / 反打探针 (40 触发场景) / 多绕圈 (6 展示场景) / 多目标泛化 (3-8 目标).
全部确定性评估 (mean 动作). 输出对比表 + 综合推荐.
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P
import viz
from eval_ckpt import load_model
from viz_multitarget import showcases
from bench_loops import detect_loops

TEACHERS = [
    ('v2-current',    'runs/kamm533_teacher.pt'),
    ('v1-backup',     'runs/backup_pre_fix/kamm533_teacher.pt'),
    ('v1_c550',       'runs/kamm533_teacher_v1_c550.pt'),
    ('v2_c1000',      'runs/kamm533_teacher_v2_c1000.pt'),
    ('v3_nofreeze',   'runs/kamm533_teacher_v3_nofreeze.pt'),
    ('v4_freeze',     'runs/kamm533_teacher_v4_freeze.pt'),
    ('v5_segmented',  'runs/kamm533_teacher_v5_segmented.pt'),
    ('v6_polarheavy', 'runs/kamm533_teacher_v6_polarheavy.pt'),
]
# 注: v2/v1 主候选排最前, 先出结果; 变体在后补完.

C, Y = P.CAR_X, P.CAR_Y

def std4(m):
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
    succ, times = P.eval_batch(m, False, scenes)
    return int(succ.sum()), float(times[succ].mean())

def rand100(m):
    scenes = []
    for i in range(100):
        rng = np.random.RandomState(i)
        s0 = np.array([rng.uniform(P.M, P.FLD-P.M), rng.uniform(P.M, P.FLD-P.M), rng.uniform(-np.pi, np.pi),
                       rng.uniform(0.5, P.V_MAX), rng.uniform(-P.DELTA_MAX, P.DELTA_MAX)], dtype=np.float32)
        tg = [np.array([rng.uniform(P.M, P.FLD-P.M), rng.uniform(P.M, P.FLD-P.M)], dtype=np.float32) for _ in range(3)]
        scenes.append((s0, tg))
    succ, times = P.eval_batch(m, False, scenes)
    return int(succ.sum()), float(times[succ].mean())

def trigger_scenes(n=40):
    scenes = [P.gen_trigger_scene(np.random.RandomState(3000+i)) for i in range(n)]
    return scenes

def probe(m):
    sc = trigger_scenes(40)
    succ, times = P.eval_batch(m, False, sc)
    cs_total = 0; n_steps = 0
    for s0, tg in sc[:12]:
        traj, gi = P.rollout_traj(m, False, s0, tg)
        if len(traj) < 3: continue
        dth = np.diff(traj[:, 2]) / P.DT
        dth = np.arctan2(np.sin(dth), np.cos(dth))
        delta = traj[:-1, 4]
        cs = (np.abs(delta) > 0.1) & (np.abs(dth) > 0.15) & (np.sign(delta) != np.sign(dth))
        cs_total += int(cs.sum()); n_steps += len(delta)
    return int(succ.sum()), float(times.mean()), cs_total

def loops(m):
    tot_t = 0.0; tot_ev = 0; ratios = []
    for name, s0, tgts in showcases():
        ta, _, gi = viz.run_traj(m, False, s0.copy(), tgts)
        ev, _ = detect_loops(ta, tgts)
        tot_t += (len(ta)-1)*P.DT; tot_ev += len(ev)
        straight = np.hypot(tgts[0][0]-s0[0], tgts[0][1]-s0[1]) + sum(
            np.hypot(tgts[k+1][0]-tgts[k][0], tgts[k+1][1]-tgts[k][1]) for k in range(len(tgts)-1))
        plen = sum(np.hypot(*(ta[i+1,:2]-ta[i,:2])) for i in range(len(ta)-1))
        ratios.append(plen/straight)
    return tot_t, tot_ev, float(np.mean(ratios))

def multitarget(m, ns=(3, 4, 5, 6, 8), n_scenes=100):
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
        succ, times = P.eval_batch(m, False, scenes)
        tot_succ += int(succ.sum()); tot_t += float(times.sum()); tot_n += len(scenes)
    P.MAX_EVAL_STEPS = old
    return tot_succ, tot_t/tot_n

def main():
    rows = []
    for name, path in TEACHERS:
        if not os.path.exists(path):
            print(f'skip {name}: {path} 不存在'); continue
        try:
            m, use_gp = load_model(path)
            if use_gp:
                print(f'skip {name}: 是蒸馏模型'); continue
            m.eval()
        except Exception as e:
            print(f'skip {name}: 加载失败 ({e})'); continue
        s4, t4 = std4(m)
        s100, t100 = rand100(m)
        sp, tp, cs = probe(m)
        tl, lev, pr = loops(m)
        smt, tmt = multitarget(m)
        rows.append((name, s4, t4, s100, t100, sp, tp, cs, tl, lev, pr, smt, tmt))
        print(f'{name:>14s}: std4={s4:2d}/16 {t4:4.1f}s | r100={s100:3d} {t100:4.1f}s | '
              f'probe={sp:2d}/40 {tp:4.1f}s cs={cs:3d} | loops t={tl:4.1f}s ev={lev} path={pr:.2f}x | '
              f'MT={smt}/500 {tmt:4.1f}s')
    print('\n=== 综合推荐 (按 random100+MT 成功率, 时间与绕圈次之) ===')
    if not rows: return
    rows.sort(key=lambda r: (-r[3], -r[11], r[9], r[4]))
    for i, r in enumerate(rows[:3]):
        print(f'  #{i+1} {r[0]}: r100={r[3]}/100, MT={r[11]}/500, loops_ev={r[9]}, r100_time={r[4]:.1f}s')

if __name__ == '__main__':
    main()
