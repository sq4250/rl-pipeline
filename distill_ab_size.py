"""容量 A/B: 在 v3 教师上蒸馏 GP-Medium (~9K, ≤10K 部署预算) vs 现 GP-Small 3.7K.

流程与 phase_d 完全一致 (BC 距离加权 200ep + DAgger×12 两级选优), 仅换学生架构.
产物: runs/ab_gpmed_bc.pt / ab_gpmed_d{n}.pt / ab_gpmed.pt (best, 带 arch='gpmed')
末尾对比: teacher / GP-Small (D4) / GP-Med (best) — 同一 200 随机批 + scene 77/126/38 特写.

续跑: --from-round N  从 ab_gpmed_d{N}.pt 继续 N+1..12 (需 ab_gpmed_progress.json)
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P
import viz
from eval_ckpt import load_model
from bench_loops import detect_loops

SAVE = 'runs/ab_gpmed'
PROG = f'{SAVE}_progress.json'

def save_ckpt_student(student, t, path):
    torch.save({'model_state_dict': student.state_dict(), 'arch': 'gpmed', 'time': t}, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from-round', type=int, default=0, help='从该轮继续 (0=从头 BC)')
    ap.add_argument('--save', default='runs/ab_gpmed', help='产物前缀')
    args = ap.parse_args()
    global SAVE, PROG
    SAVE = args.save; PROG = f'{SAVE}_progress.json'

    teacher, _ = load_model('runs/kamm533_teacher.pt'); teacher.eval()
    P.set_teacher_deterministic(teacher)
    student = P.GPMedium().to(P.DEV)
    n_params = sum(p.numel() for p in student.parameters())
    print(f'=== 容量 A/B: GP-Small 3762 vs GP-Medium {n_params} (teacher 76.7K) ===')

    prog = {}
    if args.from_round > 0:
        if not os.path.exists(f'{SAVE}_d{args.from_round}.pt') or not os.path.exists(PROG):
            raise SystemExit(f'续跑起点缺失: 需要 {SAVE}_d{args.from_round}.pt 与 {PROG}')
        with open(PROG) as f: prog = json.load(f)
        student.load_state_dict(torch.load(f'{SAVE}_d{args.from_round}.pt',
                                           map_location=P.DEV, weights_only=False)['model_state_dict'])
        # 用已有轮次重建两阶段最优 (BC 起)
        best_time, best_rand, best_round = None, -1.0, 0
        for r in sorted(prog.keys(), key=lambda k: -1 if k == 'BC' else int(k)):
            t, st, sr = prog[r]['t'], prog[r]['st'], prog[r]['sr']
            if sr > best_rand or (sr == best_rand and (best_time is None or t < best_time)):
                best_time, best_rand, best_round = t, sr, 0 if r == 'BC' else int(r)
        print(f'续跑自 D{args.from_round}: 已重建 best = D{best_round} ({best_time:.2f}s, rand={best_rand:.2f})')
    else:
        # ---- D1: BC ----
        print('\n--- D1: BC (200 ep, distance-weighted) ---')
        X_pol, V2, V3, Y = P.collect_bc(teacher, noise_std=0.03)
        print(f'  {X_pol.shape[0]} samples')
        P.train_epochs(student, X_pol, V2, V3, Y, 200, 2e-3)
        t_bc, st_bc, sr_bc = P.calc_mean_time(student)
        print(f'  BC mean time: {t_bc:.2f}s  tight={st_bc:.2f}  rand={sr_bc:.2f}')
        save_ckpt_student(student, t_bc, f'{SAVE}_bc.pt')
        prog['BC'] = {'t': t_bc, 'st': st_bc, 'sr': sr_bc}
        with open(PROG, 'w') as f: json.dump(prog, f)
        best_time, best_rand, best_round = t_bc, sr_bc, 0
        save_ckpt_student(student, t_bc, f'{SAVE}.pt')

    # ---- D2: DAgger ×12 ----
    print(f'\n--- D2: DAgger ×{max(0, 12 - args.from_round)} (speed-selected, random-gated) ---')
    for dagger_it in range(args.from_round + 1, 13):
        print(f'\n=== DAgger round {dagger_it} ===')
        Xn, V2n, V3n, Yn = P.collect_dagger(teacher, student)
        print(f'  Dataset: {Xn.shape[0]} samples')
        lr = 5e-4 if dagger_it <= 6 else 2e-4
        P.train_epochs(student, Xn, V2n, V3n, Yn, 100, lr)
        t, st, sr = P.calc_mean_time(student)
        print(f'  D{dagger_it} mean time: {t:.2f}s  tight={st:.2f}  rand={sr:.2f}')
        if sr > best_rand or (sr == best_rand and t < best_time):
            best_time, best_rand, best_round = t, sr, dagger_it
            save_ckpt_student(student, t, f'{SAVE}.pt')
            print(f'  -> Best! Saved (D{dagger_it})')
        save_ckpt_student(student, t, f'{SAVE}_d{dagger_it}.pt')
        prog[str(dagger_it)] = {'t': t, 'st': st, 'sr': sr}
        with open(PROG, 'w') as f: json.dump(prog, f)

    best_src = f'D{best_round}' if best_round > 0 else 'BC'
    print(f'\nBest: {best_src} with {best_time:.2f}s (rand={best_rand:.2f}) -> {SAVE}.pt')

    # ---- 最终对比: 同一 200 随机批 ----
    print('\n=== Final: teacher vs GP-Small(D4) vs GP-Med(best) on same 200 random scenes ===')
    gp_small, _ = load_model('runs/gp_small_kamm533.pt'); gp_small.eval()
    gp_med, _ = load_model(f'{SAVE}.pt'); gp_med.eval()

    rng0 = np.random.RandomState(1000)
    S = []
    for i in range(200):
        rng = np.random.RandomState(1000 + i)
        s0 = np.array([rng.uniform(P.M, P.FLD-P.M), rng.uniform(P.M, P.FLD-P.M),
                       rng.uniform(-np.pi, np.pi), rng.uniform(0.5, P.V_MAX),
                       rng.uniform(-P.DELTA_MAX, P.DELTA_MAX)], dtype=np.float32)
        tg = [np.array([rng.uniform(P.M, P.FLD-P.M), rng.uniform(P.M, P.FLD-P.M)], dtype=np.float32)
              for _ in range(3)]
        S.append((s0, tg))
    res = {}
    for name, m, ug in (('teacher', teacher, False), ('GP-Small', gp_small, True), ('GP-Med', gp_med, True)):
        succ, times = P.eval_batch(m, ug, S)
        ok = times[succ]
        res[name] = succ
        print(f'  {name:>9s}: {int(succ.sum())}/200  mean={ok.mean():.2f}s  p90={np.quantile(ok, .9):.2f}s  max={ok.max():.2f}s')

    # ---- 特写: scene 77 (失败/僵持), 126 (路径不直), 38 (长距高速) ----
    print('\n=== 特写场景 ===')
    def scene(i):
        rng = np.random.RandomState(1000 + i)
        s0 = np.array([rng.uniform(P.M, P.FLD-P.M), rng.uniform(P.M, P.FLD-P.M), rng.uniform(-np.pi, np.pi),
                       rng.uniform(0.5, P.V_MAX), rng.uniform(-P.DELTA_MAX, P.DELTA_MAX)], dtype=np.float32)
        tg = [np.array([rng.uniform(P.M, P.FLD-P.M), rng.uniform(P.M, P.FLD-P.M)], dtype=np.float32) for _ in range(3)]
        return s0, tg
    for i in (77, 126, 38):
        s0, tg = scene(i)
        straight = np.hypot(tg[0][0]-s0[0], tg[0][1]-s0[1]) + sum(
            np.hypot(tg[k+1][0]-tg[k][0], tg[k+1][1]-tg[k][1]) for k in range(2))
        print(f'  --- scene {i}: straight-line {straight:.2f}m ---')
        for tag, m, ug in (('T', teacher, False), ('S', gp_small, True), ('M', gp_med, True)):
            ta, _, gi = viz.run_traj(m, ug, s0.copy(), tg)
            ev, _ = detect_loops(ta, tg)
            plen = sum(np.hypot(*(ta[i+1,:2]-ta[i,:2])) for i in range(len(ta)-1))
            print(f'    {tag}: {gi}/3 {(len(ta)-1)*P.DT:5.1f}s  v_p50={np.median(ta[:,3]):.2f}  '
                  f'path={plen:.2f}m ({plen/straight:.2f}x)  loops={ev if ev else "-"}')
        viz.render_compare(f'scene {i} — teacher vs GP-Small vs GP-Med', s0, tg,
            [('teacher', teacher, False, '#ff6666'), ('GP-Small', gp_small, True, '#44ff44'),
             ('GP-Med', gp_med, True, '#66ccff')],
            f'runs/viz_ab_rand{i}.png')
    print('Done')

if __name__ == '__main__':
    main()
