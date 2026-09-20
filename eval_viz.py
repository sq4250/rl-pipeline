"""评测 + 可视化: 加载学生模型, 跑标准/随机场景, 出轨迹图与动画.

只依赖 pipeline.py (自动识别教师 GatedConcatActor / 学生 GP-Medium·GP-Small).

用法:
  python eval_viz.py                     # 默认 runs/kamm533_student.pt
  python eval_viz.py --gif               # 额外出 6 信标路线动画
  python eval_viz.py --ckpt runs/xxx.pt  # 指定 checkpoint
  python eval_viz.py --n-random 200      # 随机场景数 (默认 100)
输出: runs/viz_eval_*.png / .gif
"""
import os, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P

DEF_CKPT = 'runs/kamm533_student.pt'
DEF_TEACHER = 'runs/kamm533_teacher.pt'
PAGE = '#0d0d0d'


# ═══════════════════ 模型加载 (按 arch 字段自动重建结构) ═══════════════════
def load_any(path):
    """返回 (model, is_student). 教师 → forward(o1,o2,o3,v2,v3); 学生 → deploy(p8,v2,v3)."""
    ck = torch.load(path, map_location=P.DEV, weights_only=False)
    if 'model_state_dict' in ck:
        cls = {'gpsmall': P.GPSmall, 'gpmed': P.GPMedium}.get(ck.get('arch', 'gpsmall'))
        if cls is None:
            raise SystemExit(f'未知学生架构: {ck.get("arch")} ({path})')
        m = cls().to(P.DEV); m.load_state_dict(ck['model_state_dict']); m.eval()
        return m, True
    if 'actor_state_dict' in ck:
        m = P.GatedConcatActor().to(P.DEV); m.load_state_dict(ck['actor_state_dict'])
        P.set_teacher_deterministic(m); m.eval()
        return m, False
    raise SystemExit(f'无法识别的 checkpoint: {path}')


# ═══════════════════ 单场景 rollout ═══════════════════
def run_traj(model, is_student, s0, tgts, max_steps=600):
    s = torch.tensor(s0, device=P.DEV, dtype=torch.float32).unsqueeze(0)
    gi = 0; traj = [np.asarray(s0, dtype=np.float64).copy()]
    for _ in range(max_steps):
        if gi >= len(tgts): break
        cur = torch.tensor(tgts[min(gi, len(tgts)-1)], device=P.DEV, dtype=torch.float32).unsqueeze(0)
        nxt = torch.tensor(tgts[min(gi+1, len(tgts)-1)], device=P.DEV, dtype=torch.float32).unsqueeze(0)
        nn_ = torch.tensor(tgts[min(gi+2, len(tgts)-1)], device=P.DEV, dtype=torch.float32).unsqueeze(0)
        v2 = torch.tensor([float(gi < len(tgts)-1)], device=P.DEV)
        v3 = torch.tensor([float(gi < len(tgts)-2)], device=P.DEV)
        with torch.no_grad():
            if is_student:
                act = model.deploy(P.polar_8d(s, cur, nxt, nn_), v2, v3)
            else:
                o1, o2, o3 = P.polar_obs(s, cur, nxt, nn_)
                act, _ = model.forward(o1, o2, o3, v2, v3)
        sn = P.SIM.simulate(s, act)
        snp = sn[0].cpu().numpy().astype(np.float64)
        if P.check_hit_substep_np(traj[-1][:2], snp[:2], tgts[gi], P.TOL): gi += 1
        traj.append(snp); s = sn
    return np.array(traj), gi


# ═══════════════════ 场景 ═══════════════════
def scenes_standard():
    C, Y = P.CAR_X, P.CAR_Y
    base = [
        ('tight-180', np.array([0., 0., np.pi/4, 0., 0.], dtype=np.float32),
         [np.array([C+1.5, Y+1.5]), np.array([C+0.5, Y+3.0]), np.array([C-1.0, Y+1.5])]),
        ('tight-TR', np.array([0., 0., 0., 0., 0.], dtype=np.float32),
         [np.array([C+2.0, Y+0.0]), np.array([C+2.5, Y+2.5]), np.array([C+0.5, Y+2.0])]),
        ('anti-steer', np.array([C, Y, 0., 2.0, 0.], dtype=np.float32),
         [np.array([C-1.0, Y+1.5]), np.array([C+1.0, Y+4.0]), np.array([C+3.0, Y+2.0])]),
        ('straight', np.array([C, Y, 0., 0., 0.], dtype=np.float32),
         [np.array([C+2.0, Y+0.1]), np.array([C+4.0, Y-0.1]), np.array([C+6.0, Y+0.0])]),
    ]
    return base


def scenes_random(n, seed=1000):
    out = []
    for i in range(n):
        rng = np.random.RandomState(seed + i)
        s0 = np.array([rng.uniform(P.M, P.FLD-P.M), rng.uniform(P.M, P.FLD-P.M),
                       rng.uniform(-np.pi, np.pi), rng.uniform(0.5, P.V_MAX),
                       rng.uniform(-P.DELTA_MAX, P.DELTA_MAX)], dtype=np.float32)
        tg = [np.array([rng.uniform(P.M, P.FLD-P.M), rng.uniform(P.M, P.FLD-P.M)], dtype=np.float32)
              for _ in range(3)]
        out.append((f'rand-{i}', s0, tg))
    return out


# ═══════════════════ 渲染 ═══════════════════
V_CMAP = plt.get_cmap('turbo')


def panel(ax, title, tgts, traj, idx):
    ax.set_facecolor(PAGE)
    ax.set_xlim(0, P.FLD); ax.set_ylim(0, P.FLD); ax.set_aspect('equal')
    for k, g in enumerate(tgts):
        c = '#ff4444' if k == 0 else ('#44ff44' if k == len(tgts)-1 else '#ffaa00')
        ax.plot(g[0], g[1], marker='x', color=c, ms=11, mew=2.5)
        ax.plot(g[0], g[1], 'o', mfc='none', mec=c, ms=9, mew=0.8, alpha=0.7)
    j = min(idx, len(traj)-1)
    seg = traj[:j+1, :2]
    if len(seg) > 1:
        pts = seg.reshape(-1, 1, 2); pairs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(pairs, cmap=V_CMAP, norm=plt.Normalize(0, P.V_MAX))
        lc.set_array(traj[:j, 3]); lc.set_linewidth(2.0); lc.set_alpha(0.9)
        ax.add_collection(lc)
    x, y, th, dlt = traj[j, 0], traj[j, 1], traj[j, 2], traj[j, 4]
    bx, by = 0.15*np.cos(th), 0.15*np.sin(th)
    ax.plot([x-bx, x+bx], [y-by, y+by], color='#ffcc00', lw=2.0)
    fx, fy, wth = x+bx, y+by, th+dlt
    wx, wy = 0.075*np.cos(wth), 0.075*np.sin(wth)
    ax.plot([fx-wx, fx+wx], [fy-wy, fy+wy], color='#ff4444', lw=2.4)
    ax.set_title(title, color='white', fontsize=10)
    ax.tick_params(colors='#666666', labelsize=7)


def render_compare(tag, s0, tgts, models, out_png):
    """models: [(label, traj, gi, color)]"""
    fig, (ax_t, ax_v) = plt.subplots(1, 2, figsize=(15, 5.4), facecolor=PAGE,
                                     gridspec_kw={'width_ratios': [1.3, 0.7]})
    ax_v.set_facecolor(PAGE)
    mx = 0.0
    for lab, traj, gi, col in models:
        tt = (len(traj)-1)*P.DT
        ax_t.plot(traj[:, 0], traj[:, 1], color=col, lw=1.8, alpha=0.8,
                  label=f'{lab} ({gi}/{len(tgts)}, {tt:.1f}s)')
        ax_v.plot(np.arange(len(traj))*P.DT, traj[:, 3], color=col, lw=1.4, label=lab)
        mx = max(mx, tt)
    for k, g in enumerate(tgts):
        c = '#ff4444' if k == 0 else ('#44ff44' if k == len(tgts)-1 else '#ffaa00')
        ax_t.plot(g[0], g[1], marker='x', color=c, ms=12, mew=2.5)
        ax_t.text(g[0]+0.1, g[1]+0.1, f'g{k+1}', color=c, fontsize=8, weight='bold')
    # 视窗自适应: 部分标准场景的目标在场外 (7×7 之外), 钉死 0..FLD 会裁掉轨迹
    allp = np.concatenate([t[:, :2] for _, t, _, _ in models] + [np.asarray(tgts, dtype=float)])
    lo, hi = allp.min(axis=0) - 0.6, allp.max(axis=0) + 0.6
    ax_t.add_patch(plt.Rectangle((0, 0), P.FLD, P.FLD, fc='none', ec='#555555', ls=':', lw=0.9))
    ax_t.set_xlim(lo[0], hi[0]); ax_t.set_ylim(lo[1], hi[1])
    ax_t.set_aspect('equal', adjustable='datalim')   # 等比例但填满面板 (不裁轨迹)
    ax_t.set_facecolor(PAGE); ax_t.grid(alpha=0.15)
    ax_t.legend(fontsize=8, labelcolor='white', facecolor='#1a1a1a', edgecolor='#444')
    ax_t.set_title(f'{tag}   [dotted box = 7x7 field]', color='white')
    ax_t.tick_params(colors='#666666', labelsize=7)
    ax_v.set_xlim(0, mx+0.3); ax_v.set_ylim(0, P.V_MAX+0.3)
    ax_v.axhline(P.V_MAX, color='#666', ls='--', lw=0.6)
    ax_v.set_xlabel('t [s]', color='#aaa'); ax_v.set_title('speed [m/s]', color='white')
    ax_v.legend(fontsize=8, labelcolor='white', facecolor='#1a1a1a', edgecolor='#444')
    ax_v.grid(alpha=0.15); ax_v.tick_params(colors='#666666', labelsize=7)
    fig.tight_layout(); fig.savefig(out_png, dpi=130, facecolor=PAGE); plt.close(fig)
    print(f'  -> {out_png}')


ROUTE = np.array([[0, 0], [1.715, 0.815], [3.445, 1.43], [4.565, 0.095],
                  [2.965, -0.075], [3.70, -1.48], [1.91, -1.09], [0, 0]])


def render_gif(tag, traj, out_gif, route=None):
    fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor=PAGE)
    ax.set_facecolor(PAGE); ax.set_aspect('equal')
    ref = [traj[:, :2]]
    if route is not None:
        ax.plot(route[:, 0], route[:, 1], '--', color='#c98500', lw=1.0, alpha=0.6)
        ax.scatter(route[:, 0], route[:, 1], marker='x', s=70, c='#c98500', zorder=5)
        ref.append(np.asarray(route, dtype=float))
    allp = np.concatenate(ref)
    lo, hi = allp.min(axis=0) - 0.8, allp.max(axis=0) + 0.8
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    trail = LineCollection([], cmap=V_CMAP, norm=plt.Normalize(0, P.V_MAX))
    trail.set_linewidth(2.2); ax.add_collection(trail)
    ax.set_title(tag, color='white')
    stride = max(1, len(traj)//400)
    frames = list(range(0, len(traj), stride))
    if len(traj)-1 not in frames: frames.append(len(traj)-1)
    objs = []

    def update(i):
        j = frames[i]
        for o in objs: o.remove()
        objs.clear()
        seg = traj[:j+1, :2]
        if len(seg) > 1:
            pts = seg.reshape(-1, 1, 2); pairs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            trail.set_segments(pairs); trail.set_array(traj[:j, 3])
        else:
            trail.set_segments([])
        x, y, th, dlt = traj[j, 0], traj[j, 1], traj[j, 2], traj[j, 4]
        bx, by = 0.15*np.cos(th), 0.15*np.sin(th)
        l1, = ax.plot([x-bx, x+bx], [y-by, y+by], color='#ffcc00', lw=2.2)
        fx, fy, wth = x+bx, y+by, th+dlt
        wx, wy = 0.075*np.cos(wth), 0.075*np.sin(wth)
        l2, = ax.plot([fx-wx, fx+wx], [fy-wy, fy+wy], color='#ff4444', lw=2.6)
        objs.extend([l1, l2])

    ani = FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False)
    ani.save(out_gif, writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f'  -> {out_gif}')


# ═══════════════════ main ═══════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=DEF_CKPT)
    ap.add_argument('--teacher', default=DEF_TEACHER)
    ap.add_argument('--n-random', type=int, default=100)
    ap.add_argument('--gif', action='store_true')
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit(f'{args.ckpt} 不存在 — 先跑 python pipeline.py')
    model, is_student = load_any(args.ckpt)
    kind = 'student' if is_student else 'teacher'
    n_par = sum(p.numel() for p in model.parameters())
    print(f'Loaded {args.ckpt}  [{kind}, {n_par} params]')

    # ── 批量评测 ──
    std = scenes_standard()
    for nm, s0, tg in std:
        succ, times = P.eval_batch(model, is_student, [(s0, tg)]*6)
        print(f'  {nm:>12s}: {int(succ.sum())}/6  {times[succ].mean():.2f}s')
    rnd = scenes_random(args.n_random)
    succ, times = P.eval_batch(model, is_student, [(s0, tg) for _, s0, tg in rnd])
    print(f'  {"random":>12s}: {int(succ.sum())}/{args.n_random}  '
          f'mean={times[succ].mean():.2f}s  p90={np.quantile(times[succ], .9):.2f}s')

    # ── 可视化: 学生 vs 教师 (若教师存在) ──
    models = [(kind, model, is_student, '#44ff44')]
    if os.path.exists(args.teacher) and args.teacher != args.ckpt:
        try:
            teach, ts = load_any(args.teacher)
            if ts == is_student:
                print(f'(跳过教师对比: 同为 {"学生" if ts else "教师"} 类型)')
            else:
                models.append(('teacher', teach, ts, '#ff6666'))
        except Exception as e:
            print(f'(教师加载失败, 跳过对比: {e})')

    for nm, s0, tg in std:
        runs = []
        for lab, m, ist, col in models:
            traj, gi = run_traj(m, ist, s0.copy(), tg)
            runs.append((lab, traj, gi, col))
            print(f'    {nm} {lab}: {gi}/{len(tg)} {(len(traj)-1)*P.DT:.1f}s')
        render_compare(f'{nm} — {kind} vs teacher', s0, tg, runs, f'runs/viz_eval_{nm}.png')
        if args.gif:
            render_gif(f'{nm} — {kind}', runs[0][1], f'runs/viz_eval_{nm}.gif')

    # ── 6 信标路线动画 ──
    if args.gif:
        tg = [np.array(p, dtype=np.float32) for p in ROUTE[1:]]
        s0 = np.array([0., 0., 0., 0., 0.], dtype=np.float32)
        traj, gi = run_traj(model, is_student, s0, tg)
        print(f'  {"route6":>12s}: {gi}/{len(tg)} {(len(traj)-1)*P.DT:.2f}s')
        render_gif('6-beacon route — ' + kind, traj, 'runs/viz_eval_route6.gif', route=ROUTE)

    print('Done.')


if __name__ == '__main__':
    main()
