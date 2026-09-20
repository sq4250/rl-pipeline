"""轨迹可视化 + 快速测试: 默认跑 6 信标路线, 可选生成 N 目标随机场景.

只依赖 pipeline.py (自动识别教师 GatedConcatActor / 学生 GP-Medium·GP-Small).

用法 (默认跑 6 信标路线; --n-target 额外加一个 N 目标随机场景):
  python eval_viz.py                       # GIF (单车)
  python eval_viz.py --png                 # GIF + PNG
  python eval_viz.py --compare             # GIF 里两辆车同时跑 (学生 + 教师)
  python eval_viz.py --compare --png       # 两者都叠加对比
  python eval_viz.py --ckpt runs/xxx.pt    # 换驱动模型 (默认 runs/kamm533_student.pt)
  python eval_viz.py --n-target 5 --seed 7 # 额外 N 目标随机场景 (种子可复现)
输出: <out-dir>/viz_eval_route6.gif (+ .png 若 --png)
      <out-dir>/viz_eval_{N}t_s{seed}.gif (+ .png)
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
V_CMAP = plt.get_cmap('turbo')

# 当年 6 信标坐标 (m) + 回家点
ROUTE = np.array([[0, 0], [1.715, 0.815], [3.445, 1.43], [4.565, 0.095],
                  [2.965, -0.075], [3.70, -1.48], [1.91, -1.09], [0, 0]])


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


# ═══════════════════ 随机场景 (与训练同分布: 增量 polar 链) ═══════════════════
def scene_random(n_target, seed):
    """N 段腿增量 polar 链, 分布对齐 pipeline.gen_scenes_polar.

    角度: 侧前/侧后/正后 = 40/40/20 (σ=π/4, 正后 ×0.6)
    距离: 半正态 σ=1.5 截断 [0.05, 6.5]
    初始: 车在场中心朝 +x, v~U[0,V_MAX], δ~U[-δmax, δmax]
    """
    rng = np.random.RandomState(seed)
    s0 = np.array([P.CAR_X, P.CAR_Y, 0.0, rng.uniform(0.5, P.V_MAX),
                   rng.uniform(-P.DELTA_MAX, P.DELTA_MAX)], dtype=np.float32)
    hdg = 0.0
    pos = np.array([P.CAR_X, P.CAR_Y])
    tg = []
    for _ in range(n_target):
        r = rng.rand()
        if r < 0.4:    dth = rng.normal(np.pi/2, np.pi/4)
        elif r < 0.8:  dth = rng.normal(-np.pi/2, np.pi/4)
        else:          dth = rng.normal(np.pi, np.pi/4*0.6)
        hdg += dth
        d = np.clip(abs(rng.normal(0.0, 1.5)), 0.05, 6.5)
        pos = pos + d*np.array([np.cos(hdg), np.sin(hdg)])
        pos = np.clip(pos, P.M, P.FLD - P.M)
        tg.append(pos.astype(np.float32).copy())
    return s0, tg


# ═══════════════════ 渲染 ═══════════════════
def goal_color(k, n):
    return '#ff4444' if k == 0 else ('#44ff44' if k == n-1 else '#ffaa00')


def render_gif(tag, runs, out_gif, tgts=None, route=None):
    """动画: runs = [(label, traj, color)] — 单模型按速度着色, 多模型各车一色同跑.

    tgts: 目标点 (X + 到达容差圈 + 光晕 + g{k} 标签); route: 可选导航虚线.
    """
    single = len(runs) == 1
    fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor=PAGE)
    ax.set_facecolor(PAGE); ax.set_aspect('equal')
    ref = [t[:, :2] for _, t, _ in runs]
    if route is not None:
        ax.plot(route[:, 0], route[:, 1], '--', color='#c98500', lw=1.0, alpha=0.5,
                zorder=4, label='route')
        ref.append(np.asarray(route, dtype=float))
    if tgts is not None:
        for k, g in enumerate(tgts):
            c = goal_color(k, len(tgts))
            ax.plot(g[0], g[1], marker='x', color=c, ms=13, mew=2.6, zorder=7)
            ax.add_patch(plt.Circle((g[0], g[1]), P.TOL, fc='none', ec=c, lw=1.0,
                                    alpha=0.85, zorder=6))
            ax.add_patch(plt.Circle((g[0], g[1]), 0.25, fc=c, ec='none', alpha=0.10, zorder=2))
            ax.text(g[0]+0.14, g[1]+0.14, f'g{k+1}', color=c, fontsize=10,
                    weight='bold', zorder=8)
        ref.append(np.asarray(tgts, dtype=float))
    allp = np.concatenate(ref)
    lo, hi = allp.min(axis=0) - 0.8, allp.max(axis=0) + 0.8
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    # 轨迹: 单模型按速度着色 (带色条); 多模型各车一色 (带图例)
    trails = []
    for lab, traj, col in runs:
        if single:
            lc = LineCollection([], cmap=V_CMAP, norm=plt.Normalize(0, P.V_MAX))
            lc.set_linewidth(2.2)
        else:
            lc = LineCollection([], colors=col, lw=2.0, alpha=0.85, label=lab)
        ax.add_collection(lc); trails.append(lc)
    if single:
        cb = fig.colorbar(trails[0], ax=ax, fraction=0.036, pad=0.02)
        cb.set_label('v [m/s]', color='#c3c2b7'); cb.ax.tick_params(colors='#898781', labelsize=7)
    else:
        ax.legend(fontsize=10, labelcolor='white', facecolor='#1a1a1a', edgecolor='#444',
                  loc='upper left')
    ax.set_title(tag, color='white')
    n_max = max(len(t) for _, t, _ in runs)
    stride = max(1, n_max//400)
    frames = list(range(0, n_max, stride))
    if n_max-1 not in frames: frames.append(n_max-1)
    objs = []

    def update(i):
        for o in objs: o.remove()
        objs.clear()
        for (lab, traj, col), trail in zip(runs, trails):
            j = min(frames[i], len(traj)-1)          # 先跑完的车停在终点
            seg = traj[:j+1, :2]
            if len(seg) > 1:
                pts = seg.reshape(-1, 1, 2)
                trail.set_segments(np.concatenate([pts[:-1], pts[1:]], axis=1))
                if single: trail.set_array(traj[:j, 3])
            else:
                trail.set_segments([])
            x, y, th, dlt = traj[j, 0], traj[j, 1], traj[j, 2], traj[j, 4]
            bx, by = 0.15*np.cos(th), 0.15*np.sin(th)
            body = col if not single else '#ffcc00'
            l1, = ax.plot([x-bx, x+bx], [y-by, y+by], color=body, lw=2.2, zorder=9)
            fx, fy, wth = x+bx, y+by, th+dlt
            wx, wy = 0.075*np.cos(wth), 0.075*np.sin(wth)
            l2, = ax.plot([fx-wx, fx+wx], [fy-wy, fy+wy], color='#ff4444', lw=2.6, zorder=9)
            objs.extend([l1, l2])

    ani = FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False)
    ani.save(out_gif, writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f'  -> {out_gif}')


def render_compare(tag, tgts, runs, out_png):
    """runs: [(label, traj, gi, color)] — 轨迹 (自适应视窗) + 速度曲线."""
    fig, (ax_t, ax_v) = plt.subplots(1, 2, figsize=(15, 5.4), facecolor=PAGE,
                                     gridspec_kw={'width_ratios': [1.3, 0.7]})
    ax_v.set_facecolor(PAGE); mx = 0.0
    for lab, traj, gi, col in runs:
        tt = (len(traj)-1)*P.DT
        ax_t.plot(traj[:, 0], traj[:, 1], color=col, lw=1.8, alpha=0.8,
                  label=f'{lab} ({gi}/{len(tgts)}, {tt:.1f}s)')
        ax_v.plot(np.arange(len(traj))*P.DT, traj[:, 3], color=col, lw=1.4, label=lab)
        mx = max(mx, tt)
    for k, g in enumerate(tgts):
        c = goal_color(k, len(tgts))
        ax_t.plot(g[0], g[1], marker='x', color=c, ms=12, mew=2.5)
        ax_t.text(g[0]+0.1, g[1]+0.1, f'g{k+1}', color=c, fontsize=8, weight='bold')
    allp = np.concatenate([t[:, :2] for _, t, _, _ in runs] + [np.asarray(tgts, dtype=float)])
    lo, hi = allp.min(axis=0) - 0.6, allp.max(axis=0) + 0.6
    ax_t.add_patch(plt.Rectangle((0, 0), P.FLD, P.FLD, fc='none', ec='#555555', ls=':', lw=0.9))
    ax_t.set_xlim(lo[0], hi[0]); ax_t.set_ylim(lo[1], hi[1])
    ax_t.set_aspect('equal', adjustable='datalim')
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


# ═══════════════════ main ═══════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=DEF_CKPT, help='驱动模型 (默认学生)')
    ap.add_argument('--teacher', default=DEF_TEACHER, help='对比参照; 不存在则跳过')
    ap.add_argument('--compare', action='store_true',
                    help='叠加对比: GIF 同时跑两辆车 (驱动模型 + 教师), PNG 也随之变对比图')
    ap.add_argument('--png', action='store_true', help='额外输出 PNG (轨迹 + 速度曲线)')
    ap.add_argument('--n-target', type=int, default=None,
                    help='额外跑一个 N 目标随机场景 (不给则只跑 6 信标路线)')
    ap.add_argument('--seed', type=int, default=0, help='随机场景种子 (配合 --n-target)')
    ap.add_argument('--out-dir', default='.', help='输出目录 (默认当前目录)')
    args = ap.parse_args()
    out_dir = args.out_dir if os.path.isdir(args.out_dir) else '.'
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(args.ckpt):
        raise SystemExit(f'{args.ckpt} 不存在 — 先跑 python pipeline.py')
    model, is_student = load_any(args.ckpt)
    kind = 'student' if is_student else 'teacher'
    n_par = sum(p.numel() for p in model.parameters())
    driver = f'{kind.upper()} {os.path.basename(args.ckpt)}'
    print(f'Loaded {args.ckpt}  [{kind}, {n_par} params]')

    teacher = None; t_is_st = None
    if args.compare:
        if os.path.abspath(args.teacher) == os.path.abspath(args.ckpt):
            raise SystemExit('--compare 需要 --teacher 指向另一个模型')
        if not os.path.exists(args.teacher):
            raise SystemExit(f'--compare 需要教师 checkpoint: {args.teacher} 不存在')
        try:
            teacher, t_is_st = load_any(args.teacher)
        except Exception as e:
            raise SystemExit(f'教师加载失败: {e}')
        if t_is_st == is_student:
            raise SystemExit('--compare 的两个模型须一为学生一为教师')
        print(f'Compare with {args.teacher}  [{"student" if t_is_st else "teacher"}]')

    def run_and_render(tag, stem, s0, tg, route=None):
        """跑驱动模型 (--compare 时并跑教师) → GIF; --png 时另出 PNG."""
        traj, gi = run_traj(model, is_student, s0.copy(), tg)
        t_dur = (len(traj)-1)*P.DT
        print(f'  {tag}  {kind}: {gi}/{len(tg)}  {t_dur:.2f}s')
        gif_runs = [(f'{kind} {os.path.basename(args.ckpt)}', traj, '#44ff44')]
        png_runs = [(kind, traj, gi, '#44ff44')]
        ttl = f'{tag} — driven by {driver}'
        if teacher is not None:
            t_tr, t_gi = run_traj(teacher, t_is_st, s0.copy(), tg)
            tt_dur = (len(t_tr)-1)*P.DT
            print(f'  {tag}  teacher: {t_gi}/{len(tg)}  {tt_dur:.2f}s   Δt={t_dur-tt_dur:+.2f}s')
            gif_runs.append((f'teacher {os.path.basename(args.teacher)}', t_tr, '#ff6666'))
            png_runs.append(('teacher', t_tr, t_gi, '#ff6666'))
            ttl = f'{tag} — {kind} vs teacher'
        render_gif(ttl, gif_runs, f'{stem}.gif', tgts=tg, route=route)
        if args.png:
            render_compare(ttl, tg, png_runs, f'{stem}.png')

    # ── 默认: 6 信标路线 ──
    tg = [np.array(p, dtype=np.float32) for p in ROUTE[1:]]
    run_and_render('6-beacon route', f'{out_dir}/viz_eval_route6',
                   np.array([0., 0., 0., 0., 0.], dtype=np.float32), tg, route=ROUTE)

    # ── 可选: N 目标随机场景 ──
    if args.n_target:
        s0, tg = scene_random(args.n_target, args.seed)
        run_and_render(f'{args.n_target}-target scene (seed {args.seed})',
                       f'{out_dir}/viz_eval_{args.n_target}t_s{args.seed}', s0, tg)

    print('Done.')


if __name__ == '__main__':
    main()
