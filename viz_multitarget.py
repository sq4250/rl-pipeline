"""多目标泛化素材: 训练用 3 目标链, 部署跑 4-12 目标 — 展示泛化能力.

风格 (同 viz_cluster6): T 形车体 (竖杆=θ, 横杆=θ+δ) + 速度色带 (蓝→青→黄→红);
目标点按访问顺序用同一色系着色 (g1 蓝 → gN 红), 带序号标签.

输出:
  runs/viz_mt_<name>.png/.gif          GP-Small 静态图/动画
  runs/viz_mt_compare_grandtour.gif    教师 vs GP-Small 并排动画
  控制台打印 N 目标随机场景成功率 (视频字幕素材)

用法: python viz_multitarget.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P
import viz
import viz_cluster6 as V

C = P.CAR_X; Y = P.CAR_Y
PAGE = '#0d0d0d'

def target_color(k, n):
    return V.V_CMAP(k / max(1, n - 1))

def draw_targets(ax, tgts, annotate=True):
    n = len(tgts)
    for k, g in enumerate(tgts):
        cg = target_color(k, n)
        ax.plot(g[0], g[1], marker='x', color=cg, ms=13, mew=3)
        circle = plt.Circle((g[0], g[1]), P.TOL, color=cg, fill=False, ls='--', lw=0.8, alpha=0.7)
        ax.add_patch(circle)
        if annotate:
            ax.annotate(f'g{k+1}', (g[0], g[1]), textcoords='offset points', xytext=(7, 7),
                        color=cg, fontsize=9, weight='bold')

def render_static(name, tgts, ta, gi, out_png):
    fig, ax = plt.subplots(figsize=(7, 7), facecolor=PAGE)
    ax.set_facecolor(PAGE)
    ax.set_xlim(0, P.FLD); ax.set_ylim(0, P.FLD); ax.set_aspect('equal')
    draw_targets(ax, tgts)
    V.speed_trail(ax, ta)
    for i in range(0, len(ta), 15):
        a = 0.15 + 0.85 * (i / len(ta))
        V.draw_t_car(ax, ta[i, 0], ta[i, 1], ta[i, 2], ta[i, 4], alpha=a*0.5)
    V.draw_t_car(ax, ta[-1, 0], ta[-1, 1], ta[-1, 2], ta[-1, 4], alpha=1.0)
    ax.set_title(f'{name} — {gi}/{len(tgts)} targets, {(len(ta)-1)*P.DT:.1f}s', color='white')
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor=PAGE)
    plt.close(fig)
    print(f'  -> {out_png}')

def render_gif(name, tgts, ta, gi, out_gif):
    fig, ax = plt.subplots(figsize=(7, 7), facecolor=PAGE)
    ax.set_facecolor(PAGE)
    ax.set_xlim(0, P.FLD); ax.set_ylim(0, P.FLD); ax.set_aspect('equal')
    draw_targets(ax, tgts)
    trail = LineCollection([], cmap=V.V_CMAP, norm=plt.Normalize(0, P.V_MAX))
    trail.set_linewidth(2.0); trail.set_alpha(0.85)
    ax.add_collection(trail)
    ax.set_title(f'{name} ({gi}/{len(tgts)})', color='white')
    car_objs = []
    stride = max(1, len(ta) // 400)
    frames = list(range(0, len(ta), stride))
    if len(ta)-1 not in frames: frames.append(len(ta)-1)

    def update(i):
        idx = frames[i]
        segs, v = V.speed_segments(ta, idx)
        trail.set_segments(segs); trail.set_array(v)
        for obj in car_objs: obj.remove()
        car_objs.clear()
        x, y, th, dlt = ta[idx, 0], ta[idx, 1], ta[idx, 2], ta[idx, 4]
        bx, by = V.BODY_HALF*np.cos(th), V.BODY_HALF*np.sin(th)
        l1, = ax.plot([x-bx, x+bx], [y-by, y+by], color='#ffcc00', lw=2.0)
        fx, fy = x+bx, y+by; wth = th + dlt
        wx, wy = V.WHEEL_HALF*np.cos(wth), V.WHEEL_HALF*np.sin(wth)
        l2, = ax.plot([fx-wx, fx+wx], [fy-wy, fy+wy], color='#ff4444', lw=2.5)
        car_objs.extend([l1, l2])
        return (trail,)
    ani = FuncAnimation(fig, update, frames=len(frames), interval=40, blit=False)
    ani.save(out_gif, writer=PillowWriter(fps=25))
    plt.close(fig)
    print(f'  -> {out_gif}')

def render_compare_gif(name, tgts, ta_t, gi_t, ta_g, gi_g, out_gif):
    """教师 vs GP-Small 同帧并排动画 (c6 风格)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), facecolor=PAGE)
    n_frames = max(len(ta_t), len(ta_g))
    stride = max(1, n_frames // 300)
    frames = list(range(0, n_frames, stride))
    if n_frames-1 not in frames: frames.append(n_frames-1)
    panels = {}
    def panel(ax, title, ta, gi):
        ax.set_facecolor(PAGE)
        ax.set_xlim(0, P.FLD); ax.set_ylim(0, P.FLD); ax.set_aspect('equal')
        draw_targets(ax, tgts, annotate=False)
        trail = LineCollection([], cmap=V.V_CMAP, norm=plt.Normalize(0, P.V_MAX))
        trail.set_linewidth(2.0); trail.set_alpha(0.85)
        ax.add_collection(trail)
        ax.set_title(title, color='white', fontsize=10)
        ax.tick_params(colors='#555555', labelsize=7)
        return trail
    trail_t = panel(ax1, f'Teacher ({gi_t}/{len(tgts)})', ta_t, gi_t)
    trail_g = panel(ax2, f'GP-Small ({gi_g}/{len(tgts)})', ta_g, gi_g)

    def update(i):
        idx = frames[i]
        for ax, ta, trail in ((ax1, ta_t, trail_t), (ax2, ta_g, trail_g)):
            j = min(idx, len(ta)-1)
            segs, v = V.speed_segments(ta, j)
            trail.set_segments(segs); trail.set_array(v)
            for obj in getattr(ax, '_car', []): obj.remove()
            x, y, th, dlt = ta[j, 0], ta[j, 1], ta[j, 2], ta[j, 4]
            bx, by = V.BODY_HALF*np.cos(th), V.BODY_HALF*np.sin(th)
            l1, = ax.plot([x-bx, x+bx], [y-by, y+by], color='#ffcc00', lw=2.0)
            fx, fy = x+bx, y+by; wth = th + dlt
            wx, wy = V.WHEEL_HALF*np.cos(wth), V.WHEEL_HALF*np.sin(wth)
            l2, = ax.plot([fx-wx, fx+wx], [fy-wy, fy+wy], color='#ff4444', lw=2.5)
            ax._car = [l1, l2]
    ani = FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False)
    ani.save(out_gif, writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f'  -> {out_gif}')

# ═══ 展示场景 (手工设计, 展示多目标泛化的不同侧面) ═══
def showcases():
    return [
        ('perimeter8', np.array([C, Y, 0., 0., 0.], dtype=np.float32),
         [np.array(p) for p in [(6.5,3.5),(6.5,6.5),(3.5,6.5),(0.5,6.5),(0.5,3.5),(0.5,0.5),(3.5,0.5),(6.5,0.5)]]),
        ('grandtour12', np.array([C, Y, 0., 1.0, 0.], dtype=np.float32),
         [np.array(p) for p in [(4.5,3.5),(6.0,5.5),(5.0,6.2),(2.0,6.0),(0.7,5.0),(1.5,2.8),(3.0,1.0),(5.5,0.8),(6.5,2.5),(4.0,4.5),(1.0,4.2),(3.5,5.0)]]),
        ('clusters6', np.array([C, Y, 0., 1.5, 0.], dtype=np.float32),
         [np.array(p) for p in [(5.2,4.6),(5.9,5.4),(4.6,5.6),(1.8,1.4),(1.1,2.2),(2.4,0.9)]]),
        ('zigzag6', np.array([C, Y, 0., 1.5, 0.], dtype=np.float32),
         [np.array(p) for p in [(1.0,1.0),(6.0,3.0),(1.5,5.5),(6.0,6.0),(1.0,3.0),(6.0,1.0)]]),
        ('behindchain5', np.array([C, Y, 0., 0.8, 0.], dtype=np.float32),
         [np.array(p) for p in [(3.5,1.5),(6.5,5.5),(0.8,6.2),(6.2,1.0),(3.5,3.0)]]),
        ('random6', np.array([C, Y, 0., 1.5, 0.], dtype=np.float32),
         [np.array(p) for p in [(1.2,5.7),(5.9,1.1),(0.9,1.4),(6.3,6.0),(3.4,2.5),(2.8,6.4)]]),
    ]

def bench_multi(teacher, dagger, n_scenes=100):
    """N 目标随机场景成功率 (字幕素材)."""
    print('\n=== N-target random benchmark (100 scenes each, seeds 700+) ===')
    print(f'{"targets":>8s} {"teacher":>10s} {"t_avg":>7s} {"gp-small":>10s} {"g_avg":>7s}')
    old_cap = P.MAX_EVAL_STEPS
    P.MAX_EVAL_STEPS = 1200   # 60s 上限, 多目标场景不因时限误判失败
    for N in (3, 4, 5, 6, 8):
        scenes = []
        for i in range(n_scenes):
            rng = np.random.RandomState(700 + N*100 + i)
            s0 = np.array([rng.uniform(P.M, P.FLD-P.M), rng.uniform(P.M, P.FLD-P.M), rng.uniform(-np.pi, np.pi),
                           rng.uniform(0.5, P.V_MAX), rng.uniform(-P.DELTA_MAX, P.DELTA_MAX)], dtype=np.float32)
            tg = [np.array([rng.uniform(P.M, P.FLD-P.M), rng.uniform(P.M, P.FLD-P.M)], dtype=np.float32) for _ in range(N)]
            scenes.append((s0, tg))
        s_t, t_t = P.eval_batch(teacher, False, scenes)
        s_g, t_g = P.eval_batch(dagger, True, scenes)
        print(f'{N:8d} {int(s_t.sum()):4d}/100 {t_t[s_t].mean():6.1f}s {int(s_g.sum()):4d}/100 {t_g[s_g].mean():6.1f}s')
    P.MAX_EVAL_STEPS = old_cap

def main():
    teacher = viz.load_teacher('runs/kamm533_teacher.pt')
    dagger = viz.load_gp('runs/gp_small_kamm533.pt')

    for name, s0, tgts in showcases():
        ta, _, gi = viz.run_traj(dagger, True, s0.copy(), tgts)
        print(f'{name}: {gi}/{len(tgts)} {(len(ta)-1)*P.DT:.1f}s')
        render_static(name, tgts, ta, gi, f'runs/viz_mt_{name}.png')
        render_gif(name, tgts, ta, gi, f'runs/viz_mt_{name}.gif')

    # 并排对比: 大巡回
    name, s0, tgts = 'grandtour12', None, None
    for nm, s0_, tg_ in showcases():
        if nm == 'grandtour12': name, s0, tgts = nm, s0_, tg_; break
    ta_t, _, gi_t = viz.run_traj(teacher, False, s0.copy(), tgts)
    ta_g, _, gi_g = viz.run_traj(dagger, True, s0.copy(), tgts)
    print(f'compare grandtour: teacher {gi_t}/{len(tgts)} {(len(ta_t)-1)*P.DT:.1f}s | gp {gi_g}/{len(tgts)} {(len(ta_g)-1)*P.DT:.1f}s')
    render_compare_gif(name, tgts, ta_t, gi_t, ta_g, gi_g, 'runs/viz_mt_compare_grandtour.gif')

    bench_multi(teacher, dagger)
    print('Done.')

if __name__ == '__main__':
    main()
