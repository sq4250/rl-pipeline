"""场景聚类仿真视频 (c6 风格): 随机场景按构型聚类, 每类取代表场景渲染.

风格 (同 viz_cluster6):
  - T 形车体标记: 竖杆(黄) = 车体朝向 θ, 横杆(红) = 前轮方向 θ+δ — 转角可视化
  - 速度色带: 轨迹按 v 着色 蓝→青→黄→红 (0~5 m/s)
  静态 PNG: Teacher vs GP-Small 并排对比 (c6 风格面板)
  GIF: GP-Small 动画 (T 形车体 + 速度色带 + 目标容差圈)

聚类特征: 初速 v0 / 首目标与车头偏角 ang0 / 连续段最大转角 maxleg / 最短腿 minleg
  C1 高速蛇形   C2 低速反打型   C3 中速大角   C4 高速直行   C5 短腿密集   C0 通用

用法: python viz_scene_clusters.py
输出: runs/viz_cluster_C*.png / .gif (覆盖同名文件)
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
import viz_compare

def wrap(a): return np.arctan2(np.sin(a), np.cos(a))

def feats(s0, tg):
    v0 = s0[3]
    h0 = np.arctan2(tg[0][1]-s0[1], tg[0][0]-s0[0])
    ang0 = abs(wrap(h0 - s0[2])) * 180/np.pi
    h1 = np.arctan2(tg[1][1]-tg[0][1], tg[1][0]-tg[0][0])
    h2 = np.arctan2(tg[2][1]-tg[1][1], tg[2][0]-tg[1][0])
    maxleg = max(abs(wrap(h1-h0)), abs(wrap(h2-h1))) * 180/np.pi
    minleg = min(np.hypot(*(tg[1]-tg[0])), np.hypot(*(tg[2]-tg[1])))
    return np.array([v0, ang0, maxleg, minleg])

# 优先级从上到下 (先匹配先归簇, 互斥)
CLUSTERS = [
    ('C1-highspeed-serpentine', lambda f: f[0] > 3.5 and f[1] > 120 and f[2] > 120),
    ('C2-lowv-countersteer',    lambda f: f[0] < 1.5 and f[1] > 90),
    ('C3-midv-bigangle',        lambda f: 1.5 <= f[0] <= 3.5 and f[1] > 90),
    ('C4-highv-straight',       lambda f: f[0] > 3.5 and f[1] < 45 and f[2] < 60),
    ('C5-shortlegs',            lambda f: f[3] < 0.8),
    ('C0-general',              lambda f: True),
]

def render_gif_t(name, tgts, ta, out_gif):
    """c6 风格动画: T 形车体 + 速度色带轨迹 + 目标容差圈."""
    fig, ax = plt.subplots(figsize=(7, 7), facecolor='#0d0d0d')
    ax.set_facecolor('#0d0d0d')
    ax.set_xlim(0, P.FLD); ax.set_ylim(0, P.FLD); ax.set_aspect('equal')
    for k, g in enumerate(tgts):
        cg = '#ff4444' if k == 0 else ('#44ff44' if k == len(tgts)-1 else '#ffaa00')
        ax.plot(g[0], g[1], marker='x', color=cg, ms=12, mew=2.5)
        ax.annotate(f'g{k+1}', (g[0], g[1]), textcoords='offset points', xytext=(7, 7),
                    color=cg, fontsize=9, weight='bold')
        circle = plt.Circle((g[0], g[1]), P.TOL, color=cg, fill=False, ls='--', lw=0.8, alpha=0.6)
        ax.add_patch(circle)
    trail = LineCollection([], cmap=V.V_CMAP, norm=plt.Normalize(0, P.V_MAX))
    trail.set_linewidth(2.0); trail.set_alpha(0.85)
    ax.add_collection(trail)
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

def main():
    pool = viz.random_scenes(600, seed=300)
    teacher = viz.load_teacher('runs/kamm533_teacher.pt')
    dagger = viz.load_gp('runs/gp_small_kamm533.pt')

    buckets = {c[0]: [] for c in CLUSTERS}
    for name, s0, tg in pool:
        f = feats(s0, tg)
        for cname, cond in CLUSTERS:
            if cond(f):
                buckets[cname].append((f, name, s0, tg)); break

    for cname, _ in CLUSTERS:
        b = buckets[cname]
        if not b: continue
        F = np.stack([x[0] for x in b])
        lo, hi = F.min(axis=0), F.max(axis=0)
        Fn = (F - lo) / (hi - lo + 1e-9)
        k = int(np.argmin(((Fn - Fn.mean(axis=0))**2).sum(axis=1)))
        f, name, s0, tg = b[k]
        tag = cname.split('-')[0]
        print(f'{cname}: {len(b)} scenes | rep={name} v0={f[0]:.1f} ang0={f[1]:.0f} maxleg={f[2]:.0f} minleg={f[3]:.2f}')
        ta_t, _, gi_t = viz.run_traj(teacher, False, s0.copy(), tg)
        ta_g, _, gi_g = viz.run_traj(dagger, True, s0.copy(), tg)
        # 静态并排 (c6 风格面板 + 共享速度色标)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), facecolor='#0d0d0d')
        viz_compare.panel(ax1, f'Teacher ({gi_t}/{len(tg)}, {(len(ta_t)-1)*P.DT:.1f}s)',
                          tg, ta_t, len(ta_t)-1, '#ffffff')
        lc = viz_compare.panel(ax2, f'GP-Small ({gi_g}/{len(tg)}, {(len(ta_g)-1)*P.DT:.1f}s)',
                               tg, ta_g, len(ta_g)-1, '#ffffff')
        fig.colorbar(lc, ax=[ax1, ax2], label='v (m/s)', fraction=0.046, pad=0.04)
        fig.savefig(f'runs/viz_cluster_{tag}.png', dpi=130, facecolor='#0d0d0d', bbox_inches='tight')
        plt.close(fig)
        print(f'  -> runs/viz_cluster_{tag}.png')
        render_gif_t(f'{tag} {name}', tg, ta_g, f'runs/viz_cluster_{tag}.gif')
    print('Done.')

if __name__ == '__main__':
    main()
