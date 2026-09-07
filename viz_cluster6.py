"""随机 6 目标场景可视化: 3 点聚类连续访问 + 3 点随机, "T" 形车体标记.

T 形标记:
  - 竖杆 (黄) = 车体朝向 θ
  - 横杆 (红) = 前轮方向 θ+δ, 锚在车头
  - δ=90° 时横杆 ⊥ 竖杆 → 完整 "T" (理论值, 实际 δ≤26.6°)

用法:
  python viz_cluster6.py                          # 默认 GP-Small, 4 个场景
  python viz_cluster6.py runs/kamm533_teacher.pt  # 指定任意 checkpoint (自动识别)
输出: runs/viz_c6_*.png, runs/viz_c6_*.gif
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

V_CMAP = LinearSegmentedColormap.from_list('v', ['#1a1aff', '#00ccff', '#ffff00', '#ff3300'])  # 蓝→青→黄→红

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P
import viz
from eval_ckpt import load_model

N_SCENES = 4
BODY_HALF = 0.15     # T 竖杆半长 (车体)
WHEEL_HALF = 0.14    # T 横杆半长 (前轮)
CLUSTER_R = (0.3, 0.5)   # 簇半径范围

# ═══ 场景生成: 前 3 点聚类连续, 后 3 点随机 ═══
def gen_scene(rng):
    cx = rng.uniform(1.8, P.FLD - 1.8); cy = rng.uniform(1.8, P.FLD - 1.8)
    ang0 = rng.uniform(0, 2*np.pi)
    pts = []
    for k in range(3):
        a = ang0 + k * 2*np.pi/3 + rng.uniform(-0.25, 0.25)
        d = rng.uniform(*CLUSTER_R) * rng.uniform(0.4, 1.0)
        pts.append(np.array([np.clip(cx + d*np.cos(a), P.M, P.FLD-P.M),
                             np.clip(cy + d*np.sin(a), P.M, P.FLD-P.M)]))
    pts.sort(key=lambda p: np.arctan2(p[1]-cy, p[0]-cx))   # 按角度排序, 簇内路径顺畅
    rand_pts = []
    while len(rand_pts) < 3:
        p = np.array([rng.uniform(P.M, P.FLD-P.M), rng.uniform(P.M, P.FLD-P.M)])
        if np.hypot(p[0]-cx, p[1]-cy) > CLUSTER_R[1] + 0.5: rand_pts.append(p)
    return pts + rand_pts, (cx, cy)

# ═══ T 形车体 ═══
def draw_t_car(ax, x, y, th, delta, alpha=1.0):
    bx, by = BODY_HALF*np.cos(th), BODY_HALF*np.sin(th)
    ax.plot([x-bx, x+bx], [y-by, y+by], color='#ffcc00', lw=2.0, alpha=alpha)   # 竖杆: 车体
    fx, fy = x+bx, y+by
    wth = th + delta
    wx, wy = WHEEL_HALF*np.cos(wth), WHEEL_HALF*np.sin(wth)
    ax.plot([fx-wx, fx+wx], [fy-wy, fy+wy], color='#ff4444', lw=2.5, alpha=alpha)  # 横杆: 前轮方向

# ═══ 速度着色轨迹 ═══
def speed_segments(ta, upto):
    """ta 前 upto 步的线段 + 对应速度 (供 LineCollection)."""
    if upto < 1: return np.empty((0, 2, 2)), np.empty(0)
    segs = np.stack([ta[:upto, :2], ta[1:upto+1, :2]], axis=1)
    return segs, ta[:upto, 3]

def speed_trail(ax, ta, label='v (m/s)'):
    segs, v = speed_segments(ta, len(ta)-1)
    lc = LineCollection(segs, cmap=V_CMAP, norm=plt.Normalize(0, P.V_MAX))
    lc.set_array(v); lc.set_linewidth(2.0); lc.set_alpha(0.8)
    ax.add_collection(lc)
    plt.colorbar(lc, ax=ax, label=label, fraction=0.046, pad=0.04)
    return lc

# ═══ 单场景静态图 ═══
def render_static(name, tgts, center, ta, out_png):
    fig, ax = plt.subplots(figsize=(7, 7), facecolor='#0d0d0d')
    ax.set_facecolor('#0d0d0d')
    ax.set_xlim(0, P.FLD); ax.set_ylim(0, P.FLD); ax.set_aspect('equal')
    # 目标点: 簇内蓝色 1-3, 随机绿色 4-6
    for k, g in enumerate(tgts):
        color = '#44aaff' if k < 3 else '#44ff88'
        ax.plot(g[0], g[1], marker='o' if k < 3 else 's', color=color, ms=9, mfc='none', mew=2)
        ax.annotate(f'{k+1}', (g[0], g[1]), textcoords='offset points', xytext=(7, 7),
                    color=color, fontsize=9, weight='bold')
    # 轨迹 (速度着色)
    speed_trail(ax, ta)
    # T 标记: 每隔 ~15 步淡色, 终点亮色
    for i in range(0, len(ta), 15):
        a = 0.15 + 0.85 * (i / len(ta))
        draw_t_car(ax, ta[i, 0], ta[i, 1], ta[i, 2], ta[i, 4], alpha=a*0.5)
    draw_t_car(ax, ta[-1, 0], ta[-1, 1], ta[-1, 2], ta[-1, 4], alpha=1.0)
    ax.set_title(f'{name} — 6 targets (1-3 cluster, 4-6 random)', color='white')
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor='#0d0d0d')
    plt.close(fig)
    print(f'  -> {out_png}')

# ═══ 单场景 GIF ═══
def render_gif(name, tgts, center, ta, out_gif):
    fig, ax = plt.subplots(figsize=(7, 7), facecolor='#0d0d0d')
    ax.set_facecolor('#0d0d0d')
    ax.set_xlim(0, P.FLD); ax.set_ylim(0, P.FLD); ax.set_aspect('equal')
    for k, g in enumerate(tgts):
        color = '#44aaff' if k < 3 else '#44ff88'
        ax.plot(g[0], g[1], marker='o' if k < 3 else 's', color=color, ms=9, mfc='none', mew=2)
        ax.annotate(f'{k+1}', (g[0], g[1]), textcoords='offset points', xytext=(7, 7),
                    color=color, fontsize=9, weight='bold')
    trail = LineCollection([], cmap=V_CMAP, norm=plt.Normalize(0, P.V_MAX))
    trail.set_linewidth(2.0); trail.set_alpha(0.8)
    ax.add_collection(trail)
    car_objs = []
    stride = max(1, len(ta) // 400)
    frames = list(range(0, len(ta), stride))
    if len(ta)-1 not in frames: frames.append(len(ta)-1)

    def update(i):
        idx = frames[i]
        segs, v = speed_segments(ta, idx)
        trail.set_segments(segs); trail.set_array(v)
        for obj in car_objs: obj.remove()
        car_objs.clear()
        x, y, th, dlt = ta[idx, 0], ta[idx, 1], ta[idx, 2], ta[idx, 4]
        bx, by = BODY_HALF*np.cos(th), BODY_HALF*np.sin(th)
        l1, = ax.plot([x-bx, x+bx], [y-by, y+by], color='#ffcc00', lw=2.0)
        fx, fy = x+bx, y+by; wth = th + dlt
        wx, wy = WHEEL_HALF*np.cos(wth), WHEEL_HALF*np.sin(wth)
        l2, = ax.plot([fx-wx, fx+wx], [fy-wy, fy+wy], color='#ff4444', lw=2.5)
        car_objs.extend([l1, l2])
        return (trail,)
    ani = FuncAnimation(fig, update, frames=len(frames), interval=40, blit=False)
    ani.save(out_gif, writer=PillowWriter(fps=25))
    plt.close(fig)
    print(f'  -> {out_gif}')

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'runs/gp_small_kamm533.pt'
    m, use_gp = load_model(path)
    m.eval()
    os.makedirs('runs', exist_ok=True)

    s0 = np.array([P.CAR_X, P.CAR_Y, 0., 1.5, 0.], dtype=np.float32)
    for si in range(N_SCENES):
        rng = np.random.RandomState(200 + si)
        tgts, center = gen_scene(rng)
        ta, acts, gi = viz.run_traj(m, use_gp, s0.copy(), tgts)
        tt = (len(ta)-1) * P.DT
        name = f'c6-{si}'
        print(f'{name}: {gi}/{len(tgts)} t={tt:.1f}s')
        render_static(name, tgts, center, ta, f'runs/viz_{name}.png')
        render_gif(name, tgts, center, ta, f'runs/viz_{name}.gif')
    print('Done.')

if __name__ == '__main__':
    main()
