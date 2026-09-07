"""对比并排 GIF: Teacher vs GP-Small 同场景同帧, 速度色带同标尺.

用法: python viz_compare.py [场景名 c6-0..3 | tight-180]
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P
import viz
import viz_cluster6 as V
from eval_ckpt import load_model

PAGE = '#0d0d0d'

def build_scene(name):
    if name.startswith('c6'):
        rng = np.random.RandomState(200 + int(name.split('-')[1]))
        tgts, center = V.gen_scene(rng)
        return np.array([P.CAR_X, P.CAR_Y, 0., 1.5, 0.], dtype=np.float32), tgts
    if name == 'tight-180':
        return (np.array([0., 0., np.pi/4, 0., 0.], dtype=np.float32),
                [np.array([P.CAR_X+1.5, P.CAR_Y+1.5]), np.array([P.CAR_X+0.5, P.CAR_Y+3.0]), np.array([P.CAR_X-1.0, P.CAR_Y+1.5])])
    raise ValueError(name)

def panel(ax, title, tgts, ta, idx, color_acc):
    ax.set_facecolor(PAGE)
    ax.set_xlim(0, P.FLD); ax.set_ylim(0, P.FLD); ax.set_aspect('equal')
    for k, g in enumerate(tgts):
        cg = '#ff4444' if k == 0 else ('#44ff44' if k == len(tgts)-1 else '#ffaa00')
        ax.plot(g[0], g[1], marker='x', color=cg, ms=10, mew=2.5)
    trail = LineCollection([], cmap=V.V_CMAP, norm=plt.Normalize(0, P.V_MAX))
    trail.set_linewidth(2.0); trail.set_alpha(0.85)
    ax.add_collection(trail)
    segs, v = V.speed_segments(ta, idx)
    trail.set_segments(segs); trail.set_array(v)
    x, y, th, dlt = ta[idx, 0], ta[idx, 1], ta[idx, 2], ta[idx, 4]
    bx, by = V.BODY_HALF*np.cos(th), V.BODY_HALF*np.sin(th)
    ax.plot([x-bx, x+bx], [y-by, y+by], color='#ffcc00', lw=2.0)
    fx, fy = x+bx, y+by; wth = th + dlt
    wx, wy = V.WHEEL_HALF*np.cos(wth), V.WHEEL_HALF*np.sin(wth)
    ax.plot([fx-wx, fx+wx], [fy-wy, fy+wy], color='#ff4444', lw=2.5)
    ax.set_title(title, color='white', fontsize=10)
    ax.tick_params(colors='#555555', labelsize=7)
    return trail

def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else 'c6-2'
    s0, tgts = build_scene(scene)
    teacher, _ = load_model('runs/kamm533_teacher.pt'); teacher.eval()
    gp, _ = load_model('runs/gp_small_kamm533.pt'); gp.eval()
    ta_t, _, gi_t = viz.run_traj(teacher, False, s0.copy(), tgts)
    ta_g, _, gi_g = viz.run_traj(gp, True, s0.copy(), tgts)
    print(f'{scene}: teacher {gi_t}/{len(tgts)} {(len(ta_t)-1)*P.DT:.1f}s | GP-Small {gi_g}/{len(tgts)} {(len(ta_g)-1)*P.DT:.1f}s')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), facecolor=PAGE)
    n_frames = max(len(ta_t), len(ta_g))
    stride = max(1, n_frames // 300)
    frames = list(range(0, n_frames, stride))
    if n_frames-1 not in frames: frames.append(n_frames-1)

    def update(i):
        ax1.clear(); ax2.clear()
        idx = frames[i]
        panel(ax1, f'Teacher  ({gi_t}/{len(tgts)})', tgts, ta_t, min(idx, len(ta_t)-1), '#ffffff')
        panel(ax2, f'GP-Small ({gi_g}/{len(tgts)})', tgts, ta_g, min(idx, len(ta_g)-1), '#ffffff')
    ani = FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False)
    out = f'runs/viz_compare_{scene}.gif'
    ani.save(out, writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f'  -> {out}')

if __name__ == '__main__':
    main()
