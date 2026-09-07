"""新旧模型同场景同帧对比 GIF (2×2: 行=教师/GP, 列=旧v1/新v2).

用法: python viz_oldvsnew.py random6   (场景名 = viz_multitarget.showcases 的 key)
输出: runs/viz_oldvsnew_<scene>.gif / .png
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
from eval_ckpt import load_model
from viz_multitarget import showcases, draw_targets
from bench_loops import detect_loops

PAGE = '#0d0d0d'
PAIRS = [
    ('teacher', 'runs/backup_pre_fix/kamm533_teacher.pt', 'runs/kamm533_teacher.pt'),
    ('gp',      'runs/backup_pre_fix/gp_small_kamm533.pt', 'runs/gp_small_kamm533.pt'),
]

def panel(ax, title, tgts, ta, idx):
    ax.set_facecolor(PAGE)
    ax.set_xlim(0, P.FLD); ax.set_ylim(0, P.FLD); ax.set_aspect('equal')
    draw_targets(ax, tgts, annotate=False)
    trail = LineCollection([], cmap=V.V_CMAP, norm=plt.Normalize(0, P.V_MAX))
    trail.set_linewidth(1.8); trail.set_alpha(0.85)
    ax.add_collection(trail)
    ax.set_title(title, color='white', fontsize=9)
    ax.tick_params(colors='#555555', labelsize=6)
    j = min(idx, len(ta)-1)
    segs, v = V.speed_segments(ta, j)
    trail.set_segments(segs); trail.set_array(v)
    x, y, th, dlt = ta[j, 0], ta[j, 1], ta[j, 2], ta[j, 4]
    bx, by = V.BODY_HALF*np.cos(th), V.BODY_HALF*np.sin(th)
    l1, = ax.plot([x-bx, x+bx], [y-by, y+by], color='#ffcc00', lw=1.8)
    fx, fy = x+bx, y+by; wth = th + dlt
    wx, wy = V.WHEEL_HALF*np.cos(wth), V.WHEEL_HALF*np.sin(wth)
    l2, = ax.plot([fx-wx, fx+wx], [fy-wy, fy+wy], color='#ff4444', lw=2.2)
    ax._car = [l1, l2]
    return trail

def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else 'random6'
    s0 = tgts = None
    for nm, s0_, tg_ in showcases():
        if nm == scene: s0, tgts = s0_, tg_; break
    if s0 is None: raise ValueError(f'未知场景 {scene} (可选: {[n for n,_,_ in showcases()]})')

    runs = {}   # (kind, ver) -> (traj, gi)
    for kind, old_path, new_path in PAIRS:
        for ver, path in (('old', old_path), ('new', new_path)):
            if not os.path.exists(path):
                print(f'跳过 {ver} {kind}: {path} 不存在'); continue
            m, use_gp = load_model(path); m.eval()
            ta, _, gi = viz.run_traj(m, use_gp, s0.copy(), tgts)
            runs[(kind, ver)] = (ta, gi)

    keys = [('teacher', 'old'), ('teacher', 'new'), ('gp', 'old'), ('gp', 'new')]
    keys = [k for k in keys if k in runs]
    labels = {'teacher': 'Teacher', 'gp': 'GP-Small'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), facecolor=PAGE)
    axs = {( 'teacher', 'old'): axes[0, 0], ('teacher', 'new'): axes[0, 1],
           ('gp', 'old'): axes[1, 0], ('gp', 'new'): axes[1, 1]}
    trails = {}
    for k in keys:
        ta, gi = runs[k]
        ev, _ = detect_loops(ta, tgts)
        tt = (len(ta)-1)*P.DT
        trails[k] = panel(axs[k], f'{labels[k[0]]} {k[1].upper()} — {gi}/{len(tgts)}, {tt:.1f}s, loops={len(ev)}', tgts, ta, 0)

    n_frames = max(len(runs[k][0]) for k in keys)
    stride = max(1, n_frames // 300)
    frames = list(range(0, n_frames, stride))
    if n_frames-1 not in frames: frames.append(n_frames-1)

    def update(i):
        idx = frames[i]
        for k in keys:
            ta, gi = runs[k]
            j = min(idx, len(ta)-1)
            ax = axs[k]
            for obj in getattr(ax, '_car', []): obj.remove()
            trail = trails[k]
            segs, v = V.speed_segments(ta, j)
            trail.set_segments(segs); trail.set_array(v)
            x, y, th, dlt = ta[j, 0], ta[j, 1], ta[j, 2], ta[j, 4]
            bx, by = V.BODY_HALF*np.cos(th), V.BODY_HALF*np.sin(th)
            l1, = ax.plot([x-bx, x+bx], [y-by, y+by], color='#ffcc00', lw=1.8)
            fx, fy = x+bx, y+by; wth = th + dlt
            wx, wy = V.WHEEL_HALF*np.cos(wth), V.WHEEL_HALF*np.sin(wth)
            l2, = ax.plot([fx-wx, fx+wx], [fy-wy, fy+wy], color='#ff4444', lw=2.2)
            ax._car = [l1, l2]
    ani = FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False)
    out = f'runs/viz_oldvsnew_{scene}.gif'
    ani.save(out, writer=PillowWriter(fps=20))
    fig.savefig(out.replace('.gif', '.png'), dpi=110, facecolor=PAGE, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out}')
    for k in keys:
        ta, gi = runs[k]
        ev, _ = detect_loops(ta, tgts)
        print(f'  {k[0]:>8s} {k[1]:>3s}: {gi}/{len(tgts)} {(len(ta)-1)*P.DT:5.1f}s  loops={ev}')

if __name__ == '__main__':
    main()
