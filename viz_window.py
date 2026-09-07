"""三目标滑动窗口可视化: 仅当前输入窗口 (当前+后两个, 1~3 个) 的目标高亮, 其余降亮度.

演示"规划器接收当前目标和两个即将访问的目标, 小车推进时目标不足三个 → 门控截断" (脚本 60-71 句).
窗口成员: [gi, gi+1, gi+2] (超界截断); 窗口内 alpha=1.0, 窗口外 alpha=0.12.

用法: python viz_window.py [场景名]   # 默认全部 6 个展示场景
输出: runs/viz_w_<name>.gif/.mp4 (教师渲染)
"""
import os, sys, subprocess
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
from viz_multitarget import showcases, target_color

PAGE = '#0d0d0d'
WINDOW = 3          # 输入窗口大小 (当前 + 后两个)
DIM_ALPHA = 0.12

def gi_at_frames(traj, tgts):
    """逐帧计算当前目标索引 gi (命中检测与 rollout 同逻辑)."""
    gi = 0; gis = np.zeros(len(traj), dtype=int)
    for i in range(1, len(traj)):
        if gi < len(tgts) and P.check_hit_substep_np(traj[i-1][:2], traj[i][:2], tgts[gi], P.TOL):
            gi += 1
        gis[i] = min(gi, len(tgts)-1)
    return gis

def draw_targets_toggle(ax, tgts):
    """预创建目标艺术家, 返回 [ (marker, circle, text) × n ] 供逐帧调 alpha."""
    n = len(tgts); arts = []
    for k, g in enumerate(tgts):
        cg = target_color(k, n)
        mk, = ax.plot(g[0], g[1], marker='x', color=cg, ms=13, mew=3)
        circ = plt.Circle((g[0], g[1]), P.TOL, color=cg, fill=False, ls='--', lw=0.8, alpha=0.7)
        ax.add_patch(circ)
        tx = ax.annotate(f'g{k+1}', (g[0], g[1]), textcoords='offset points', xytext=(7, 7),
                         color=cg, fontsize=9, weight='bold')
        arts.append((mk, circ, tx))
    return arts

def window_mask(gi, n):
    m = np.zeros(n, dtype=bool)
    for k in range(gi, min(gi + WINDOW, n)):
        m[k] = True
    return m

def render_window_gif(name, tgts, ta, out_gif):
    gis = gi_at_frames(ta, tgts)
    n = len(tgts)
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=PAGE)
    ax.set_facecolor(PAGE)
    ax.set_xlim(0, P.FLD); ax.set_ylim(0, P.FLD); ax.set_aspect('equal')
    arts = draw_targets_toggle(ax, tgts)
    trail = LineCollection([], cmap=V.V_CMAP, norm=plt.Normalize(0, P.V_MAX))
    trail.set_linewidth(2.0); trail.set_alpha(0.85)
    ax.add_collection(trail)
    title = ax.set_title('', color='white', fontsize=12)
    car_objs = []
    stride = max(1, len(ta) // 400)
    frames = list(range(0, len(ta), stride))
    if len(ta)-1 not in frames: frames.append(len(ta)-1)

    def update(i):
        idx = frames[i]
        segs, v = V.speed_segments(ta, idx)
        trail.set_segments(segs); trail.set_array(v)
        gi = gis[idx]
        m = window_mask(gi, n)
        for k, (mk, circ, tx) in enumerate(arts):
            a = 1.0 if m[k] else DIM_ALPHA
            mk.set_alpha(a); circ.set_alpha(0.7 * a); tx.set_alpha(a)
        title.set_text(f'{name} — window g{gi+1}..g{min(gi+WINDOW, n)} ({m.sum()}/{n} lit)')
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

def to_mp4(gif, mp4):
    try:
        r = subprocess.run(['ffmpeg', '-y', '-i', gif, '-movflags', 'faststart',
                            '-pix_fmt', 'yuv420p', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', mp4],
                           capture_output=True, timeout=180)
        if r.returncode == 0: print(f'  -> {mp4}')
        else: print(f'  [ffmpeg skip] {r.stderr.decode()[:120]}')
    except Exception as e:
        print(f'  [no ffmpeg] {e}')

def main():
    sel = sys.argv[1:] if len(sys.argv) > 1 else [nm for nm, _, _ in showcases()]
    teacher = viz.load_teacher('runs/kamm533_teacher.pt')
    for name, s0, tgts in showcases():
        if name not in sel: continue
        ta, _, gi = viz.run_traj(teacher, False, s0.copy(), tgts)
        print(f'{name}: {gi}/{len(tgts)} {(len(ta)-1)*P.DT:.1f}s')
        render_window_gif(name, tgts, ta, f'runs/viz_w_{name}.gif')
        to_mp4(f'runs/viz_w_{name}.gif', f'runs/viz_w_{name}.mp4')
    print('Done.')

if __name__ == '__main__':
    main()
