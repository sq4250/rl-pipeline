"""教师专属高质量视频素材 (v2-current 教师, 无学生).

输出 runs/viz_t_*.png/.gif (+.mp4 若 ffmpeg 可用):
  6 个多目标展示场景 (perimeter8/grandtour12/clusters6/zigzag6/behindchain5/random6)
  3 个反打特写 (触发场景中反打步数最多的 3 个, 局部放大 ~2.5m 窗口)
风格: T 形车体 (黄=θ, 红=θ+δ) + 速度色带 (蓝→青→黄→红) + 目标按访问顺序同色系着色.
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
from viz_multitarget import showcases, target_color, draw_targets

PAGE = '#0d0d0d'
DPI = 160

def load_teacher(path='runs/kamm533_teacher.pt'):
    m = viz.load_teacher(path)
    return m

def cs_count(traj):
    if len(traj) < 3: return 0
    dth = np.diff(traj[:, 2]) / P.DT
    dth = np.arctan2(np.sin(dth), np.cos(dth))
    delta = traj[:-1, 4]
    cs = (np.abs(delta) > 0.1) & (np.abs(dth) > 0.15) & (np.sign(delta) != np.sign(dth))
    return int(cs.sum())

def pick_trigger_scenes(teacher, n_pick=3, pool=24):
    """触发场景池里挑"干净反打": 成功 3/3, 时长 5-14s (排除绕圈病理场景), 反打步数优先."""
    scored = []
    for i in range(pool):
        s0, tg = P.gen_trigger_scene(np.random.RandomState(3000+i))
        ta, _, gi = viz.run_traj(teacher, False, s0.copy(), tg)
        tt = (len(ta)-1)*P.DT
        if gi != len(tg) or not (5.0 <= tt <= 14.0):
            continue
        scored.append((cs_count(ta), tt, s0, tg, ta, gi))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[:n_pick]

def zoom_window(ta, size=2.5):
    x0, x1 = ta[:, 0].min(), ta[:, 0].max()
    y0, y1 = ta[:, 1].min(), ta[:, 1].max()
    cx, cy = (x0+x1)/2, (y0+y1)/2
    half = max(x1-x0, y1-y0, size) / 2 * 1.15
    return (cx-half, cx+half), (cy-half, cy+half)

def render_static(name, tgts, ta, gi, out_png, xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=PAGE)
    ax.set_facecolor(PAGE)
    ax.set_xlim(xlim if xlim else (0, P.FLD)); ax.set_ylim(ylim if ylim else (0, P.FLD))
    ax.set_aspect('equal')
    draw_targets(ax, tgts)
    V.speed_trail(ax, ta)
    for i in range(0, len(ta), 15):
        a = 0.15 + 0.85 * (i / len(ta))
        V.draw_t_car(ax, ta[i, 0], ta[i, 1], ta[i, 2], ta[i, 4], alpha=a*0.5)
    V.draw_t_car(ax, ta[-1, 0], ta[-1, 1], ta[-1, 2], ta[-1, 4], alpha=1.0)
    ax.set_title(f'{name} — {gi}/{len(tgts)}, {(len(ta)-1)*P.DT:.1f}s', color='white', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, facecolor=PAGE)
    plt.close(fig)
    print(f'  -> {out_png}')

def render_gif(name, tgts, ta, gi, out_gif, xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=PAGE)
    ax.set_facecolor(PAGE)
    ax.set_xlim(xlim if xlim else (0, P.FLD)); ax.set_ylim(ylim if ylim else (0, P.FLD))
    ax.set_aspect('equal')
    draw_targets(ax, tgts, annotate=(xlim is None))
    trail = LineCollection([], cmap=V.V_CMAP, norm=plt.Normalize(0, P.V_MAX))
    trail.set_linewidth(2.0); trail.set_alpha(0.85)
    ax.add_collection(trail)
    ax.set_title(f'{name} ({gi}/{len(tgts)})', color='white', fontsize=12)
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
        # 注意: draw_t_car 不返回线条对象, 这里必须显式创建并登记, 否则帧帧叠加拖影
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
        if r.returncode == 0:
            print(f'  -> {mp4}')
        else:
            print(f'  [ffmpeg skip] {r.stderr.decode()[:120]}')
    except Exception as e:
        print(f'  [no ffmpeg] {e}')

def main():
    teacher = load_teacher()

    print('=== 多目标展示场景 ===')
    for name, s0, tgts in showcases():
        ta, _, gi = viz.run_traj(teacher, False, s0.copy(), tgts)
        print(f'{name}: {gi}/{len(tgts)} {(len(ta)-1)*P.DT:.1f}s')
        render_static(name, tgts, ta, gi, f'runs/viz_t_{name}.png')
        render_gif(name, tgts, ta, gi, f'runs/viz_t_{name}.gif')
        to_mp4(f'runs/viz_t_{name}.gif', f'runs/viz_t_{name}.mp4')

    print('=== 反打特写 (触发场景, 干净反打筛选, 局部放大) ===')
    for rank, (cs, tt, s0, tg, ta, gi) in enumerate(pick_trigger_scenes(teacher)):
        xl, yl = zoom_window(ta)
        name = f'countersteer{rank}'
        print(f'{name}: {gi}/{len(tg)} {tt:.1f}s cs_steps={cs}')
        render_static(name, tg, ta, gi, f'runs/viz_t_{name}.png', xl, yl)
        render_gif(name, tg, ta, gi, f'runs/viz_t_{name}.gif', xl, yl)
        to_mp4(f'runs/viz_t_{name}.gif', f'runs/viz_t_{name}.mp4')
    print('Done.')

if __name__ == '__main__':
    main()
