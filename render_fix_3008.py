"""scene 3008 修复对比素材: v2 教师 (24.5s 绕圈) vs v3 教师 (6.1s 干净) 同帧并排.

输出: runs/viz_fix_3008.png/.gif/.mp4
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import pipeline as P
import viz
import viz_cluster6 as V
import viz_oldvsnew as VOV
import viz_teacher_hd as VH
from eval_ckpt import load_model

PAGE = '#0d0d0d'

def main():
    s0, tg = P.gen_trigger_scene(np.random.RandomState(3008))
    v2, _ = load_model('runs/backup_v2/kamm533_teacher.pt'); v2.eval()
    v3, _ = load_model('runs/kamm533_teacher.pt'); v3.eval()
    ta2, _, gi2 = viz.run_traj(v2, False, s0.copy(), tg)
    ta3, _, gi3 = viz.run_traj(v3, False, s0.copy(), tg)
    t2 = (len(ta2)-1)*P.DT; t3 = (len(ta3)-1)*P.DT
    print(f'scene 3008: v2 {gi2}/3 {t2:.1f}s | v3 {gi3}/3 {t3:.1f}s')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), facecolor=PAGE)
    tr2 = VOV.panel(ax1, f'v2 teacher — {t2:.1f}s (circling)', tg, ta2, 0)
    tr3 = VOV.panel(ax2, f'v3 teacher — {t3:.1f}s (clean)', tg, ta3, 0)
    n_frames = max(len(ta2), len(ta3))
    stride = max(1, n_frames // 300)
    frames = list(range(0, n_frames, stride))
    if n_frames-1 not in frames: frames.append(n_frames-1)

    def update(i):
        idx = frames[i]
        for ax, ta, trail in ((ax1, ta2, tr2), (ax2, ta3, tr3)):
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
    ani.save('runs/viz_fix_3008.gif', writer=PillowWriter(fps=20))
    fig.savefig('runs/viz_fix_3008.png', dpi=110, facecolor=PAGE, bbox_inches='tight')
    plt.close(fig)
    VH.to_mp4('runs/viz_fix_3008.gif', 'runs/viz_fix_3008.mp4')
    print('done')

if __name__ == '__main__':
    main()
