"""SelWP 机制演示 GIF: 基础教师 + 100% SelWP 注入 → 反打绕行.

完整复刻 run_p2_bc 的 rollout 逻辑 (单环境):
  - 不可达判定: in_circle (两圆) & R_min<0.5 速度门
  - 触发后 WP 注入 (车身系几何 → 世界坐标), WP 命中/目标命中后失效
  - 画面: T 车标 + 蓝→红速度轨迹 + 不可达双圆 + WP 星标 + 触发注释
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
import viz_cluster6 as V

# ═══ 场景: 目标侧后方近距 (v=1.0 → R_min=0.33<0.5, 落入转向圆心圆内 → 触发) ═══
S0 = np.array([3.5, 3.5, 0.0, 1.0, 0.0], dtype=np.float32)
TGTS = [np.array([3.2, 3.25]), np.array([4.0, 4.2]), np.array([4.8, 3.2])]

def rollout_selwp(frozen):
    """单环境 rollout, 逻辑与 run_p2_bc 完全一致 (WP 段确定性)."""
    sp = torch.tensor(S0, device=P.DEV, dtype=torch.float32).unsqueeze(0)
    gi = 0
    wp_pos = None; wp_triggered = [False]*3; wp_dead = True
    traj = [S0.copy()]; events = []      # events: (step, kind, data)
    for step in range(P.MAX_EVAL_STEPS):
        if gi >= 3: break
        i0 = min(gi, 2); i1 = min(gi+1, 2); i2 = min(gi+2, 2)
        cg = torch.tensor(TGTS[i0], device=P.DEV, dtype=torch.float32).unsqueeze(0)
        ng = torch.tensor(TGTS[i1], device=P.DEV, dtype=torch.float32).unsqueeze(0)
        nng = torch.tensor(TGTS[i2], device=P.DEV, dtype=torch.float32).unsqueeze(0)
        v2 = float(gi < 2); v3 = float(gi < 1)
        o1, o2, o3 = P.polar_obs(sp, cg, ng, nng)

        in_circle, dx_b, dy_b, R_min = P.unreachable_mask(sp, cg)
        if not wp_triggered[i0] and bool(in_circle.any()) and bool(R_min[0].item() < 0.5):
            wp_pos = P.compute_wp_world(sp, dx_b, dy_b, R_min)[0].cpu().numpy()
            wp_triggered[i0] = True; wp_dead = False
            events.append((step, 'wp', wp_pos.copy()))

        with torch.no_grad():
            if not wp_dead:
                o1w, o2w, o3w = P.polar_obs(sp, torch.tensor(wp_pos, device=P.DEV).unsqueeze(0), cg, ng)
                act, _ = frozen.forward(o1w, o2w, o3w, torch.ones(1, device=P.DEV), torch.tensor([v2], device=P.DEV))
            else:
                act, _ = frozen.forward(o1, o2, o3, torch.tensor([v2], device=P.DEV), torch.tensor([v3], device=P.DEV))
        sn = P.SIM.simulate(sp, act)
        snp = sn[0].cpu().numpy()
        # 命中: 目标 / WP
        atg = P.check_hit_substep_np(sp[0, :2].cpu().numpy(), snp[:2], TGTS[i0], P.TOL)
        if atg:
            wp_dead = True; gi += 1
            events.append((step, 'hit', gi))
        elif not wp_dead:
            wph = P.check_hit_substep_np(sp[0, :2].cpu().numpy(), snp[:2], wp_pos, P.TOL)
            if wph: wp_dead = True; events.append((step, 'wphit', None))
        traj.append(snp); sp = sn
    return np.array(traj), events, gi

# ═══ 不可达双圆 (世界坐标) ═══
def draw_unreachable(ax, x, y, th, R_min, alpha=0.7):
    for sgn in (1., -1.):
        cx = x + sgn*R_min*np.sin(th)      # 车身系 (0, ±R) → 世界
        cy = y - sgn*R_min*np.cos(th)
        ax.add_patch(plt.Circle((cx, cy), R_min, color='#66ccff', fill=False, ls=':', lw=1.0, alpha=alpha))

def main():
    m = P.GatedConcatActor().to(P.DEV); m.eval()
    m.load_state_dict(torch.load('runs/p1_base.pt', map_location=P.DEV, weights_only=False)['actor_state_dict'])
    P.set_teacher_deterministic(m)
    ta, events, gi = rollout_selwp(m)
    print(f'结果: {gi}/3 targets, {len(ta)} 步 ({len(ta)*P.DT:.1f}s)')
    # 从事件重建 WP 活跃区间
    ev_map = {s: k for s, k, _ in events}
    wp_active_steps = set()
    active = False
    for i in range(len(ta)):
        if ev_map.get(i) == 'wp': active = True
        if active: wp_active_steps.add(i)
        if active and ev_map.get(i) in ('hit', 'wphit'): active = False
    wp_xy = next(d for s, k, d in events if k == 'wp')

    fig, ax = plt.subplots(figsize=(7, 7), facecolor='#0d0d0d')
    ax.set_facecolor('#0d0d0d')
    ax.set_xlim(0, P.FLD); ax.set_ylim(0, P.FLD); ax.set_aspect('equal')
    for k, g in enumerate(TGTS):
        cg = '#ff4444' if k == 0 else ('#44ff44' if k == 2 else '#ffaa00')
        ax.plot(g[0], g[1], marker='x', color=cg, ms=13, mew=3)
        ax.annotate(f'g{k+1}', (g[0], g[1]), textcoords='offset points', xytext=(8, 8),
                    color=cg, fontsize=10, weight='bold')
    trail = LineCollection([], cmap=V.V_CMAP, norm=plt.Normalize(0, P.V_MAX))
    trail.set_linewidth(2.2); trail.set_alpha(0.85)
    ax.add_collection(trail)
    car_objs = []
    stride = max(1, len(ta) // 300)
    frames = list(range(0, len(ta), stride))
    if len(ta)-1 not in frames: frames.append(len(ta)-1)

    def update(i):
        idx = frames[i]
        segs, v = V.speed_segments(ta, idx)
        trail.set_segments(segs); trail.set_array(v)
        for obj in car_objs: obj.remove()
        car_objs.clear()
        x, y, th, vel, dlt = ta[idx, 0], ta[idx, 1], ta[idx, 2], ta[idx, 3], ta[idx, 4]
        R_min = vel**2 / P.A_LAT
        if R_min < 0.8:
            draw_unreachable(ax, x, y, th, R_min)
            car_objs.extend(ax.patches[-2:])
        if idx in wp_active_steps:
            star, = ax.plot(wp_xy[0], wp_xy[1], marker='*', color='#ffdd00', ms=16, mew=2)
            txt = ax.annotate('SelWP', (wp_xy[0], wp_xy[1]), textcoords='offset points',
                              xytext=(10, -14), color='#ffdd00', fontsize=10, weight='bold')
            car_objs.extend([star, txt])
        bx, by = V.BODY_HALF*np.cos(th), V.BODY_HALF*np.sin(th)
        l1, = ax.plot([x-bx, x+bx], [y-by, y+by], color='#ffcc00', lw=2.0)
        fx, fy = x+bx, y+by; wth = th + dlt
        wx, wy = V.WHEEL_HALF*np.cos(wth), V.WHEEL_HALF*np.sin(wth)
        l2, = ax.plot([fx-wx, fx+wx], [fy-wy, fy+wy], color='#ff4444', lw=2.5)
        car_objs.extend([l1, l2])
        return (trail,)
    ani = FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False)
    ani.save('runs/viz_selwp.gif', writer=PillowWriter(fps=20))
    plt.close(fig)
    # 静态定格 (触发瞬间后 20 步)
    fig, ax = plt.subplots(figsize=(7, 7), facecolor='#0d0d0d')
    ax.set_facecolor('#0d0d0d')
    ax.set_xlim(0, P.FLD); ax.set_ylim(0, P.FLD); ax.set_aspect('equal')
    for k, g in enumerate(TGTS):
        cg = '#ff4444' if k == 0 else ('#44ff44' if k == 2 else '#ffaa00')
        ax.plot(g[0], g[1], marker='x', color=cg, ms=13, mew=3)
        ax.annotate(f'g{k+1}', (g[0], g[1]), textcoords='offset points', xytext=(8, 8),
                    color=cg, fontsize=10, weight='bold')
    V.speed_trail(ax, ta)
    trig_step = next(s for s, k, _ in events if k == 'wp')
    snap = min(trig_step + 25, len(ta)-1)
    x, y, th, vel, dlt = ta[snap, 0], ta[snap, 1], ta[snap, 2], ta[snap, 3], ta[snap, 4]
    draw_unreachable(ax, x, y, th, vel**2/P.A_LAT, alpha=0.9)
    ax.plot(wp_xy[0], wp_xy[1], marker='*', color='#ffdd00', ms=16, mew=2)
    ax.annotate('SelWP', (wp_xy[0], wp_xy[1]), textcoords='offset points', xytext=(10, -14),
                color='#ffdd00', fontsize=10, weight='bold')
    V.draw_t_car(ax, x, y, th, dlt, alpha=1.0)
    ax.set_title('SelWP: unreachable zone -> counter-steer waypoint', color='white')
    fig.tight_layout()
    fig.savefig('runs/viz_selwp.png', dpi=130, facecolor='#0d0d0d')
    plt.close(fig)
    print('  -> runs/viz_selwp.gif / viz_selwp.png')
    print(f'  触发步: {trig_step}, WP 位置: ({wp_xy[0]:.2f}, {wp_xy[1]:.2f})')

if __name__ == '__main__':
    main()
