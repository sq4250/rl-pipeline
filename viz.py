"""可视化: Kamm 5/3/3 Teacher vs GP-Small 轨迹/速度对比 + GIF 动画.

自包含: 直接复用 pipeline.py 的物理/网络/场景定义 (import pipeline as P).

用法:
  python viz.py                 # 4 标准场景 + 8 随机场景对比 PNG + GP-Small GIF
  python viz.py --models bc,dagger,teacher   # 对比 BC / 最优DAgger / Teacher
输出: runs/viz_*.png, runs/viz_*.gif
"""
import os, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P

plt.style.use('dark_background')

# ═══ 模型加载 ═══
def load_teacher(path):
    m = P.GatedConcatActor().to(P.DEV); m.eval()
    m.load_state_dict(torch.load(path, map_location=P.DEV, weights_only=False)['actor_state_dict'])
    P.set_teacher_deterministic(m)
    return m

def load_gp(path):
    m = P.GPSmall().to(P.DEV); m.eval()
    m.load_state_dict(torch.load(path, map_location=P.DEV, weights_only=False)['model_state_dict'])
    return m

AVAILABLE = {
    'teacher': lambda: load_teacher('runs/kamm533_teacher.pt'),
    'bc':      lambda: load_gp('runs/gp_small_kamm533_bc.pt'),
    'dagger':  lambda: load_gp('runs/gp_small_kamm533.pt'),
}

# ═══ 单环境 rollout (可视化专用, 每步 sync 可接受) ═══
def run_traj(actor, use_gp, s0, tgts, max_steps=600):   # 30s 上限, 与 pipeline.MAX_EVAL_STEPS 一致
    s = torch.tensor(s0, device=P.DEV, dtype=torch.float32).unsqueeze(0)
    gi = 0; traj = [s0.copy()]; acts = []
    for _ in range(max_steps):
        if gi >= len(tgts): break
        cur = torch.tensor(tgts[min(gi, len(tgts)-1)], device=P.DEV, dtype=torch.float32).unsqueeze(0)
        nxt = torch.tensor(tgts[min(gi+1, len(tgts)-1)], device=P.DEV, dtype=torch.float32).unsqueeze(0)
        nnxt = torch.tensor(tgts[min(gi+2, len(tgts)-1)], device=P.DEV, dtype=torch.float32).unsqueeze(0)
        v2 = torch.tensor([float(gi < len(tgts)-1)], device=P.DEV)
        v3 = torch.tensor([float(gi < len(tgts)-2)], device=P.DEV)
        with torch.no_grad():
            if use_gp:
                act = actor.deploy(P.polar_8d(s, cur, nxt, nnxt), v2, v3)
            else:
                o1, o2, o3 = P.polar_obs(s, cur, nxt, nnxt)
                act, _ = actor.forward(o1, o2, o3, v2, v3)
        sn = P.SIM.simulate(s, act)
        snp = sn[0].cpu().numpy()
        a_eff = np.array([(snp[3]-traj[-1][3])/P.DT, (snp[4]-traj[-1][4])/P.DT])
        acts.append(a_eff)
        if P.check_hit_substep_np(traj[-1][:2], snp[:2], tgts[gi], P.TOL): gi += 1
        traj.append(snp); s = sn
    return np.array(traj), np.array(acts), gi

# ═══ 场景 ═══
def standard_scenes():
    C = P.CAR_X; Y = P.CAR_Y
    return [
        ('tight-180', np.array([0., 0., np.pi/4, 0., 0.], dtype=np.float32),
         [np.array([C+1.5, Y+1.5]), np.array([C+0.5, Y+3.0]), np.array([C-1.0, Y+1.5])]),
        ('tight-TR', np.array([0., 0., 0., 0., 0.], dtype=np.float32),
         [np.array([C+2.0, Y+0.0]), np.array([C+2.5, Y+2.5]), np.array([C+0.5, Y+2.0])]),
        ('anti-steer', np.array([C, Y, 0., 2.0, 0.], dtype=np.float32),
         [np.array([C-1.0, Y+1.5]), np.array([C+1.0, Y+4.0]), np.array([C+3.0, Y+2.0])]),
        ('straight', np.array([C, Y, 0., 0., 0.], dtype=np.float32),
         [np.array([C+2.0, Y+0.1]), np.array([C+4.0, Y-0.1]), np.array([C+6.0, Y+0.0])]),
    ]

def random_scenes(n, seed=42):
    scenes = []
    for i in range(n):
        rng = np.random.RandomState(seed + i)
        s0 = np.array([rng.uniform(P.M, P.FLD-P.M), rng.uniform(P.M, P.FLD-P.M),
                       rng.uniform(-np.pi, np.pi), rng.uniform(0.5, P.V_MAX),
                       rng.uniform(-P.DELTA_MAX, P.DELTA_MAX)], dtype=np.float32)
        tg = [np.array([rng.uniform(P.M, P.FLD-P.M), rng.uniform(P.M, P.FLD-P.M)], dtype=np.float32) for _ in range(3)]
        scenes.append((f'rand-{i}', s0, tg))
    return scenes

# ═══ 渲染 ═══
CAR_LEN, CAR_WID = 0.30, 0.18

def draw_car(ax, xy, th, color):
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    corners = np.array([[-CAR_LEN/2, -CAR_WID/2], [CAR_LEN/2, -CAR_WID/2],
                        [CAR_LEN/2, CAR_WID/2], [-CAR_LEN/2, CAR_WID/2],
                        [-CAR_LEN/2, -CAR_WID/2]]) @ R.T + xy
    ax.plot(corners[:, 0], corners[:, 1], color=color, lw=1.5)
    ax.plot([xy[0], xy[0] + CAR_LEN/2*np.cos(th)], [xy[1], xy[1] + CAR_LEN/2*np.sin(th)], color=color, lw=2.0)

def render_compare(name, s0, tgts, models, out_png):
    fig, (ax_t, ax_v) = plt.subplots(1, 2, figsize=(15, 5.2), facecolor='#0d0d0d',
                                     gridspec_kw={'width_ratios': [1.35, 0.65]})
    ax_t.set_facecolor('#0d0d0d'); ax_v.set_facecolor('#0d0d0d')
    ax_t.set_xlim(0, P.FLD); ax_t.set_ylim(0, P.FLD); ax_t.set_aspect('equal')
    max_t = 0
    for tag, actor, use_gp, color in models:
        ta, acts, gi = run_traj(actor, use_gp, s0, tgts)
        tt = (len(ta)-1) * P.DT
        ax_t.plot(ta[:, 0], ta[:, 1], color=color, lw=1.8, alpha=0.75, label=f'{tag} ({gi}/{len(tgts)} {tt:.1f}s)')
        ax_v.plot(np.arange(len(ta))*P.DT, ta[:, 3], color=color, lw=1.5)
        max_t = max(max_t, tt)
    for k, g in enumerate(tgts):
        cg = '#ff4444' if k == 0 else ('#44ff44' if k == len(tgts)-1 else '#ffaa00')
        ax_t.plot(g[0], g[1], marker='x', color=cg, ms=12, mew=2.5)
        ax_t.text(g[0]+0.12, g[1]+0.12, f'g{k+1}', color=cg, fontsize=7, weight='bold')
    ax_t.legend(fontsize=7, labelcolor='white', facecolor='#1a1a1a', edgecolor='#444444')
    ax_t.set_title(f'{name} — Kamm 5/3/3', color='white')
    ax_v.set_xlim(0, max_t + 0.3); ax_v.set_ylim(0, P.V_MAX + 0.3)
    ax_v.axhline(y=P.V_MAX, color='#666666', ls='--', lw=0.5)
    ax_v.set_xlabel('t(s)', color='#aaaaaa'); ax_v.set_title('Speed', color='white')
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor='#0d0d0d')
    plt.close(fig)
    print(f'  -> {out_png}')

def render_gif(name, s0, tgts, actor, use_gp, out_gif):
    """GP-Small 动画: 车体 + 目标 + 轨迹."""
    ta, acts, gi = run_traj(actor, use_gp, s0, tgts)
    fig, ax = plt.subplots(figsize=(7, 7), facecolor='#0d0d0d')
    ax.set_facecolor('#0d0d0d')
    ax.set_xlim(0, P.FLD); ax.set_ylim(0, P.FLD); ax.set_aspect('equal')
    for k, g in enumerate(tgts):
        cg = '#ff4444' if k == 0 else ('#44ff44' if k == len(tgts)-1 else '#ffaa00')
        ax.plot(g[0], g[1], marker='x', color=cg, ms=14, mew=3)
        circle = plt.Circle((g[0], g[1]), P.TOL, color=cg, fill=False, ls='--', lw=0.8, alpha=0.7)
        ax.add_patch(circle)
    trail, = ax.plot([], [], color='#66ccff', lw=1.4, alpha=0.6)
    ax.set_title(f'{name} ({gi}/{len(tgts)})', color='white')
    stride = max(1, len(ta) // 400)   # 最多 ~400 帧
    frames = list(range(0, len(ta), stride))
    if len(ta)-1 not in frames: frames.append(len(ta)-1)
    car_objects = []   # 每帧重建的车体元素

    def update(i):
        idx = frames[i]
        trail.set_data(ta[:idx+1, 0], ta[:idx+1, 1])
        for obj in car_objects: obj.remove()
        car_objects.clear()
        ln, = ax.plot([ta[idx, 0], ta[idx, 0]+CAR_LEN/2*np.cos(ta[idx, 2])],
                      [ta[idx, 1], ta[idx, 1]+CAR_LEN/2*np.sin(ta[idx, 2])], color='#ffcc00', lw=2.0)
        rect = Rectangle((ta[idx, 0]-CAR_LEN/2, ta[idx, 1]-CAR_WID/2), CAR_LEN, CAR_WID,
                         angle=np.degrees(ta[idx, 2]), fc='#ffcc00', ec='white', lw=0.8, alpha=0.9)
        ax.add_patch(rect)
        car_objects.extend([ln, rect])
        return (trail,)
    ani = FuncAnimation(fig, update, frames=len(frames), interval=40, blit=False)
    ani.save(out_gif, writer=PillowWriter(fps=25))
    plt.close(fig)
    print(f'  -> {out_gif}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='bc,dagger,teacher', help='逗号分隔: bc / dagger / teacher')
    ap.add_argument('--no-gif', action='store_true')
    args = ap.parse_args()
    os.makedirs('runs', exist_ok=True)

    tags = [t.strip() for t in args.models.split(',')]
    models = []
    for tag in tags:
        if tag not in AVAILABLE:
            print(f'未知模型: {tag} (可选: {list(AVAILABLE)})'); sys.exit(1)
        actor = AVAILABLE[tag]()
        models.append((tag, actor, tag != 'teacher', None))
    colors = {'teacher': '#ff6666', 'bc': '#66ccff', 'dagger': '#44ff44'}
    models = [(tag, a, ug, colors[tag]) for tag, a, ug, _ in models]

    # 标准场景
    for name, s0, tg in standard_scenes():
        render_compare(name, s0, tg, models, f'runs/viz_{name}.png')
        if not args.no_gif:
            gp_actor = next(a for tag, a, ug, _ in models if ug)
            render_gif(name, s0, tg, gp_actor, True, f'runs/viz_{name}.gif')
    # 随机场景 (只出对比图)
    for name, s0, tg in random_scenes(8):
        render_compare(name, s0, tg, models, f'runs/viz_{name}.png')
    print('Done.')

if __name__ == '__main__':
    main()
