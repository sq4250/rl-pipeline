"""视频素材: 训练四阶段 + 访问逻辑可视化 (16:9 高清).

图1 viz_stages.png     四阶段卡片链 (冷启动 → SelWP → 纯PPO → 蒸馏)
图2 viz_p2_flow.png    P2 数据流: 冻结教师+WP 示范 → 真实观测记录 → 空白双网络
图3 viz_distill.png    蒸馏: Teacher 76.7K → BC/DAgger×12 两级选优 → GP-Small 3.7K
图4 viz_visit.png      访问逻辑: 活跃/探索点池 → TSP 队列 → 三目标滑动窗口

输出: runs/viz_*.png (2880×1620 @150dpi)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P

SURFACE = '#1a1a19'; PAGE = '#0d0d0d'
INK = '#ffffff'; INK2 = '#c3c2b7'; MUTED = '#898781'
BORDER = '#3a3a37'; ZERO = '#3f3f3c'
C_BLUE = '#3987e5'; C_ORANGE = '#d95926'; C_AQUA = '#199e70'; C_YELLOW = '#c98500'

def box(ax, x, y, w, h, text, fc, ec=BORDER, fs=8.5, tc=INK, bold=False, lw=1.2, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.015,rounding_size=0.03',
                                fc=fc, ec=ec, lw=lw, linestyle=ls))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', color=tc, fontsize=fs,
            weight='bold' if bold else 'normal', linespacing=1.4)

def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=12,
                                 color=color, lw=lw, shrinkA=2, shrinkB=2))

def panel(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.05',
                                fc=SURFACE, ec=BORDER, lw=1.4))
    ax.text(x+0.25, y+h-0.42, title, color=INK, fontsize=11.5, weight='bold')

def badge(ax, x, y, w, text, color):
    box(ax, x, y, w, 0.5, text, ZERO, ec=color, fs=7.5, tc=color, bold=True)

# ═══════════════ 图1: 四阶段卡片链 ═══════════════
def stages():
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=PAGE)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98]); ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 19.2); ax.set_ylim(0, 10.8); ax.axis('off')
    ax.text(0.5, 10.1, 'Four-stage training pipeline — cold start to deployment', color=INK,
            fontsize=16, weight='bold')
    ax.text(0.5, 9.45, 'P1 基础网络只是示范教师; 最终策略的血统: 空白网络 → PPO → 蒸馏', color=INK2,
            fontsize=10.5)

    cards = [
        ('Phase A', 'cold start', C_BLUE,
         ['TOL course 2.0 → 0.10', 'polar explore (+long legs)', 'anneal to deterministic'],
         ['narrow → full state', '1300 iters'], 'runs/p1_base.pt'),
        ('Phase B', 'SelWP fusion', C_ORANGE,
         ['frozen P1: rollout only', '100% SelWP countersteer demo', 'blank actor + critic from scratch'],
         ['frozen', 'WP ★', 'zero-weight', '300 iters'], 'runs/p2_bc.pt'),
        ('Phase C', 'pure PPO fine-tune', C_AQUA,
         ['noise breaks deadlocks', 'random ↔ polar alternation', 'anneal converges'],
         ['no WP', 'full state', '1000 iters'], 'runs/kamm533_teacher.pt'),
        ('Phase D', 'distill to deploy', C_YELLOW,
         ['BC distance-weighted', 'DAgger ×12', 'two-stage selection (rand → speed)'],
         ['3.7K params', 'deterministic', '→ MCU'], 'runs/gp_small.pt'),
    ]
    x = 0.5
    for name, sub, color, lines, bgs, out in cards:
        # 标题块
        box(ax, x, 7.9, 4.3, 1.15, f'{name}\n{sub}', color, fs=10.5, bold=True)
        # 机制行
        for i, line in enumerate(lines):
            box(ax, x, 6.6 - i*0.75, 4.3, 0.62, line, SURFACE, ec=BORDER, fs=8.2, tc=INK2)
        # badges
        bx = x
        for t in bgs:
            bw = 0.62 + 0.115*len(t)
            badge(ax, bx, 3.55, bw, t, color)
            bx += bw + 0.12
        # 产出
        box(ax, x, 2.55, 4.3, 0.62, out, ZERO, ec=color, fs=8.2, tc=color)
        if x > 0.5:
            arrow(ax, x-0.22, 8.45, x, 8.45, color=INK2, lw=1.4)
        x += 4.3 + 0.42
    ax.text(0.5, 1.55, 'every stage outputs a checkpoint — resume from any point (--from b/c/d)',
            color=MUTED, fontsize=9, style='italic')
    fig.savefig('runs/viz_stages.png', dpi=150, facecolor=PAGE, bbox_inches='tight')
    plt.close(fig)
    print('  -> runs/viz_stages.png')

# ═══════════════ 图2: P2 数据流 ═══════════════
def p2_flow():
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=PAGE)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98]); ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 19.2); ax.set_ylim(0, 10.8); ax.axis('off')
    ax.text(0.5, 10.1, 'Phase B — how the demo becomes the blank nets', color=INK,
            fontsize=16, weight='bold')

    # ── 左: 示范产生 ──
    panel(ax, 0.5, 1.6, 5.6, 8.0, '1. produce the demo')
    ax2 = fig.add_axes([0.75, 2.5, 3.3, 3.3]); ax2.set_facecolor(SURFACE)
    ax2.set_xlim(0, 5); ax2.set_ylim(0, 5); ax2.set_aspect('equal'); ax2.axis('off')
    cx, cy, R = 2.5, 2.3, 0.95
    # 不可达双圆
    for sgn in (1, -1):
        ax2.add_patch(Circle((cx, cy - sgn*R), R, fc='#14315a', ec=C_BLUE, ls='--', lw=1.3, alpha=0.5))
    # 车 (T 标)
    ax2.plot([cx-0.35, cx+0.35], [cy, cy], color='#ffcc00', lw=2.5)
    ax2.plot([cx+0.35, cx+0.35], [cy-0.3, cy+0.3], color='#ff4444', lw=2.5)
    ax2.text(cx-0.3, cy+0.55, 'car', color='#ffcc00', fontsize=9)
    # g1 在圆内, WP 在外
    g1 = (cx+0.42, cy-0.52)
    ax2.plot(*g1, marker='x', color='#ff4444', ms=11, mew=3)
    ax2.text(g1[0]+0.12, g1[1]-0.28, 'g1 (unreachable)', color='#ff4444', fontsize=8.5)
    wp = (cx+1.5, cy+0.9)
    ax2.text(wp[0], wp[1], '★', color='#ffcc00', fontsize=16, ha='center', va='center')
    ax2.text(wp[0]+0.28, wp[1]+0.05, 'WP (tangent\npoint of CC arcs)', color='#ffcc00', fontsize=8)
    # 反打轨迹示意 (两弧)
    t = np.linspace(-1.05, 0.35, 40)
    ax2.plot(cx + R*np.cos(t), cy - R + R*np.sin(t), color='#66ccff', lw=2.0, ls='-')
    t2 = np.linspace(-2.3, -0.9, 40)
    ax2.plot(wp[0] + 0.95*np.cos(t2 + 0.8), wp[1] + 0.95*np.sin(t2 + 0.8), color='#66ccff', lw=2.0)
    ax2.text(cx-0.15, cy-1.5, 'countersteer: turn away first,\nthen circle back', color='#66ccff', fontsize=8)
    box(ax, 0.75, 2.6, 5.1, 0.7, 'frozen P1 teacher (rollout only)', ZERO, ec=C_ORANGE, fs=9, tc=C_ORANGE, bold=True)
    box(ax, 0.75, 1.85, 5.1, 0.6, 'trigger: goal inside circle AND speed gate R_min<0.5', SURFACE, ec=BORDER, fs=7.8, tc=INK2)
    box(ax, 0.75, 6.6, 5.1, 1.5, 'SelWP injects the waypoint\nonly to query the ACTION', ZERO, ec=C_YELLOW, fs=9.5, tc='#ffcc00', bold=True)
    arrow(ax, 3.3, 6.6, 3.3, 6.25, color=C_YELLOW)

    # ── 中: 记录三件套 ──
    panel(ax, 6.7, 1.6, 5.6, 8.0, '2. record under real observation')
    items = [
        ('real observation o', '8D polar (real goals only)', C_BLUE, 'drives BC input'),
        ('demo action a_eff', 'post-sim effective action\n(contains the countersteer)', C_ORANGE, 'BC target'),
        ('reward → GAE', 'bootstrapped by the blank\ncritic itself', C_AQUA, 'value loss target'),
    ]
    for i, (t, sub, c, use) in enumerate(items):
        y = 7.9 - i*2.1
        box(ax, 7.0, y, 5.0, 1.35, f'{t}\n{sub}', SURFACE, ec=c, fs=9, tc=INK)
        box(ax, 7.0, y-0.55, 5.0, 0.5, use, ZERO, ec=ZERO, fs=7.5, tc=INK2)
    ax.text(7.0, 2.05, 'the waypoint never enters any\nnetwork input — it only chooses\nwhich states get visited', color='#ffcc00',
            fontsize=9.5, ha='center', weight='bold')

    # ── 中→右 箭头
    arrow(ax, 12.35, 8.3, 12.85, 8.3, color=INK2, lw=1.5)
    arrow(ax, 12.35, 6.2, 12.85, 6.2, color=INK2, lw=1.5)
    arrow(ax, 12.35, 4.1, 12.85, 4.1, color=INK2, lw=1.5)

    # ── 右: 空白双网络 ──
    panel(ax, 12.9, 1.6, 5.8, 8.0, '3. blank nets learn')
    box(ax, 13.2, 6.9, 5.2, 1.6, 'blank ACTOR (fresh init)\nBC: weighted MSE on a_eff\nw = 1/(d1+0.15)',
        ZERO, ec=C_BLUE, fs=9, ls='--', bold=True)
    box(ax, 13.2, 3.6, 5.2, 1.6, 'blank CRITIC (fresh init)\nPPO clipped value loss\non rollout GAE',
        ZERO, ec=C_AQUA, fs=9, ls='--', bold=True)
    badge(ax, 13.2, 8.75, 1.35, 'zero-weight', C_ORANGE)
    badge(ax, 14.7, 8.75, 1.05, 'no P1', C_ORANGE)
    ax.text(15.8, 8.9, 'P1 weights are\ndiscarded here', color=C_ORANGE, fontsize=8.5)
    ax.text(15.8, 5.5, 'P1 critic never used —\nGAE bootstraps from\nthe blank critic itself', color=MUTED, fontsize=8.5)
    ax.text(15.8, 2.35, 'next: pure PPO on this\npair (Phase C), no WP', color=C_AQUA, fontsize=9)
    fig.savefig('runs/viz_p2_flow.png', dpi=150, facecolor=PAGE, bbox_inches='tight')
    plt.close(fig)
    print('  -> runs/viz_p2_flow.png')

# ═══════════════ 图3: 蒸馏 ═══════════════
def distill():
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=PAGE)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98]); ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 19.2); ax.set_ylim(0, 10.8); ax.axis('off')
    ax.text(0.5, 10.1, 'Phase D — distill the teacher into a 3.7K chip model', color=INK,
            fontsize=16, weight='bold')

    box(ax, 0.6, 4.6, 3.2, 2.0, 'Teacher\nGatedConcatActor\n76.7K params\nGELU, stochastic', '#14315a', ec=C_BLUE, fs=9.5, bold=True)

    # BC 路径 (上)
    box(ax, 5.0, 7.3, 3.4, 1.3, 'BC (200 ep, lr 2e-3)\nnoise 0.03 · distance-\nweighted w=1/(d+0.15)', SURFACE, ec=C_ORANGE, fs=8.5)
    arrow(ax, 3.8, 6.0, 5.0, 7.6, color=C_ORANGE)
    # DAgger 路径 (下, 带回流)
    box(ax, 5.0, 3.9, 3.4, 1.3, 'DAgger ×12 (100 ep)\nstudent rolls out,\nteacher labels on-policy\nlr 5e-4 → 2e-4', SURFACE, ec=C_ORANGE, fs=8.5)
    arrow(ax, 3.8, 5.0, 5.0, 4.5, color=C_ORANGE)
    arrow(ax, 8.4, 4.55, 12.6, 4.55, color=C_ORANGE, lw=0.9)
    # 学生
    box(ax, 12.6, 4.0, 3.0, 1.6, 'Student\nGP-Small\n3.7K params\nReLU, deterministic', '#0f4a37', ec=C_AQUA, fs=9.5, bold=True)
    arrow(ax, 8.4, 7.95, 14.1, 5.7, color=C_ORANGE)
    # 两级选优
    box(ax, 10.2, 6.6, 4.6, 1.6, 'two-stage selection\neach round: (1) random succ\n(seeds 500-599) (2) tight time', SURFACE, ec=C_YELLOW, fs=8.8, tc='#ffcc00', bold=True)
    arrow(ax, 15.6, 5.6, 14.1, 6.6, color=C_YELLOW)
    arrow(ax, 12.5, 7.4, 10.2, 7.4, color=C_YELLOW)
    # 部署
    box(ax, 16.4, 4.6, 2.2, 1.8, 'deploy\n→ MCU\nnormalization\nfused into\nweights', ZERO, ec=C_YELLOW, fs=8.5, tc='#ffcc00', bold=True)
    arrow(ax, 15.6, 4.8, 16.4, 5.3, color=C_YELLOW)
    # 数据徽章
    badge(ax, 0.6, 1.9, 3.4, 'teacher: random100 = 1.00, probe 40/40', C_BLUE)
    badge(ax, 5.0, 1.9, 3.6, 'selected D3: 5.83s, rand = 1.00', C_AQUA)
    badge(ax, 9.4, 1.9, 4.2, 'student: random100 = 1.00 (3762 params)', C_YELLOW)
    badge(ax, 14.4, 1.9, 3.4, 'N-target 3-8: 500/500 scenes', C_ORANGE)
    ax.text(0.6, 1.05, 'same polar input; the student sees full-speed full-steer scenes (gen_mixed, long-leg mixture)',
            color=MUTED, fontsize=9, style='italic')
    fig.savefig('runs/viz_distill.png', dpi=150, facecolor=PAGE, bbox_inches='tight')
    plt.close(fig)
    print('  -> runs/viz_distill.png')

# ═══════════════ 图4: 访问逻辑 ═══════════════
def visit():
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=PAGE)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98]); ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 19.2); ax.set_ylim(0, 10.8); ax.axis('off')
    ax.text(0.5, 10.1, 'Visit logic — pools, TSP queue, sliding window', color=INK,
            fontsize=16, weight='bold')

    # 池
    box(ax, 0.6, 5.6, 2.6, 2.4, 'active pool\n(lit beacons)\n\ncandidate →\nconfirmed (id)', SURFACE, ec=C_AQUA, fs=8.8)
    box(ax, 0.6, 1.7, 2.6, 2.4, 'explore pool\n(optional\nwaypoints)', SURFACE, ec=C_BLUE, fs=8.8)
    # TSP
    box(ax, 4.3, 6.2, 2.9, 1.7, 'TSP sort\nactive points\n(from car / current\ntarget, anti-jump)', SURFACE, ec=C_ORANGE, fs=8.3)
    box(ax, 4.3, 2.9, 2.9, 1.7, 'TSP sort\nexplore points\n(append after\nactive queue)', SURFACE, ec=C_ORANGE, fs=8.3)
    arrow(ax, 3.2, 6.5, 4.3, 7.0, color=C_AQUA)
    arrow(ax, 3.2, 3.3, 4.3, 3.8, color=C_BLUE)
    # 队列
    box(ax, 8.2, 4.3, 3.2, 2.0, 'visit queue\n(ids → coords\nanti-jump)', SURFACE, ec=C_YELLOW, fs=9, tc='#ffcc00', bold=True)
    arrow(ax, 7.2, 7.0, 8.2, 5.6, color=C_ORANGE)
    arrow(ax, 7.2, 3.8, 8.2, 4.9, color=C_ORANGE)
    # 滑动窗口
    box(ax, 12.4, 5.9, 2.9, 1.9, '3-target window\n[g_i, g_i+1, g_i+2]\ngates v2 v3', SURFACE, ec=C_BLUE, fs=8.8)
    arrow(ax, 11.4, 5.3, 12.4, 6.5, color=C_YELLOW)
    # 规划器
    box(ax, 16.0, 5.9, 2.6, 1.9, 'planner\nGP-Small\n(3.7K)', '#0f4a37', ec=C_AQUA, fs=9, bold=True)
    arrow(ax, 15.3, 6.85, 16.0, 6.85)
    # 到达后滑
    box(ax, 16.0, 2.4, 2.6, 1.4, 'reach g_i →\nwindow slides', SURFACE, ec=BORDER, fs=8.2, tc=INK2)
    arrow(ax, 17.3, 5.9, 17.3, 3.9, color=INK2)
    arrow(ax, 14.7, 3.1, 13.85, 6.0, color=INK2)   # 回流
    # 重建队列
    box(ax, 12.0, 1.5, 3.4, 1.5, 'new active point\n→ rebuild queue', SURFACE, ec=C_AQUA, fs=8.5)
    arrow(ax, 13.6, 3.0, 13.6, 3.05, color=C_AQUA)  # 占位避免误连
    arrow(ax, 12.0, 2.25, 9.8, 4.3, color=C_AQUA)
    ax.text(0.6, 0.75, 'explore points: beacons themselves at startup; later replaced by fewer waypoints with enough FOV — one planner serves both',
            color=MUTED, fontsize=9, style='italic')
    fig.savefig('runs/viz_visit.png', dpi=150, facecolor=PAGE, bbox_inches='tight')
    plt.close(fig)
    print('  -> runs/viz_visit.png')

if __name__ == '__main__':
    os.makedirs('runs', exist_ok=True)
    stages()
    p2_flow()
    distill()
    visit()
    print('Done.')
