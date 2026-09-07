"""视频素材: CC / CCC 弧示意图 (Dubins 风格, 真实相切几何, 目标在不可达区内).

左 (前): CCC — 终端航向固定, 三弧: 反打弧 R60 + 主弧 L180 + 对航向弧 R60
右 (后): CC  — 终端航向自由, 两弧: 反打弧 R29 + 大回绕 L313
          目标构造在车身左侧 0.5m — 位于左转圆心圆内部 (不可达区), 由几何解出;
          不可达双圆淡色画出, 两弧切点 ★ = 人为航点 WP.

圆心构造: 每个新圆心 = 前圆心 + 2R·切点方向 → 严格相切 (代码内断言验证).

输出: runs/viz_ccc_cc.png (16:9 高清)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SURFACE = '#1a1a19'; PAGE = '#0d0d0d'
INK = '#ffffff'; INK2 = '#c3c2b7'; MUTED = '#898781'
C_BLUE = '#3987e5'; C_ORANGE = '#d95926'; C_AQUA = '#199e70'; C_YELLOW = '#c98500'
ARC_COLORS = [C_AQUA, C_ORANGE, C_BLUE]

R = 1.0

def build_path(p0, h0_deg, arcs):
    """按 (转向, 角度) 序列生成相切圆弧路径. 返回 (segs, centers, pos_end, h_end)."""
    pos = np.array(p0, dtype=float); h = h0_deg
    segs = []; centers = []
    for turn, ang in arcs:
        s = 1.0 if turn == 'L' else -1.0
        perp = np.radians(h + s*90)
        c = pos + R*np.array([np.cos(perp), np.sin(perp)])
        a0 = np.radians(h - s*90)
        t = np.linspace(0.0, np.radians(ang), 120)
        phi = a0 + s*t
        arc = c + R*np.stack([np.cos(phi), np.sin(phi)], axis=1)
        segs.append(arc); centers.append(c)
        pos = arc[-1]; h = h + s*ang
    return segs, centers, pos, h

def draw_car(ax, p0, h0_deg):
    x, y = p0; th = np.radians(h0_deg)
    ax.plot([x-0.34*np.cos(th), x+0.34*np.cos(th)], [y-0.34*np.sin(th), y+0.34*np.sin(th)],
            color='#ffcc00', lw=3.2)
    ax.plot(x, y, marker='o', ms=6, color='#ffcc00')
    ax.text(x-0.55, y-0.3, 'car', color='#ffcc00', fontsize=10)

def draw_goal(ax, goal, h_end, heading_free):
    ax.plot(goal[0], goal[1], marker='x', color='#44ff44', ms=13, mew=3.5)
    if heading_free:
        ax.add_patch(Circle(goal, 0.3, fc='none', ec='#44ff44', ls='--', lw=1.1))
        ax.annotate('goal\n(heading free)', (goal[0], goal[1]), textcoords='offset points',
                    xytext=(10, -6), color='#44ff44', fontsize=9.5, weight='bold')
    else:
        eh = np.radians(h_end)
        ax.annotate('', xy=(goal[0]+0.52*np.cos(eh), goal[1]+0.52*np.sin(eh)),
                    xytext=(goal[0], goal[1]),
                    arrowprops=dict(arrowstyle='-|>', color='#44ff44', lw=2.2, mutation_scale=16))
        ax.annotate('goal\n(fixed heading)', (goal[0], goal[1]), textcoords='offset points',
                    xytext=(10, -6), color='#44ff44', fontsize=9.5, weight='bold')

def draw_panel(ax, title, arcs, notes, p0, h0, heading_free, show_unreachable=False):
    segs, centers, goal, h_end = build_path(p0, h0, arcs)
    for i in range(1, len(centers)):
        d = np.linalg.norm(centers[i] - centers[i-1])
        assert abs(d - 2*R) < 1e-9, f'circles not tangent: {d}'
    # 不可达双圆 (车身系左右两圆, 淡色)
    if show_unreachable:
        th = np.radians(h0)
        cL = p0 + R*np.array([-np.sin(th), np.cos(th)])
        cR = p0 + R*np.array([np.sin(th), -np.cos(th)])
        for c, lab in ((cL, 'unreachable\ncircle'), (cR, None)):
            ax.add_patch(Circle(c, R, fc='#14315a', ec=C_BLUE, ls='--', lw=1.2, alpha=0.45))
        ax.text(cL[0]-0.25, cL[1]-0.1, lab, color=C_BLUE, fontsize=8.5, ha='center')
    # 路径圆 (淡)
    for c in centers:
        ax.add_patch(Circle(c, R, fc='none', ec=MUTED, ls='--', lw=0.8, alpha=0.55))
    # 弧
    for i, seg in enumerate(segs):
        ax.plot(seg[:, 0], seg[:, 1], color=ARC_COLORS[i % len(ARC_COLORS)], lw=4.5, alpha=0.95)
    # 切点
    for i, seg in enumerate(segs[:-1]):
        t = seg[-1]
        ax.text(t[0], t[1], '★', color='#ffcc00', fontsize=15, ha='center', va='center')
        ax.text(t[0]-0.16, t[1]-0.26, f'T{i+1}', color='#ffcc00', fontsize=9)
    draw_car(ax, p0, h0)
    draw_goal(ax, goal, h_end, heading_free)
    for i, (txt, col) in enumerate(notes):
        ax.text(0.15, 6.1 - i*0.46, txt, color=col, fontsize=10.5, weight='bold')
    ax.set_xlim(0, 6.6); ax.set_ylim(0, 6.6); ax.set_aspect('equal')
    ax.set_facecolor(SURFACE); ax.axis('off')
    ax.set_title(title, color=INK, fontsize=13, loc='left', pad=10)

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19.2, 10.8), facecolor=PAGE)
    # CCC 在前: 侧后目标 + 固定终端航向, 三段弧都可见
    draw_panel(ax1, 'CCC — terminal heading fixed (3 arcs)',
               [('R', 60), ('L', 180), ('R', 60)],
               [('C1: countersteer away', C_AQUA),
                ('C2: main arc', C_ORANGE),
                ('C3: align the fixed heading', C_BLUE),
                ('three arcs = countersteer + CC', MUTED)],
               p0=np.array([2.4, 1.0]), h0=90.0, heading_free=False)
    # CC 在后: 目标在不可达圆内 (车身左 0.5m), 自由航向, 反打+大回绕
    draw_panel(ax2, 'CC — terminal heading free (2 arcs)',
               [('R', 29), ('L', 313)],
               [('C1: countersteer away (29°)', C_AQUA),
                ('C2: sweep back to goal (313°)', C_ORANGE),
                ('tangent point ★ = WP', '#ffcc00'),
                ('goal inside min-turn circle →', MUTED),
                ('must turn away before turning back', MUTED)],
               p0=np.array([2.6, 1.7]), h0=0.0, heading_free=True, show_unreachable=True)
    out = 'runs/viz_ccc_cc.png'
    fig.savefig(out, dpi=150, facecolor=PAGE, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out}')

if __name__ == '__main__':
    main()
