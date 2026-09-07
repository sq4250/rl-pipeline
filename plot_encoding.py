"""视频素材: 极坐标编码可视化 + 门控结构可视化.

图1 viz_encoding.png: 世界视图 (车+3航点+距离/角度标注) → 8D 归一化输入向量
图2 viz_gate.png: 共享 target_enc + v2/v3 门控开关 + 三种门状态的 80D 拼接
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P
import viz_cluster6 as V

SURFACE = '#1a1a19'; PAGE = '#0d0d0d'
INK = '#ffffff'; INK2 = '#c3c2b7'; MUTED = '#898781'
BORDER = '#3a3a37'; ZERO = '#3f3f3c'
C_BLUE = '#3987e5'; C_ORANGE = '#d95926'; C_AQUA = '#199e70'; C_YELLOW = '#c98500'

def box(ax, x, y, w, h, text, fc, ec=BORDER, fs=8, tc=INK, bold=False, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.015,rounding_size=0.03',
                                fc=fc, ec=ec, lw=lw))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', color=tc, fontsize=fs,
            weight='bold' if bold else 'normal', linespacing=1.35)

def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=10,
                                 color=color, lw=lw, shrinkA=2, shrinkB=2))

def wrap(a): return np.arctan2(np.sin(a), np.cos(a))

def draw_signed_arc(ax, center, r, a_prev, dth, color, lw=1.2, ext=0.35):
    """从上一段方向 a_prev 出发, 按 Δθ 符号方向画弧 (正=逆时针, 负=顺时针).

    上一段方向以虚线延长穿过顶点 → 弧线起点落在延长线上, 角度起点一目了然.
    显式采样, 不依赖 Arc 的负角度行为; 起点小圆点 + 终点箭头标方向.
    """
    # 上一段方向虚线延长 (穿过顶点, 延伸到弧外)
    ax.plot([center[0], center[0] + (r + ext)*np.cos(a_prev)],
            [center[1], center[1] + (r + ext)*np.sin(a_prev)],
            color=color, lw=1.0, ls='--', alpha=0.6, zorder=2)
    t = np.linspace(0.0, float(dth), 80)
    xs = center[0] + r*np.cos(a_prev + t)
    ys = center[1] + r*np.sin(a_prev + t)
    ax.plot(xs, ys, color=color, lw=lw)
    ax.plot([center[0] + r*np.cos(a_prev)], [center[1] + r*np.sin(a_prev)],
            marker='o', ms=3, color=color)
    te = float(dth)
    ax.annotate('', xy=(center[0] + r*np.cos(a_prev+te), center[1] + r*np.sin(a_prev+te)),
                xytext=(center[0] + r*np.cos(a_prev + te*0.93), center[1] + r*np.sin(a_prev + te*0.93)),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=1.1, mutation_scale=11))

# ═══════════════ 图1: 编码 ═══════════════
def fig_encoding():
    car = np.array([3.5, 3.5]); th0 = 0.0
    g1 = np.array([5.2, 4.7]); g2 = np.array([3.7, 1.9]); g3 = np.array([2.3, 2.4])
    d1 = np.hypot(*(g1-car)); h1 = np.arctan2(g1[1]-car[1], g1[0]-car[0]); dth1 = wrap(h1-th0)
    d12 = np.hypot(*(g2-g1)); h12 = np.arctan2(g2[1]-g1[1], g2[0]-g1[0]); dth12 = wrap(h12-h1)
    d23 = np.hypot(*(g3-g2)); h23 = np.arctan2(g3[1]-g2[1], g3[0]-g2[0]); dth23 = wrap(h23-h12)
    v, dlt = 2.0, 0.15
    vals = [v/P.V_MAX, dlt/P.DELTA_MAX, d1/P.D_SCALE, dth1/np.pi, d12/P.D_SCALE, dth12/np.pi, d23/P.D_SCALE, dth23/np.pi]

    fig = plt.figure(figsize=(12, 5.4), facecolor=PAGE)
    # ── 左: 世界视图 ──
    ax = fig.add_axes([0.04, 0.08, 0.42, 0.86]); ax.set_facecolor(SURFACE)
    ax.set_xlim(0, P.FLD); ax.set_ylim(0, P.FLD); ax.set_aspect('equal')
    ax.tick_params(colors=MUTED, labelsize=7)
    V.draw_t_car(ax, car[0], car[1], th0, dlt, alpha=1.0)
    for k, g, c in [(0, g1, '#ff4444'), (1, g2, '#ffaa00'), (2, g3, '#44ff44')]:
        ax.plot(g[0], g[1], marker='x', color=c, ms=12, mew=3)
        ax.annotate(f'g{k+1}', (g[0], g[1]), textcoords='offset points', xytext=(8, 8),
                    color=c, fontsize=9, weight='bold')
    # 距离线
    ax.plot([car[0], g1[0]], [car[1], g1[1]], color=C_BLUE, lw=1.3, ls='--', alpha=0.8)
    ax.text((car[0]+g1[0])/2+0.12, (car[1]+g1[1])/2-0.18, f'd1={d1:.2f}', color=C_BLUE, fontsize=8.5)
    ax.plot([g1[0], g2[0]], [g1[1], g2[1]], color=C_ORANGE, lw=1.3, ls='--', alpha=0.8)
    ax.text((g1[0]+g2[0])/2+0.12, (g1[1]+g2[1])/2-0.2, f'd12={d12:.2f}', color=C_ORANGE, fontsize=8.5)
    ax.plot([g2[0], g3[0]], [g2[1], g3[1]], color=C_AQUA, lw=1.3, ls='--', alpha=0.8)
    ax.text((g2[0]+g3[0])/2-0.62, (g2[1]+g3[1])/2-0.18, f'd23={d23:.2f}', color=C_AQUA, fontsize=8.5)
    # 角度弧: 从上一段方向出发按 Δθ 符号扫过
    draw_signed_arc(ax, car, 1.1, th0, dth1, C_BLUE)
    ax.text(car[0]+0.62, car[1]+0.22, f'Δθ1={np.degrees(dth1):.0f}°', color=C_BLUE, fontsize=8)
    draw_signed_arc(ax, g1, 0.9, h1, dth12, C_ORANGE)
    ax.text(g1[0]+0.15, g1[1]+0.5, f'Δθ12={np.degrees(dth12):.0f}°', color=C_ORANGE, fontsize=8)
    draw_signed_arc(ax, g2, 0.9, h12, dth23, C_AQUA)
    ax.text(g2[0]-0.75, g2[1]+0.32, f'Δθ23={np.degrees(dth23):.0f}°', color=C_AQUA, fontsize=8)
    ax.set_title('Polar geometry (world view)', color=INK, fontsize=10, loc='left', pad=8)

    # ── 右: 8D 向量 ──
    ax2 = fig.add_axes([0.52, 0.08, 0.44, 0.86]); ax2.set_facecolor(SURFACE)
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis('off')
    groups = [
        ('o1  car→g1', 4, C_BLUE, ['v / V_MAX', 'δ / δmax', 'd1 / Dscale', 'Δθ1 / π']),
        ('o2  g1→g2', 2, C_ORANGE, ['d12 / Dscale', 'Δθ12 / π']),
        ('o3  g2→g3', 2, C_AQUA, ['d23 / Dscale', 'Δθ23 / π']),
    ]
    y = 8.6; vi = 0
    for name, n, color, labels in groups:
        ax2.text(0.4, y+0.32, name, color=color, fontsize=8.5, weight='bold')
        for j in range(n):
            box(ax2, 0.4, y-0.62, 4.6, 0.5, f'{labels[j]}', SURFACE, ec=color, fs=7.5)
            box(ax2, 5.15, y-0.62, 1.6, 0.5, f'{vals[vi]:.2f}', '#14315a', ec=color, fs=8, bold=True)
            y -= 0.66; vi += 1
        y -= 0.42
    arrow(ax2, 6.9, 5.0, 8.3, 5.0, color=INK2)
    box(ax2, 8.35, 4.55, 1.55, 0.9, '8D\ninput', C_YELLOW, fs=8, bold=True)
    ax2.set_title('Polar 8D encoding (normalized)', color=INK, fontsize=10, loc='left', pad=8)
    fig.savefig('runs/viz_encoding.png', dpi=150, facecolor=PAGE, bbox_inches='tight')
    plt.close(fig)
    print('  -> runs/viz_encoding.png')

# ═══════════════ 图2: 门控 ═══════════════
def fig_gate():
    fig = plt.figure(figsize=(12, 5.2), facecolor=PAGE)
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.9]); ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.4); ax.axis('off')

    # 三个输入槽
    slots = [('o1\n(v δ d1 Δθ1)', 1.9, C_BLUE), ('o2\n(d12 Δθ12)', 4.4, C_ORANGE), ('o3\n(d23 Δθ23)', 6.9, C_AQUA)]
    for label, x, c in slots:
        box(ax, x, 3.9, 1.5, 1.1, label, SURFACE, ec=c, fs=8)
    # 编码器
    box(ax, 1.9, 2.5, 1.5, 0.9, 'state_enc\n2→32', '#14315a', fs=7.5)
    box(ax, 4.4, 2.5, 1.5, 0.9, 'target_enc\n2→16', '#5a2f14', fs=7.5)
    box(ax, 6.9, 2.5, 1.5, 0.9, 'target_enc\n2→16 (shared)', '#5a2f14', fs=7.5)
    # 共享权重括号 (o2/o3)
    ax.plot([4.4, 8.4], [1.92, 1.92], color=MUTED, lw=1.0)
    ax.text(6.4, 1.72, 'shared weights W_tgt', color=MUTED, fontsize=7, ha='center')
    for x0 in (1.9, 4.4, 6.9):
        arrow(ax, x0+0.75, 3.9, x0+0.75, 3.4)
    # 门
    for x0, gv in [(5.15, '×v2'), (7.65, '×v3')]:
        ax.add_patch(Circle((x0, 1.7), 0.34, fc='#14315a', ec=C_YELLOW, lw=1.6))
        ax.text(x0, 1.7, gv, color='#ffcc00', fontsize=8, ha='center', va='center', weight='bold')
    arrow(ax, 2.65, 2.5, 2.65, 1.95)
    arrow(ax, 5.15, 2.5, 5.15, 2.04)
    arrow(ax, 7.65, 2.5, 7.65, 2.04)
    # 拼接条
    ax.text(9.7, 3.55, 'concat', color=INK2, fontsize=8, ha='center')
    segs = [('32', 1.6, C_BLUE, 'state'), ('16', 0.8, C_ORANGE, 't1'), ('16', 0.8, C_ORANGE, 'v2·t2'), ('16', 0.8, C_ORANGE, 'v3·t3')]
    x = 9.7 - (1.6+0.8*3+3*0.06)/2
    for label, w, c, sub in segs:
        box(ax, x, 1.95, w, 0.85, label, SURFACE, ec=c, fs=8, bold=True)
        ax.text(x+w/2, 1.72, sub, color=MUTED, fontsize=6.5, ha='center')
        x += w + 0.06
    for x0 in (2.65, 5.15, 7.65):
        arrow(ax, x0, 1.36, 9.7, 1.9, color=MUTED, lw=0.9)
    # 三种门状态
    ax.text(0.2, 1.02, 'gate states:', color=INK2, fontsize=8)
    states = [
        ('3 targets', 'v2=1, v3=1', 'full 80D', C_AQUA),
        ('2 targets', 'v2=1, v3=0', 'o3 branch → 0', C_YELLOW),
        ('1 target', 'v2=0, v3=0', 'o2,o3 → 0', '#e66767'),
    ]
    x = 0.2
    for name, g, effect, c in states:
        box(ax, x, 0.05, 2.4, 0.8, f'{name}\n{g}', SURFACE, ec=c, fs=7.5)
        box(ax, x, -0.62, 2.4, 0.55, effect, ZERO, ec=ZERO, fs=7, tc=INK2)
        x += 2.6
    ax.set_title('Gated concat — non-existent targets are zeroed, never encoded', color=INK,
                 fontsize=10, loc='left', pad=8)
    fig.savefig('runs/viz_gate.png', dpi=150, facecolor=PAGE, bbox_inches='tight')
    plt.close(fig)
    print('  -> runs/viz_gate.png')

if __name__ == '__main__':
    os.makedirs('runs', exist_ok=True)
    fig_encoding()
    fig_gate()
    print('Done.')
