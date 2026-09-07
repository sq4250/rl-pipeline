"""视频素材: 输入编码公式+图解合一图 (16:9 高清).

左: 世界视图 (全部几何量标注: d1/d12/d23, 航向 θ, 视线角 h, 相对角 Δθ 弧线)
右上: 公式区 (几何定义 → wrap → o1/o2/o3 编码式 + 场景数值示例)
右下: 编码器/门控/拼接流 (state_enc/target_enc 共享权重, v2/v3 门, 80D 拼接)

输出: runs/viz_encoding_formula.png (2880×1620 @150dpi)
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

def draw_signed_arc(ax, center, r, a_prev, dth, color, lw=1.6, ext=0.4):
    """从上一段方向 a_prev 出发, 按 Δθ 符号方向画弧 (正=逆时针, 负=顺时针).

    上一段方向以虚线延长穿过顶点 → 弧线起点落在延长线上, 角度起点一目了然.
    显式采样, 不依赖 Arc 的负角度行为; 起点小圆点 + 终点箭头标方向.
    """
    # 上一段方向虚线延长 (穿过顶点, 延伸到弧外)
    ax.plot([center[0], center[0] + (r + ext)*np.cos(a_prev)],
            [center[1], center[1] + (r + ext)*np.sin(a_prev)],
            color=color, lw=1.1, ls='--', alpha=0.65, zorder=2)
    t = np.linspace(0.0, float(dth), 80)
    xs = center[0] + r*np.cos(a_prev + t)
    ys = center[1] + r*np.sin(a_prev + t)
    ax.plot(xs, ys, color=color, lw=lw)
    ax.plot([center[0] + r*np.cos(a_prev)], [center[1] + r*np.sin(a_prev)],
            marker='o', ms=3.5, color=color)
    te = float(dth)
    ax.annotate('', xy=(center[0] + r*np.cos(a_prev+te), center[1] + r*np.sin(a_prev+te)),
                xytext=(center[0] + r*np.cos(a_prev + te*0.93), center[1] + r*np.sin(a_prev + te*0.93)),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=1.3, mutation_scale=12))

def main():
    # ── 场景几何 (与 plot_encoding 一致) ──
    car = np.array([3.5, 3.5]); th0 = 0.0
    g1 = np.array([5.2, 4.7]); g2 = np.array([3.7, 1.9]); g3 = np.array([2.3, 2.4])
    d1 = np.hypot(*(g1-car)); h1 = np.arctan2(g1[1]-car[1], g1[0]-car[0]); dth1 = wrap(h1-th0)
    d12 = np.hypot(*(g2-g1)); h12 = np.arctan2(g2[1]-g1[1], g2[0]-g1[0]); dth12 = wrap(h12-h1)
    d23 = np.hypot(*(g3-g2)); h23 = np.arctan2(g3[1]-g2[1], g3[0]-g2[0]); dth23 = wrap(h23-h12)
    v, dlt = 2.0, 0.15
    vals = [v/P.V_MAX, dlt/P.DELTA_MAX, d1/P.D_SCALE, dth1/np.pi, d12/P.D_SCALE, dth12/np.pi, d23/P.D_SCALE, dth23/np.pi]

    fig = plt.figure(figsize=(19.2, 10.8), facecolor=PAGE)

    # ═════════ 左: 世界视图 ═════════
    ax = fig.add_axes([0.03, 0.06, 0.40, 0.88]); ax.set_facecolor(SURFACE)
    ax.set_xlim(0, P.FLD); ax.set_ylim(0, P.FLD); ax.set_aspect('equal')
    ax.tick_params(colors=MUTED, labelsize=8)
    V.draw_t_car(ax, car[0], car[1], th0, dlt, alpha=1.0)
    ax.text(car[0]-0.85, car[1]-0.42, r'$\theta=0$', color='#ffcc00', fontsize=9)
    for k, g, c in [(0, g1, '#ff4444'), (1, g2, '#ffaa00'), (2, g3, '#44ff44')]:
        ax.plot(g[0], g[1], marker='x', color=c, ms=14, mew=3.5)
        ax.annotate(f'$g_{{{k+1}}}$', (g[0], g[1]), textcoords='offset points', xytext=(9, 9),
                    color=c, fontsize=11, weight='bold')
    # 距离
    ax.plot([car[0], g1[0]], [car[1], g1[1]], color=C_BLUE, lw=1.4, ls='--', alpha=0.85)
    ax.text((car[0]+g1[0])/2+0.14, (car[1]+g1[1])/2-0.22, f'$d_1={d1:.2f}$', color=C_BLUE, fontsize=10)
    ax.plot([g1[0], g2[0]], [g1[1], g2[1]], color=C_ORANGE, lw=1.4, ls='--', alpha=0.85)
    ax.text((g1[0]+g2[0])/2+0.14, (g1[1]+g2[1])/2-0.24, f'$d_{{12}}={d12:.2f}$', color=C_ORANGE, fontsize=10)
    ax.plot([g2[0], g3[0]], [g2[1], g3[1]], color=C_AQUA, lw=1.4, ls='--', alpha=0.85)
    ax.text((g2[0]+g3[0])/2-0.78, (g2[1]+g3[1])/2-0.22, f'$d_{{23}}={d23:.2f}$', color=C_AQUA, fontsize=10)
    # 视线角
    ax.text(car[0]+0.75, car[1]-0.62, f'$h_{{c\\to g1}}={np.degrees(h1):.0f}°$', color=C_BLUE, fontsize=9)
    ax.text(g1[0]+0.1, g1[1]-0.72, f'$h_{{g1\\to g2}}={np.degrees(h12):.0f}°$', color=C_ORANGE, fontsize=9)
    ax.text(g2[0]-1.5, g2[1]-0.5, f'$h_{{g2\\to g3}}={np.degrees(h23):.0f}°$', color=C_AQUA, fontsize=9)
    # 相对角弧: 从上一段方向 (车头 θ / car→g1 / g1→g2) 出发按 Δθ 符号扫过
    draw_signed_arc(ax, car, 1.15, th0, dth1, C_BLUE)
    ax.text(car[0]+0.66, car[1]+0.3, f'$\\Delta\\theta_1={np.degrees(dth1):.0f}°$', color=C_BLUE, fontsize=10)
    draw_signed_arc(ax, g1, 1.0, h1, dth12, C_ORANGE)
    ax.text(g1[0]+0.18, g1[1]+0.6, f'$\\Delta\\theta_{{12}}={np.degrees(dth12):.0f}°$', color=C_ORANGE, fontsize=10)
    draw_signed_arc(ax, g2, 1.0, h12, dth23, C_AQUA)
    ax.text(g2[0]-0.85, g2[1]+0.42, f'$\\Delta\\theta_{{23}}={np.degrees(dth23):.0f}°$', color=C_AQUA, fontsize=10)
    ax.set_title('Geometry — world view', color=INK, fontsize=13, loc='left', pad=10)

    # ═════════ 右上: 公式区 ═════════
    ax2 = fig.add_axes([0.46, 0.50, 0.52, 0.44]); ax2.set_facecolor(SURFACE)
    ax2.set_xlim(0, 12); ax2.set_ylim(0, 10.5); ax2.axis('off')
    ax2.set_title('Incremental polar encoding — formulas', color=INK, fontsize=13, loc='left', pad=10)
    ax2.text(0.15, 9.2, 'definitions', color=MUTED, fontsize=10, style='italic')
    ax2.text(0.15, 8.35,
             r'$d_1=|g_1-p|,\ \ d_{12}=|g_2-g_1|,\ \ d_{23}=|g_3-g_2|$',
             color=INK2, fontsize=12.5)
    ax2.text(0.15, 7.45,
             r'$h_{c\to g1}=\mathrm{atan2}(g_{1y}-p_y,\ g_{1x}-p_x),\quad h_{g1\to g2}=\mathrm{atan2}(g_{2y}-g_{1y},\ g_{2x}-g_{1x})$',
             color=INK2, fontsize=12)
    # 相对角: wrap 记法 (语义自明, 不展开实现)
    ax2.text(0.15, 6.7, r'$\Delta\theta_1=\mathrm{wrap}(h_{c\to g1}-\theta)$', color=C_BLUE, fontsize=11.5)
    ax2.text(0.15, 6.25, r'$\Delta\theta_{12}=\mathrm{wrap}(h_{g1\to g2}-h_{c\to g1})$', color=C_ORANGE, fontsize=11.5)
    ax2.text(0.15, 5.8, r'$\Delta\theta_{23}=\mathrm{wrap}(h_{g2\to g3}-h_{g1\to g2})$', color=C_AQUA, fontsize=11.5)
    ax2.text(0.15, 5.25, 'normalized input vectors', color=MUTED, fontsize=10, style='italic')
    ax2.text(0.15, 4.28, r'$\mathbf{o}_1=\left[\ \frac{v}{V_{max}},\ \frac{\delta}{\delta_{max}},\ \frac{d_1}{D_{scale}},\ \frac{\Delta\theta_1}{\pi}\ \right]$',
             color=C_BLUE, fontsize=12.5)
    ax2.text(0.15, 3.05, r'$\mathbf{o}_2=\left[\ \frac{d_{12}}{D_{scale}},\ \frac{\Delta\theta_{12}}{\pi}\ \right]$',
             color=C_ORANGE, fontsize=12.5)
    ax2.text(0.15, 1.82, r'$\mathbf{o}_3=\left[\ \frac{d_{23}}{D_{scale}},\ \frac{\Delta\theta_{23}}{\pi}\ \right]$',
             color=C_AQUA, fontsize=12.5)
    # 数值示例
    box(ax2, 7.4, 0.55, 4.4, 5.3, '', ZERO, ec=BORDER)
    ax2.text(7.65, 5.5, 'this scene →', color=MUTED, fontsize=9.5)
    nums = [
        (r'$v/V_{max}$', f'{vals[0]:.2f}', C_BLUE), (r'$\delta/\delta_{max}$', f'{vals[1]:.2f}', C_BLUE),
        (r'$d_1/D_{scale}$', f'{vals[2]:.2f}', C_BLUE), (r'$\Delta\theta_1/\pi$', f'{vals[3]:+.2f}', C_BLUE),
        (r'$d_{12}/D_{scale}$', f'{vals[4]:.2f}', C_ORANGE), (r'$\Delta\theta_{12}/\pi$', f'{vals[5]:+.2f}', C_ORANGE),
        (r'$d_{23}/D_{scale}$', f'{vals[6]:.2f}', C_AQUA), (r'$\Delta\theta_{23}/\pi$', f'{vals[7]:+.2f}', C_AQUA),
    ]
    for i, (lab, val, c) in enumerate(nums):
        ax2.text(7.6, 4.7 - i*0.55, lab, color=c, fontsize=10.5)
        ax2.text(10.7, 4.7 - i*0.55, val, color=INK, fontsize=11, weight='bold', ha='right')

    # ═════════ 右下: 编码/门控流 ═════════
    ax3 = fig.add_axes([0.46, 0.06, 0.52, 0.40]); ax3.set_facecolor(SURFACE)
    ax3.set_xlim(0, 12); ax3.set_ylim(0, 9.4); ax3.axis('off')
    ax3.set_title('Encoder + gates → 80D input', color=INK, fontsize=13, loc='left', pad=10)
    # 输入槽: o1 拆两路 (o1[:2] → state_enc, o1[2:] → target_enc)
    box(ax3, 0.7, 6.6, 2.2, 1.25, 'o1\n[v δ d1 Δθ1]', SURFACE, ec=C_BLUE, fs=9)
    box(ax3, 4.6, 6.6, 1.7, 1.25, 'o2\n[d12 Δθ12]', SURFACE, ec=C_ORANGE, fs=9)
    box(ax3, 7.2, 6.6, 1.7, 1.25, 'o3\n[d23 Δθ23]', SURFACE, ec=C_AQUA, fs=9)
    # 编码器: state_enc + 三个 target_enc (t1 无门)
    box(ax3, 0.75, 4.4, 1.5, 1.0, 'state_enc\n2→32', '#14315a', fs=8.5)
    box(ax3, 2.35, 4.4, 1.5, 1.0, 'target_enc\n2→16', '#5a2f14', fs=8.5)
    box(ax3, 4.6, 4.4, 1.7, 1.0, 'target_enc\n2→16 (shared)', '#5a2f14', fs=8.5)
    box(ax3, 7.2, 4.4, 1.7, 1.0, 'target_enc\n2→16 (shared)', '#5a2f14', fs=8.5)
    ax3.plot([4.6, 8.9], [4.38, 4.38], color=MUTED, lw=1.0)
    ax3.text(6.75, 4.15, 'shared weights', color=MUTED, fontsize=8, ha='center')
    arrow(ax3, 1.5, 6.6, 1.5, 5.45)      # o1[:2] → state_enc
    arrow(ax3, 2.7, 6.6, 3.1, 5.45)      # o1[2:] → target_enc (t1)
    arrow(ax3, 5.45, 6.6, 5.45, 5.45)
    arrow(ax3, 8.05, 6.6, 8.05, 5.45)
    ax3.text(1.5, 6.1, 'o1[:2]', color=C_BLUE, fontsize=7, ha='center')
    ax3.text(2.62, 6.1, 'o1[2:]', color=C_BLUE, fontsize=7, ha='center')
    # t1 分支: 无门, 直下总线
    arrow(ax3, 3.1, 4.4, 3.1, 2.6)
    ax3.text(3.1, 3.72, 'no gate', color=INK2, fontsize=8, ha='center')
    ax3.text(2.72, 3.4, '16', color=C_ORANGE, fontsize=7.5)
    # 门 (v2/v3): 位于编码器与总线之间, 流单调向下
    for x0, gv in [(5.45, '×v2'), (8.05, '×v3')]:
        ax3.add_patch(Circle((x0, 3.35), 0.4, fc='#14315a', ec=C_YELLOW, lw=1.8))
        ax3.text(x0, 3.35, gv, color='#ffcc00', fontsize=9, ha='center', va='center', weight='bold')
        arrow(ax3, x0, 4.4, x0, 3.75)
        arrow(ax3, x0, 2.95, x0, 2.6)
        ax3.text(x0+0.5, 3.35, '16', color=C_ORANGE, fontsize=7.5, va='center')
    # state 分支下总线 + 总线收束到 trunk
    arrow(ax3, 1.5, 4.4, 1.5, 2.6)
    ax3.text(1.12, 3.4, '32', color=C_BLUE, fontsize=7.5)
    ax3.plot([1.5, 9.9], [2.6, 2.6], color=MUTED, lw=1.1)
    ax3.plot(9.9, 2.6, marker='o', ms=5, color=INK2)
    box(ax3, 10.35, 2.0, 1.55, 1.2, 'trunk\n128-256-128', '#14315a', fs=8.5, bold=True)
    arrow(ax3, 9.9, 2.6, 10.33, 2.6, color=MUTED, lw=1.1)
    ax3.text(11.12, 1.78, r'$x\in\mathbb{R}^{80}$', color=MUTED, fontsize=9)
    # 门公式
    ax3.text(0.15, 0.62, r'$v_2=\mathbb{1}\{\mathrm{remaining}\geq 2\},\quad v_3=\mathbb{1}\{\mathrm{remaining}\geq 3\}$',
             color=INK2, fontsize=11)
    ax3.text(0.15, 0.02, 'nonexistent targets are multiplied by 0 before encoding — input dim stays 80D',
             color=MUTED, fontsize=9, style='italic')

    out = 'runs/viz_encoding_formula.png'
    fig.savefig(out, dpi=150, facecolor=PAGE, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out}')

if __name__ == '__main__':
    os.makedirs('runs', exist_ok=True)
    main()
