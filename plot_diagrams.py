"""视频素材: 管线时间轴 + 网络架构图.

视觉规范: 深色表面 #1a1a19; 分类色 (dark 已验证): 蓝#3987e5 橙#d95926 青#199e70 黄#c98500;
直接标注 (身份不依赖颜色); 细线框.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SURFACE = '#1a1a19'; PAGE = '#0d0d0d'
INK = '#ffffff'; INK2 = '#c3c2b7'; MUTED = '#898781'
BORDER = '#3a3a37'
C_BLUE = '#3987e5'; C_ORANGE = '#d95926'; C_AQUA = '#199e70'; C_YELLOW = '#c98500'

def box(ax, x, y, w, h, text, fc, ec=BORDER, fs=8.5, tc=INK, bold=False, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.04',
                                fc=fc, ec=ec, lw=lw))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', color=tc, fontsize=fs,
            weight='bold' if bold else 'normal', linespacing=1.4)

def arrow(ax, x0, y0, x1, y1, color=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=11,
                                 color=color, lw=1.1, shrinkA=2, shrinkB=2))

# ═══════════════ 管线时间轴 ═══════════════
def timeline():
    fig, ax = plt.subplots(figsize=(12, 3.2), facecolor=PAGE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.6); ax.axis('off')
    phases = [
        ('Phase A', '基础能力\nTOL 课程 → polar 探索 → anneal', 1.30, C_BLUE, '1300 iters'),
        ('Phase B', 'SelWP 融合\n100% 注入 + blank 网络', 0.30, C_ORANGE, '300'),
        ('Phase C', '交替续训\n噪声 → random↔polar → anneal', 1.00, C_AQUA, '1000'),
        ('Phase D', '蒸馏\nBC + DAgger×12 速度选优', 0.75, C_YELLOW, 'GP-Small 3.7K'),
    ]
    x = 0.4
    for name, desc, w, color, count in phases:
        bw = w * 6.6
        box(ax, x, 0.7, bw, 1.5, f'{name}\n{desc}', color, fs=8)
        ax.text(x+bw/2, 0.52, count, ha='center', va='center', color=INK2, fontsize=7.5)
        if x > 0.5:
            arrow(ax, x-0.18, 1.45, x, 1.45)
        x += bw + 0.42
    ax.text(0.4, 2.42, 'Pipeline A -> B -> C -> D (cold start -> deployment)', color=INK,
            fontsize=11, weight='bold', ha='left')
    fig.savefig('runs/viz_timeline.png', dpi=150, facecolor=PAGE, bbox_inches='tight')
    plt.close(fig)
    print('  -> runs/viz_timeline.png')

# ═══════════════ 网络架构图 ═══════════════
def arch():
    fig, ax = plt.subplots(figsize=(12, 5.6), facecolor=PAGE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.6); ax.axis('off')

    # 输入块 (共享)
    box(ax, 0.3, 2.35, 1.6, 1.0, 'polar 8D\nv δ d1 Δθ1\nd12 Δθ12 d23 Δθ23', SURFACE, ec='#4a4a45', fs=7.5)

    # ── Teacher 行 (y≈3.6) ──
    box(ax, 2.4, 3.05, 1.9, 0.8, 'encoders\nstate 2→32\ntarget 2→16 ×3 (gates)', '#14315a', fs=7.5)
    box(ax, 4.7, 3.05, 1.5, 0.8, 'concat\n80D', SURFACE, fs=8)
    box(ax, 6.5, 3.05, 1.9, 0.8, 'trunk\n128→256→128\nGELU', '#5a2f14', fs=7.5)
    box(ax, 8.7, 3.05, 1.3, 0.8, 'head\n→2', '#0f4a37', fs=8)
    box(ax, 10.3, 3.05, 1.5, 0.8, 'tanh+bias\n[-3,5]×[-14,14]', '#4a3d00', fs=7.5)
    arrow(ax, 1.9, 2.85, 2.4, 3.45)
    arrow(ax, 4.3, 3.45, 4.7, 3.45)
    arrow(ax, 6.2, 3.45, 6.5, 3.45)
    arrow(ax, 8.4, 3.45, 8.7, 3.45)
    arrow(ax, 10.0, 3.45, 10.3, 3.45)
    ax.text(6.0, 4.15, 'Teacher  GatedConcatActor (76.7K) — GELU, stochastic (log_std)',
            color=C_BLUE, fontsize=9.5, weight='bold', ha='center')

    # ── GP-Small 行 (y≈1.4) ──
    box(ax, 2.4, 1.05, 1.9, 0.8, 'encoders\nstate 2→8\ntarget 2→8 ×3 (gates)', '#14315a', fs=7.5)
    box(ax, 4.7, 1.05, 1.5, 0.8, 'concat\n32D', SURFACE, fs=8)
    box(ax, 6.5, 1.05, 1.9, 0.8, 'fc\n48→32→16\nReLU', '#5a2f14', fs=7.5)
    box(ax, 8.7, 1.05, 1.3, 0.8, 'fc4\n→2', '#0f4a37', fs=8)
    box(ax, 10.3, 1.05, 1.5, 0.8, 'linear\n+clamp', '#4a3d00', fs=8)
    arrow(ax, 1.9, 2.35, 2.4, 1.45)
    arrow(ax, 4.3, 1.45, 4.7, 1.45)
    arrow(ax, 6.2, 1.45, 6.5, 1.45)
    arrow(ax, 8.4, 1.45, 8.7, 1.45)
    arrow(ax, 10.0, 1.45, 10.3, 1.45)
    ax.text(6.0, 0.35, 'GP-Small (3.7K) — ReLU, deterministic, 部署产物',
            color=C_AQUA, fontsize=9.5, weight='bold', ha='center')

    # 输入分支线
    arrow(ax, 1.9, 2.85, 2.4, 2.85)
    arrow(ax, 1.9, 1.85, 2.4, 1.85)
    ax.text(0.3, 5.32, 'Network architecture — same polar input, different heads',
            color=INK, fontsize=11, weight='bold', ha='left')
    fig.savefig('runs/viz_arch.png', dpi=150, facecolor=PAGE, bbox_inches='tight')
    plt.close(fig)
    print('  -> runs/viz_arch.png')

if __name__ == '__main__':
    os.makedirs('runs', exist_ok=True)
    timeline()
    arch()
    print('Done.')
