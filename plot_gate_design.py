"""精确门控/编码器设计图 (Teacher 版 2→32/2→16 + GP-Small 版 2→8/2→8).

精确复刻 forward():
  s  = state_enc (o1[:, :2])            # [v/V_MAX, δ/δmax] → s_out
  t1 = target_enc(o1[:, 2:])            # [d1/5, Δθ1/π]    → t_out  无门 (当前目标必存在)
  t2 = v2 * target_enc(o2)              # [d12/5, Δθ12/π]  → t_out  门 v2
  t3 = v3 * target_enc(o3)              # [d23/5, Δθ23/π]  → t_out  门 v3
  cat = [s | t1 | t2 | t3] = (s_out+3*t_out)D
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

SURFACE = '#1a1a19'; PAGE = '#0d0d0d'
INK = '#ffffff'; INK2 = '#c3c2b7'; MUTED = '#898781'
BORDER = '#3a3a37'
C_BLUE = '#3987e5'; C_ORANGE = '#d95926'; C_AQUA = '#199e70'; C_YELLOW = '#c98500'

def box(ax, x, y, w, h, text, fc, ec=BORDER, fs=8, tc=INK, bold=False, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.015,rounding_size=0.03',
                                fc=fc, ec=ec, lw=lw))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', color=tc, fontsize=fs,
            weight='bold' if bold else 'normal', linespacing=1.3)

def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=10,
                                 color=color, lw=lw, shrinkA=2, shrinkB=2))

def draw_topology(ax, xa, xb, xc, xd, s_out, t_out, tag, tag_color,
                  y_in=4.6, y_enc=3.1, y_gate=1.7, y_cat=0.55):
    """4 分支: A=state(o1前2) B=t1(o1后2) C=t2(o2) D=t3(o3)."""
    w = 1.9; h_in = 0.85; h_enc = 0.8
    # ── 输入行 ──
    # o1 横跨 A+B, 内部 2|2 切分
    box(ax, xa-w/2, y_in, 2*w, h_in, '', SURFACE, ec=C_BLUE, lw=1.4)
    ax.plot([xa, xa], [y_in+0.08, y_in+h_in-0.08], color=C_BLUE, lw=1.0, ls=':')
    ax.text(xa-w/4, y_in+h_in/2, 'v/V_MAX\nδ/δmax', ha='center', va='center', fontsize=6.8,
            color=C_BLUE, linespacing=1.3)
    ax.text(xa+w/4, y_in+h_in/2, 'd1/5\nΔθ1/π', ha='center', va='center', fontsize=6.8,
            color=C_ORANGE, linespacing=1.3)
    ax.text(xa, y_in+h_in+0.22, 'o1', ha='center', fontsize=8, color=INK2, weight='bold')
    box(ax, xc-w/2, y_in, w, h_in, 'd12/5\nΔθ12/π', SURFACE, ec=C_ORANGE, fs=6.8)
    ax.text(xc, y_in+h_in+0.22, 'o2', ha='center', fontsize=8, color=INK2, weight='bold')
    box(ax, xd-w/2, y_in, w, h_in, 'd23/5\nΔθ23/π', SURFACE, ec=C_AQUA, fs=6.8)
    ax.text(xd, y_in+h_in+0.22, 'o3', ha='center', fontsize=8, color=INK2, weight='bold')
    # ── 编码器行 ──
    box(ax, xa-w/2, y_enc, w, h_enc, f'state_enc\n2→{s_out}', '#14315a', ec=C_BLUE, fs=7.5, bold=True)
    for x, ec in ((xb, C_ORANGE), (xc, C_ORANGE), (xd, C_AQUA)):
        box(ax, x-w/2, y_enc, w, h_enc, f'target_enc\n2→{t_out}', '#5a2f14', ec=ec, fs=7.5, bold=True)
    # 共享权重括号 (B/C/D 三个 target_enc)
    ax.plot([xb-w/2, xd+w/2], [y_enc-0.2, y_enc-0.2], color=MUTED, lw=1.0)
    ax.text((xb+xd)/2, y_enc-0.42, 'shared weights  W_tgt', ha='center', fontsize=7, color=MUTED)
    # 输入→编码器连线 + 维度标注
    arrow(ax, xa-w/4, y_in, xa-w/4, y_enc+h_enc, color=C_BLUE)
    dim_label = lambda x, y, t: ax.text(x, y, t, color=C_YELLOW, fontsize=7, ha='center', weight='bold')
    arrow(ax, xa+w/4, y_in, xb-w/4, y_enc+h_enc, color=C_ORANGE)
    arrow(ax, xc, y_in, xc, y_enc+h_enc, color=C_ORANGE)
    arrow(ax, xd, y_in, xd, y_enc+h_enc, color=C_AQUA)
    # ── 门行 ──
    arrow(ax, xa, y_enc, xa, y_gate+0.3, color=C_BLUE)
    dim_label(xa, (y_enc+y_gate+0.3)/2, f'{s_out}')
    arrow(ax, xb, y_enc, xb, y_gate+0.3, color=C_ORANGE)
    dim_label(xb, (y_enc+y_gate+0.3)/2, f'{t_out}')
    ax.text(xb, y_gate+0.05, 'no gate\n(target exists)', fontsize=6.3, color=INK2, ha='center')
    for x, gv, c in ((xc, '×v2', C_ORANGE), (xd, '×v3', C_AQUA)):
        arrow(ax, x, y_enc, x, y_gate+0.3, color=c)
        dim_label(x, (y_enc+y_gate+0.3)/2, f'{t_out}')
        ax.add_patch(Circle((x, y_gate), 0.28, fc='#14315a', ec=C_YELLOW, lw=1.5))
        ax.text(x, y_gate, gv, color='#ffcc00', fontsize=7.5, ha='center', va='center', weight='bold')
    # ── 拼接条 ──
    for x in (xa, xb, xc, xd):
        arrow(ax, x, y_gate-0.28, x, y_cat+0.4, color=MUTED, lw=0.9)
    bw = 0.7
    cat_x0 = xa - w/2
    for i, (dim, c) in enumerate(((s_out, C_BLUE), (t_out, C_ORANGE), (t_out, C_ORANGE), (t_out, C_ORANGE))):
        box(ax, cat_x0+i*(bw+0.08), y_cat, bw, 0.42, str(dim), SURFACE, ec=c, fs=8, bold=True)
    cat_dim = s_out + 3*t_out
    ax.text(cat_x0+1.5*(bw+0.08), y_cat+0.6, f'concat = {cat_dim}D', fontsize=8.5, color=INK, weight='bold', ha='center')
    ax.text((xa+xd)/2, y_in+h_in+0.85, tag, fontsize=10, color=tag_color, weight='bold', ha='center')

if __name__ == '__main__':
    os.makedirs('runs', exist_ok=True)
    for dims, tag, color, fname in (
        ((32, 16), 'Teacher — GatedConcatActor', C_BLUE, 'runs/viz_gate_teacher.png'),
        ((8, 8), 'GP-Small — 3.7K', C_AQUA, 'runs/viz_gate_gp.png'),
    ):
        fig = plt.figure(figsize=(11, 6.4), facecolor=PAGE)
        ax = fig.add_axes([0.03, 0.05, 0.94, 0.9]); ax.set_facecolor(SURFACE)
        ax.set_xlim(0, 12); ax.set_ylim(-0.7, 6.4); ax.axis('off')
        draw_topology(ax, 1.75, 4.15, 6.55, 8.95, dims[0], dims[1], tag, color)
        fig.savefig(fname, dpi=150, facecolor=PAGE, bbox_inches='tight')
        plt.close(fig)
        print(f'  -> {fname}')
