"""训练曲线图 (视频素材): rs / loss 双面板小倍数 + 相位边界标注.

数据源: 两次运行的 stdout 日志
  - v1: A(1300) + B(300) + C(550)   — TOL 课程 + SelWP + 单轮续训
  - v2: C(1000) 交替续训             — 从 p2_bc 起点
同时导出 CSV (供 MATLAB 手工精修).

视觉规范 (dataviz):
  - 深色表面 #1a1a19, 主墨白 #ffffff, 次墨 #c3c2b7
  - rs/a_loss = 蓝 #3987e5 (单序列无图例, 标题点名)
  - c_loss = 橙 #d95926 (双序列图例)
  - 相位边界: 虚线 + 顶部标签; 相位带: 极淡交替底色
  - 无双轴: rs 与 loss 分面板 (小倍数)
"""
import os, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
LOG_V1 = r'C:\Users\tsian\AppData\Local\Temp\claude\c--Devel-test\a984dbac-65dd-4083-be5c-53c7e38bfb94\tasks\b8qenst28.output'
LOG_V2 = r'C:\Users\tsian\AppData\Local\Temp\claude\c--Devel-test\a984dbac-65dd-4083-be5c-53c7e38bfb94\tasks\bvumv9vi6.output'

SURFACE = '#1a1a19'; PAGE = '#0d0d0d'
INK = '#ffffff'; INK2 = '#c3c2b7'; MUTED = '#898781'
GRID = '#2c2c2a'; BAND = '#222221'
BLUE = '#3987e5'; ORANGE = '#d95926'

RE_PHASE = re.compile(r'it\s+(\d+): a=([-\d.eE+]+) c=([-\d.eE+]+) r=([-\d.eE+]+)  rs=([-\d.eE+]+)')
RE_TOL   = re.compile(r'it\s+(\d+) tol=([\d.]+) a=([-\d.eE+]+) c=([-\d.eE+]+) r=([-\d.eE+]+) rs=([-\d.eE+]+)')
RE_P2    = re.compile(r'P2\s+(\d+)/(\d+): suc=([\d.]+)')

def parse_log(path):
    """返回 [(phase_name, [(it, rs, a, c), ...]), ...] 按出现顺序."""
    lines = open(path, encoding='utf-8', errors='replace').read().splitlines()
    phases = []
    cur = None
    for ln in lines:
        new_phase = None
        if ln.startswith('=== P1a'): new_phase = 'P1a TOL'
        elif ln.startswith('P1b:'): new_phase = 'P1b'
        elif ln.startswith('P1c:'): new_phase = 'P1c'
        elif ln.startswith('P1d:'): new_phase = 'P1d'
        elif ln.startswith('=== P2'): new_phase = 'P2 SelWP'
        elif ln.startswith('C1:'): new_phase = 'C1'
        elif ln.startswith('C2:'): new_phase = 'C2'
        elif ln.startswith('C3:'): new_phase = 'C3'
        elif ln.startswith('C4:'): new_phase = 'C4'
        elif ln.startswith('C5:'): new_phase = 'C5'
        if new_phase:
            if cur is not None and cur[1]: phases.append(cur)
            cur = (new_phase, [])
        if cur is None: continue
        ln = ln.strip()
        m = RE_PHASE.match(ln)
        if m:
            it, a, c, r, rs = int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
            if not cur[1] or it > cur[1][-1][0]:
                cur[1].append((it, rs, a, c))
            continue
        m = RE_TOL.match(ln)
        if m:
            it, tol, a, c, r, rs = int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6))
            if not cur[1] or it > cur[1][-1][0]:
                cur[1].append((it, rs, a, c, tol))
            continue
        m = RE_P2.match(ln)
        if m:
            it, tot, suc = int(m.group(1)), int(m.group(2)), float(m.group(3))
            if not cur[1] or it > cur[1][-1][0]:
                cur[1].append((it, suc, np.nan, np.nan))
    if cur is not None and cur[1]: phases.append(cur)
    return phases

def plot_run(phases, out_png, csv_tag, title):
    # 全局迭代轴
    x_ofs = 0
    phase_ranges = []
    for name, rows in phases:
        phase_ranges.append((name, x_ofs, x_ofs + len(rows)))
        x_ofs += len(rows)
    fig, (ax_r, ax_l) = plt.subplots(2, 1, figsize=(11, 6.5), facecolor=PAGE,
                                     gridspec_kw={'height_ratios': [1.1, 1]}, sharex=True)
    for ax in (ax_r, ax_l):
        ax.set_facecolor(SURFACE)
    # 相位带 + 边界
    for i, (name, x0, x1) in enumerate(phase_ranges):
        if i % 2 == 0:
            for ax in (ax_r, ax_l): ax.axvspan(x0, x1, color=BAND, lw=0, zorder=0)
        for ax in (ax_r, ax_l):
            ax.axvline(x0, color=GRID, lw=0.8, ls='--', zorder=1)
        ax_r.text((x0+x1)/2, 1.012, name, transform=ax_r.get_xaxis_transform(),
                  ha='center', va='bottom', color=INK2, fontsize=8)
    # rs 面板
    x_all = []; rs_all = []
    off = 0
    for name, rows in phases:
        for row in rows:
            x_all.append(off + row[0]); rs_all.append(row[1])
        off += len(rows)
    ax_r.plot(x_all, rs_all, color=BLUE, lw=1.6, zorder=2)
    ax_r.set_ylim(0.4, 1.02)
    ax_r.set_ylabel('success rate', color=INK2, fontsize=9)
    ax_r.set_title(title, color=INK, fontsize=11, loc='left', pad=10)
    ax_r.grid(color=GRID, lw=0.5, alpha=0.6)
    ax_r.tick_params(colors=MUTED, labelsize=8)
    # loss 面板 (log)
    x_a, y_a, x_c, y_c = [], [], [], []
    off = 0
    for name, rows in phases:
        for row in rows:
            if len(row) >= 4 and not np.isnan(row[2]):
                x_a.append(off + row[0]); y_a.append(row[2])
                x_c.append(off + row[0]); y_c.append(row[3])
        off += len(rows)
    ax_l.plot(x_a, y_a, color=BLUE, lw=1.4, label='actor loss', zorder=2)
    ax_l.plot(x_c, y_c, color=ORANGE, lw=1.4, label='critic loss', zorder=2)
    ax_l.set_yscale('log')
    ax_l.set_ylabel('loss (log)', color=INK2, fontsize=9)
    ax_l.set_xlabel('iteration', color=INK2, fontsize=9)
    ax_l.grid(color=GRID, lw=0.5, alpha=0.6, which='both')
    ax_l.tick_params(colors=MUTED, labelsize=8)
    ax_l.legend(fontsize=8, labelcolor=INK, facecolor=SURFACE, edgecolor=GRID, loc='upper right')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150, facecolor=PAGE)
    plt.close(fig)
    # CSV (MATLAB 可用)
    csv = f'runs/curves_{csv_tag}.csv'
    with open(csv, 'w', encoding='utf-8') as f:
        f.write('phase,iter_local,iter_global,rs,actor_loss,critic_loss\n')
        off = 0
        for name, rows in phases:
            for row in rows:
                rs = row[1]
                a = row[2] if len(row) >= 4 else ''
                c = row[3] if len(row) >= 4 else ''
                f.write(f'{name},{row[0]},{off+row[0]},{rs},{a},{c}\n')
            off += len(rows)
    print(f'  -> {out_png}, {csv} ({len(phases)} phases)')

if __name__ == '__main__':
    os.makedirs('runs', exist_ok=True)
    p1 = parse_log(LOG_V1)
    plot_run(p1, 'runs/viz_curves_v1.png', 'v1', 'Full pipeline (v1): TOL curriculum -> SelWP -> continuation')
    p2 = parse_log(LOG_V2)
    plot_run(p2, 'runs/viz_curves_v2.png', 'v2', 'Extended continuation (v2): polar/random alternating 1000 iters')
    print('Done.')
