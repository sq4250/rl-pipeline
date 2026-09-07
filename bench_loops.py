"""多绕圈检测基准: 到达目标 k 后, 若到 k+1 的距离先增大 >0.5m 再减小 → 记一次多绕.

衡量"长腿后接急转"场景的效率 (时间/绕行量/路径比), 用于 retrain 前后对比.

用法: python bench_loops.py [checkpoint ...]    # 默认对比 teacher + gp-small
输出: 每场景的 succ / time / 绕行事件 / 路径直线比
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P
import viz
from eval_ckpt import load_model
from viz_multitarget import showcases

def detect_loops(traj, tgts):
    """返回 (绕行事件 [(目标k, 最大绕行量m)], 各目标命中帧)."""
    gi = 0; events = []; hits = [0]
    for i in range(1, len(traj)):
        if gi < len(tgts) and P.check_hit_substep_np(traj[i-1][:2], traj[i][:2], tgts[gi], P.TOL):
            gi += 1; hits.append(i)
    for k in range(len(hits)-1):
        if k+1 >= len(tgts): break
        seg = traj[hits[k]:hits[k+1]]
        d0 = np.hypot(seg[0,0]-tgts[k+1][0], seg[0,1]-tgts[k+1][1])
        dmax = max(np.hypot(seg[:,0]-tgts[k+1][0], seg[:,1]-tgts[k+1][1]))
        if dmax - d0 > 0.5: events.append((k+1, round(float(dmax-d0), 2)))
    return events, hits

def main():
    paths = sys.argv[1:] or ['runs/kamm533_teacher.pt', 'runs/gp_small_kamm533.pt']
    models = []
    for p in paths:
        m, use_gp = load_model(p); m.eval()
        models.append((os.path.basename(p)[:24], m, use_gp))
    print(f'{"scene":>14s} {"model":>24s} {"succ":>5s} {"time":>6s} {"loops":>14s} {"path/str":>8s}')
    for name, s0, tgts in showcases():
        straight = np.hypot(tgts[0][0]-s0[0], tgts[0][1]-s0[1]) + sum(
            np.hypot(tgts[k+1][0]-tgts[k][0], tgts[k+1][1]-tgts[k][1]) for k in range(len(tgts)-1))
        for tag, m, use_gp in models:
            ta, _, gi = viz.run_traj(m, use_gp, s0.copy(), tgts)
            ev, hits = detect_loops(ta, tgts)
            plen = sum(np.hypot(*(ta[i+1,:2]-ta[i,:2])) for i in range(len(ta)-1))
            print(f'{name:>14s} {tag:>24s} {gi:2d}/{len(tgts):<2d} {(len(ta)-1)*P.DT:5.1f}s  {str(ev) if ev else "-":>14s}  {plen/straight:6.2f}x')

if __name__ == '__main__':
    main()
