"""任意 checkpoint 评估: 自动识别教师/蒸馏模型, 随机场景成功率 + 平均速度.

识别规则: checkpoint 含 model_state_dict → 蒸馏模型 (GPSmall);
         含 actor_state_dict → 教师 (GatedConcatActor).

用法:
  python eval_ckpt.py runs/kamm533_teacher.pt
  python eval_ckpt.py runs/kamm533_teacher.pt runs/kamm533_teacher_v1_c550.pt runs/gp_small_kamm533.pt
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P

N_RANDOM = 200   # 随机场景数 (种子 1000+, 与管线内建评估的 0-99 错开)

def load_model(path):
    ck = torch.load(path, map_location=P.DEV, weights_only=False)
    if 'model_state_dict' in ck:
        arch = ck.get('arch', 'gpsmall')
        cls = {'gpsmall': P.GPSmall, 'gpmed': P.GPMedium}.get(arch)
        if cls is None: raise ValueError(f'未知学生架构: {arch}')
        m = cls().to(P.DEV); m.load_state_dict(ck['model_state_dict'])
        return m, True   # 蒸馏模型
    if 'actor_state_dict' in ck:
        m = P.GatedConcatActor().to(P.DEV); m.load_state_dict(ck['actor_state_dict'])
        P.set_teacher_deterministic(m)
        return m, False  # 教师
    raise ValueError(f'无法识别的 checkpoint: {path}')

def eval_random(m, use_gp):
    scenes = []
    for i in range(N_RANDOM):
        rng = np.random.RandomState(1000 + i)
        s0 = np.array([rng.uniform(P.M, P.FLD-P.M), rng.uniform(P.M, P.FLD-P.M),
                       rng.uniform(-np.pi, np.pi), rng.uniform(0.5, P.V_MAX),
                       rng.uniform(-P.DELTA_MAX, P.DELTA_MAX)], dtype=np.float32)
        tg = [np.array([rng.uniform(P.M, P.FLD-P.M), rng.uniform(P.M, P.FLD-P.M)], dtype=np.float32) for _ in range(3)]
        scenes.append((s0, tg))
    succ, times = P.eval_batch(m, use_gp, scenes)
    s_ok = int(succ.sum()); ok_times = times[succ]
    avg = float(ok_times.mean()) if len(ok_times) else float('nan')
    return s_ok, avg

def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__); sys.exit(1)
    print(f'{"checkpoint":>38s}  {"type":>8s}  {"succ":>8s}  {"avg time":>9s}')
    print('-'*72)
    for path in paths:
        m, use_gp = load_model(path)
        m.eval()
        s_ok, avg = eval_random(m, use_gp)
        kind = 'GP-Small' if use_gp else 'Teacher'
        print(f'{path:>38s}  {kind:>8s}  {s_ok}/{N_RANDOM}   {avg:.1f}s')

if __name__ == '__main__':
    main()
