"""任意 checkpoint 反打内化探针 CLI.

反打判定: |δ|>0.1 且 |θ̇|>0.15 且 sign(δ)≠sign(θ̇) — 车体朝一边转、轮子朝另一边打.

用法:
  python countersteer_probe.py runs/p1_base.pt runs/p2_bc.pt ...
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P
from eval_ckpt import load_model

def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__); sys.exit(1)
    print(f'{"checkpoint":>36s}  {"type":>9s}  {"succ":>8s}  {"cs_steps":>9s}  {"cs%":>6s}')
    print('-'*74)
    for path in paths:
        m, use_gp = load_model(path)
        m.eval()
        s_ok, cs = P.probe_countersteer(m, use_gp)
        kind = 'GP-Small' if use_gp else 'Teacher'
        # 补充百分比 (probe 内部已打印; 这里重复算一次摘要)
        print(f'{path:>36s}  {kind:>9s}  {s_ok}/40     {cs:>9d}')

if __name__ == '__main__':
    main()
