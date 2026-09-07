"""P2 近目标加权 A/B: 同一 p1_base 冻结教师, weighted vs baseline BC → 反打探针对比.

用法:
  python ab_p2_weight.py           # 小规模 (E=256, 100 iter) 两轮串行
输出: runs/ab_p2_weighted.pt / runs/ab_p2_baseline.pt + 探针结果
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import pipeline as P

P.E = 256          # 1/4 并行环境 (完整 P2 为 1024)
N_ITER = 100       # 缩减迭代 (完整 P2 为 300)

def run_one(near_weight, tag):
    torch.manual_seed(1234)   # 两轮同种子 → 场景序列一致, A/B 公平
    frozen = P.GatedConcatActor().to(P.DEV)
    dummy_c = P.GatedConcatCritic().to(P.DEV)
    P.load_ckpt('runs/p1_base.pt', frozen, dummy_c)
    blank_a = P.GatedConcatActor().to(P.DEV)
    blank_c = P.GatedConcatCritic().to(P.DEV)
    print(f'\n{"#"*60}\n# A/B run: {tag} (near_weight={near_weight})\n{"#"*60}')
    best = P.run_p2_bc(frozen, blank_a, blank_c, N_ITER, near_weight=near_weight)
    P.set_teacher_deterministic(blank_a)
    torch.save({'actor_state_dict': blank_a.state_dict(),
                'critic_state_dict': blank_c.state_dict(),
                'best': best, 'near_weight': near_weight,
                'E': P.E, 'n_iter': N_ITER}, f'runs/ab_p2_{tag}.pt')
    print(f'\n=== probe: {tag} ===')
    return P.probe_countersteer(blank_a, False)

if __name__ == '__main__':
    s1, c1 = run_one(True, 'weighted')
    s2, c2 = run_one(False, 'baseline')
    print(f'\n{"="*60}\nA/B SUMMARY (E=256, {N_ITER} iter)')
    print(f'  weighted: succ={s1}/40  cs_steps={c1}')
    print(f'  baseline: succ={s2}/40  cs_steps={c2}')
    print('='*60)
