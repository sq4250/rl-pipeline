"""Smoke test: exercise every pipeline component with tiny scale."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import pipeline as P

# 1. Sim
sp = torch.rand(16, 5, device=P.DEV)
act = torch.randn(16, 2, device=P.DEV)
sn = P.SIM.simulate(sp, act)
assert sn.shape == (16, 5), sn.shape
print('1. KammSim OK')

# 2. Scene gens
for fn in (P.gen_scenes_random, P.gen_scenes_polar):
    out = fn(64, P.DEV)
    assert all(t.shape[0] == 64 for t in out), [t.shape for t in out]
print('2. scene gens OK')
out = P.gen_mixed(64, P.DEV)
print('   gen_mixed OK')

# 3. PPO iteration (tiny scale)
P.E = 64
actor = P.GatedConcatActor().to(P.DEV); critic = P.GatedConcatCritic().to(P.DEV)
scaler = torch.amp.GradScaler('cuda')
opt_a, opt_c = P.make_opts(actor, critic)
r = P.do_iteration(actor, critic, opt_a, opt_c, scaler, use_polar=True)
print('3. do_iteration OK:', tuple(f'{x:.3f}' if isinstance(x, float) else x for x in r))
r = P.do_iteration(actor, critic, opt_a, opt_c, scaler, use_polar=False, tol_val=2.0)
print('   do_iteration (TOL=2.0, random) OK')

# 4. run_p2_bc (1 iter)
blank_a = P.GatedConcatActor().to(P.DEV); blank_c = P.GatedConcatCritic().to(P.DEV)
best = P.run_p2_bc(actor, blank_a, blank_c, 1)
print('4. run_p2_bc OK, suc=', f'{best:.3f}')

# 5. Eval (1 trial each)
P.set_teacher_deterministic(actor)
P.eval_model(actor, False, n_scene_trials=1)
print('5. eval_model teacher OK')

# 6. Distill components (tiny)
student = P.GPSmall().to(P.DEV)
X, V2, V3, Y = P.collect_bc(actor, noise_std=0.03)
print(f'6. collect_bc OK: {X.shape[0]} samples')
P.train_epochs(student, X, V2, V3, Y, 2, 2e-3)
print('   train_epochs OK')
t, st, sr = P.calc_mean_time(student, n_random=10)
print(f'   calc_mean_time OK: {t:.2f}s  tight={st:.2f}  rand={sr:.2f}')
P.eval_model(student, True, n_scene_trials=1)
print('7. eval_model gp OK')

# 7. SelWP geometry sanity
st = torch.tensor([[3.5, 3.5, 0.0, 0.8, 0.0]], device=P.DEV)
tg = torch.tensor([[3.5, 2.5]], device=P.DEV)  # 正后方 1m
in_c, dx, dy, Rmin = P.unreachable_mask(st, tg)
wp = P.compute_wp_world(st, dx, dy, Rmin)
print(f'8. SelWP: in_circle={in_c.item()} R_min={Rmin.item():.3f} wp={wp[0].tolist()}')

print('\nALL SMOKE TESTS PASSED')
