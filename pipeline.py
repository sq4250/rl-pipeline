"""Kamm 5/3/3 自包含完整管线 — 冷启动 → SelWP → 续训 → 蒸馏.

唯一依赖: torch + numpy. 无任何外部模块引用.

Phase A: 基础到达能力 (TOL 课程 → transition → polar 探索 → anneal)
Phase B: SelWP 融合 (frozen base 仅 rollout + 100% SelWP + 速度门 → blank 网络)
Phase C: 续训 (噪声破死锁 → random straight-line → 最终 anneal, 无 SelWP)
Phase D: 蒸馏 (BC 距离加权 + DAgger×12 速度选优 → GP-Small 3.7K)

用法:
  python pipeline.py            # 完整流程 (A→B→C→D)
  python pipeline.py --from c   # 从 Phase C 续跑 (需 runs/p2_bc.pt)
  python pipeline.py --from d   # 从 Phase D 续跑 (需 runs/kamm533_teacher.pt)
"""
import os, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('runs', exist_ok=True)

# ═══════════════════ 全局配置 ═══════════════════
A_LONG  = 5.0      # 纵向加速上限 [m/s²]
A_BRAKE = 3.0      # 制动上限 [m/s²]
A_LAT   = 3.0      # 横向加速度上限 [m/s²]  (R_min = v²/3)
A_MAX   = A_LONG
V_MAX   = 5.0      # 最大速度 [m/s]
O_MAX   = 14.0     # 前轮转角角速度上限 [rad/s]
WHEELBASE = 0.15                       # 轴距 [m]
DELTA_MAX = float(np.arctan(WHEELBASE / 0.30))  # = arctan(0.5) ≈ 0.4636
DT, TOL, FLD, M = 0.05, 0.05, 7.0, 0.5   # 外步 / 到达容差 / 场地 / 边距
NSR = 0.3                                # 距离塑形系数
D_SCALE = 5.0                            # 距离归一化尺度 (观测向量中 d / D_SCALE)
E, STEPS = 1024, 180                     # 并行环境数 / 单 episode 步数
GAMMA, LAM, EPS, VC, MG, BS = 0.995, 0.95, 0.15, 0.5, 1.0, 2048
CAR_X, CAR_Y, CAR_TH = 3.5, 3.5, 0.0
DEV = torch.device('cuda')

# ═══════════════════ 物理仿真 (Kamm 非对称摩擦椭圆) ═══════════════════
class KammSim:
    def __init__(self):
        self.am, self.bm, self.lm = A_LONG, A_BRAKE, A_LAT
        self.vm, self.om = V_MAX, O_MAX
        self.dm, self.L, self.dt = DELTA_MAX, WHEELBASE, 0.005
    def simulate(self, s, a, dt_outer=DT):
        n = int(dt_outer / self.dt); s = s.clone()
        for _ in range(n):
            a_raw, o_raw = a[:, 0], a[:, 1]
            v, delta, th = s[:, 3], s[:, 4], s[:, 2]
            omega = torch.clamp(o_raw, -self.om, self.om)
            a_nom = torch.clamp(a_raw, -self.bm, self.am)
            v_nom = v + a_nom * self.dt
            v_clip = (v_nom < 0.) | (v_nom > self.vm)
            v_nom = torch.clamp(v_nom, 0., self.vm)
            a_long = a_nom.clone(); a_long[v_clip] = (v_nom[v_clip] - v[v_clip]) / self.dt
            semi = torch.where(a_long >= 0, self.am, self.bm)
            r = torch.clamp(a_long / (semi + 1e-8), -1., 1.)
            alm = self.lm * torch.sqrt(torch.clamp(1. - r**2, min=0.))
            vs = torch.clamp(v_nom, min=0.01)
            dl = torch.atan(alm * self.L / vs**2)
            md = torch.minimum(torch.full_like(dl, self.dm), dl)
            dn = delta + omega * self.dt; dn = torch.clamp(dn, -md, md)
            omega = (dn - delta) / self.dt
            xn = s[:, 0] + v * torch.cos(th) * self.dt
            yn = s[:, 1] + v * torch.sin(th) * self.dt
            th_n = th + v * torch.tan(dn) / self.L * self.dt
            th_n = torch.atan2(torch.sin(th_n), torch.cos(th_n))
            s = torch.stack([xn, yn, th_n, v_nom, dn], dim=1)
        return s

SIM = KammSim()

# ═══════════════════ 命中检测 ═══════════════════
def check_hit_substep_np(prev_xy, next_xy, target_xy, tol):
    """线段 prev→next 是否穿过 target 的 TOL 圆 (标量 numpy 版)."""
    ax, ay = prev_xy[0], prev_xy[1]; bx, by = next_xy[0], next_xy[1]
    px, py = target_xy[0], target_xy[1]
    if (bx - px)**2 + (by - py)**2 < tol*tol: return True
    abx, aby = bx - ax, by - ay; ab2 = abx*abx + aby*aby
    if ab2 < 1e-12: return False
    t = max(0.0, min(1.0, ((px - ax)*abx + (py - ay)*aby) / ab2))
    cx, cy = ax + t*abx, ay + t*aby
    return (px - cx)**2 + (py - cy)**2 < tol*tol

def check_hit_batch(prev_xy, next_xy, target_xy, tol):
    """GPU 向量化命中检测. 输入均为 (B,2)."""
    ax, ay = prev_xy[:, 0], prev_xy[:, 1]; bx, by = next_xy[:, 0], next_xy[:, 1]
    px, py = target_xy[:, 0], target_xy[:, 1]
    hit = ((bx - px)**2 + (by - py)**2) < tol*tol
    abx, aby = bx - ax, by - ay; ab2 = abx*abx + aby*aby
    t = torch.clamp(((px - ax)*abx + (py - ay)*aby) / (ab2 + 1e-12), 0., 1.)
    cx, cy = ax + t*abx, ay + t*aby
    return hit | (((px - cx)**2 + (py - cy)**2) < tol*tol)

# ═══════════════════ 网络 ═══════════════════
class GatedConcatActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_enc = nn.Linear(2, 32); self.target_enc = nn.Linear(2, 16)
        self.trunk = nn.Sequential(nn.Linear(80, 128), nn.GELU(), nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU())
        self.mean_head = nn.Linear(128, 2)
        self.log_std = nn.Parameter(torch.tensor([0.5, 1.5]))
    def forward(self, o1, o2, o3, v2, v3):
        s = self.state_enc(o1[:, :2]); t1 = self.target_enc(o1[:, 2:])
        t2 = v2.unsqueeze(1) * self.target_enc(o2)
        t3 = v3.unsqueeze(1) * self.target_enc(o3)
        m = self.mean_head(self.trunk(torch.cat([s, t1, t2, t3], dim=1)))
        a_center = (A_LONG - A_BRAKE) / 2; a_half = (A_LONG + A_BRAKE) / 2
        return torch.stack([torch.tanh(m[:, 0]) * a_half + a_center, torch.tanh(m[:, 1]) * O_MAX], dim=1), self.log_std.exp()
    def sample(self, o1, o2, o3, v2, v3):
        mean, std = self.forward(o1, o2, o3, v2, v3)
        dist = torch.distributions.Normal(mean, std)
        act = dist.sample()
        return act, dist.log_prob(act).sum(dim=-1), mean.detach()

class GatedConcatCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_enc = nn.Linear(2, 32); self.target_enc = nn.Linear(2, 16)
        self.trunk = nn.Sequential(nn.Linear(80, 128), nn.GELU(), nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU())
        self.head = nn.Linear(128, 1)
    def forward(self, o1, o2, o3, v2, v3):
        s = self.state_enc(o1[:, :2]); t1 = self.target_enc(o1[:, 2:])
        t2 = v2.unsqueeze(1) * self.target_enc(o2)
        t3 = v3.unsqueeze(1) * self.target_enc(o3)
        return self.head(self.trunk(torch.cat([s, t1, t2, t3], dim=1))).squeeze(-1)

class GPSmall(nn.Module):
    """蒸馏学生: 3762 参数."""
    def __init__(self):
        super().__init__()
        self.state_enc = nn.Linear(2, 8); self.target_enc = nn.Linear(2, 8)
        self.fc1 = nn.Linear(32, 48); self.fc2 = nn.Linear(48, 32); self.fc3 = nn.Linear(32, 16); self.fc4 = nn.Linear(16, 2)
        for m_ in [self.state_enc, self.target_enc]: nn.init.xavier_uniform_(m_.weight); nn.init.zeros_(m_.bias)
        for m_ in [self.fc1, self.fc2, self.fc3]: nn.init.kaiming_uniform_(m_.weight, nonlinearity='relu'); nn.init.constant_(m_.bias, 0.01)
        nn.init.xavier_uniform_(self.fc4.weight); nn.init.zeros_(self.fc4.bias)
    def forward(self, p8, v2, v3):
        s = self.state_enc(p8[:, :2]); t1 = self.target_enc(p8[:, 2:4])
        t2 = v2.unsqueeze(1) * self.target_enc(p8[:, 4:6])
        t3 = v3.unsqueeze(1) * self.target_enc(p8[:, 6:8])
        h = F.relu(self.fc1(torch.cat([s, t1, t2, t3], dim=1))); h = F.relu(self.fc2(h)); h = F.relu(self.fc3(h))
        return self.fc4(h)
    def deploy(self, p8, v2, v3):
        r = self.forward(p8, v2, v3)
        return torch.stack([torch.clamp(r[:, 0], -A_BRAKE, A_LONG), torch.clamp(r[:, 1], -O_MAX, O_MAX)], dim=1)

# ═══════════════════ 观测与 SelWP ═══════════════════
def polar_obs(sp, g1, g2, g3):
    x, y, th = sp[:, 0], sp[:, 1], sp[:, 2]; v, dlt = sp[:, 3], sp[:, 4]
    def seg(a, b):
        dx, dy = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
        return torch.sqrt(dx**2 + dy**2), torch.atan2(dy, dx)
    def wrap(a, b): return torch.atan2(torch.sin(a - b), torch.cos(a - b))
    d1, h_car_g1 = seg(sp[:, :2], g1); d12, h_g1_g2 = seg(g1, g2); d23, h_g2_g3 = seg(g2, g3)
    o1 = torch.stack([v / V_MAX, dlt / DELTA_MAX, d1 / D_SCALE, wrap(h_car_g1, th) / np.pi], dim=1)
    o2 = torch.stack([d12 / D_SCALE, wrap(h_g1_g2, h_car_g1) / np.pi], dim=1)
    o3 = torch.stack([d23 / D_SCALE, wrap(h_g2_g3, h_g1_g2) / np.pi], dim=1)
    return o1, o2, o3
def polar_8d(sp, g1, g2, g3):
    o1, o2, o3 = polar_obs(sp, g1, g2, g3); return torch.cat([o1, o2, o3], dim=1)

def unreachable_mask(states, cg):
    """触发判定用真实物理 R_min, 无 margin."""
    x, y, th = states[:, 0], states[:, 1], states[:, 2]; v = states[:, 3]
    dx_w, dy_w = cg[:, 0] - x, cg[:, 1] - y
    c, s = torch.cos(th), torch.sin(th)
    dx = dx_w * c + dy_w * s; dy = -dx_w * s + dy_w * c
    R_min = v**2 / A_LAT
    in_circle = (dx**2 + (dy - R_min)**2 < R_min**2) | (dx**2 + (dy + R_min)**2 < R_min**2)
    return in_circle, dx, dy, R_min

def compute_wp_world(states, dx, dy, R_min):
    """WP 生成用 R_min+0.05. 车身系几何 → 世界坐标."""
    x, y, th = states[:, 0], states[:, 1], states[:, 2]
    sign = torch.where(dy >= 0, torch.tensor(1., device=DEV), torch.tensor(-1., device=DEV))
    R = R_min + 0.05
    c1x = torch.zeros_like(x); c1y = -sign * R
    tx, ty = dx, dy
    d_c1_t = torch.sqrt((tx - c1x)**2 + (ty - c1y)**2 + 1e-12)
    a = ((2*R)**2 - R**2 + d_c1_t**2) / (2*d_c1_t + 1e-12)
    h = torch.sqrt(torch.clamp((2*R)**2 - a**2, min=0.) + 1e-12)
    mx = c1x + a * (tx - c1x) / (d_c1_t + 1e-12); my = c1y + a * (ty - c1y) / (d_c1_t + 1e-12)
    px = -(ty - c1y) / (d_c1_t + 1e-12); py = (tx - c1x) / (d_c1_t + 1e-12)
    c2x_a, c2y_a = mx + h*px, my + h*py
    c2x_b, c2y_b = mx - h*px, my - h*py
    wp_x_a, wp_x_b = (c1x + c2x_a)/2., (c1x + c2x_b)/2.
    pick_a = wp_x_a > 0
    wp_x_body = torch.where(pick_a, wp_x_a, wp_x_b)
    wp_y_body = torch.where(pick_a, (c1y + c2y_a)/2., (c1y + c2y_b)/2.)
    c, s = torch.cos(th), torch.sin(th)
    return torch.stack([x + wp_x_body*c - wp_y_body*s, y + wp_x_body*s + wp_y_body*c], dim=1)

# ═══════════════════ 场景生成 ═══════════════════
def _polar_angles(n, n_side, n_behind, angle_sigma, dev_local):
    th_a = torch.randn(n_side, device=dev_local) * angle_sigma + np.pi/2
    th_b = torch.randn(n_side, device=dev_local) * angle_sigma - np.pi/2
    th_c = torch.randn(n_behind, device=dev_local) * angle_sigma
    sign_c = (torch.randint(0, 2, (n_behind,), device=dev_local).float() * 2 - 1)
    th_c = sign_c * np.pi + th_c
    th = torch.cat([th_a, th_b, th_c], dim=0)
    th = torch.atan2(torch.sin(th), torch.cos(th))
    return th[torch.randperm(n, device=dev_local)]

LONG_MIN, LONG_FRAC = 3.0, 0.25   # 长腿: [LONG_MIN, max_d] 均匀采样, 占比/腿 (≈58% 场景含 ≥1 长腿)
CLOSE_MIN, CLOSE_MAX, CLOSE_FRAC = 0.05, 0.65, 0.20   # 超近目标带: 补足 0.5m 以内的欠采样 (反打触发区)

def _polar_dists(n, min_d, max_d, dist_sigma, dev_local, long_frac=LONG_FRAC, close_frac=CLOSE_FRAC):
    """拒绝采样距离, 保证返回恰 n 个 (候选不足时循环补足).

    混合三带:
      close_frac: 超近目标 [CLOSE_MIN, CLOSE_MAX] 均匀 — 0.5m 以内欠采样补足 (反打触发区)
      long_frac:  长腿 [LONG_MIN, max_d] 均匀 — 远距离 (全速逼近→提前刹车) 欠采样补足
      其余: 半正态 [min_d, max_d]
    """
    n_long = int(n * long_frac); n_close = int(n * close_frac)
    n_short = n - n_long - n_close
    d = torch.abs(torch.randn(max(10, int(n*4)), device=dev_local) * dist_sigma)
    d = d[(d >= min_d) & (d <= max_d)]
    while d.shape[0] < n_short:
        extra = torch.abs(torch.randn(max(10, int(n*4)), device=dev_local) * dist_sigma)
        d = torch.cat([d, extra[(extra >= min_d) & (extra <= max_d)]])
    d = d[:n_short]
    if n_close > 0:
        dc = torch.rand(max(10, int(n_close*4)), device=dev_local) * (CLOSE_MAX - CLOSE_MIN) + CLOSE_MIN
        while dc.shape[0] < n_close:
            extra = torch.rand(max(10, int(n_close*4)), device=dev_local) * (CLOSE_MAX - CLOSE_MIN) + CLOSE_MIN
            dc = torch.cat([dc, extra])
        d = torch.cat([d, dc[:n_close]])
    if n_long > 0:
        dl = torch.rand(max(10, int(n_long*4)), device=dev_local) * (max_d - LONG_MIN) + LONG_MIN
        dl = dl[dl <= max_d]
        while dl.shape[0] < n_long:
            extra = torch.rand(max(10, int(n_long*4)), device=dev_local) * (max_d - LONG_MIN) + LONG_MIN
            dl = torch.cat([dl, extra[extra <= max_d]])
        d = torch.cat([d, dl[:n_long]])
    return d[torch.randperm(n, device=dev_local)]

def gen_scenes_random(n, dev_local, full_state=False):
    """简单随机航点 — Phase A 基础训练 / Phase C straight-line."""
    v_scale = V_MAX if full_state else V_MAX * 0.5
    d_scale = DELTA_MAX if full_state else DELTA_MAX * 0.3
    sp = torch.stack([torch.full((n,), CAR_X, device=dev_local), torch.full((n,), CAR_Y, device=dev_local),
        torch.full((n,), CAR_TH, device=dev_local), torch.rand(n, device=dev_local) * v_scale,
        (torch.rand(n, device=dev_local)*2 - 1) * d_scale], dim=1)
    gs = [torch.stack([torch.rand(n, device=dev_local)*(FLD-2*M)+M, torch.rand(n, device=dev_local)*(FLD-2*M)+M], dim=1) for _ in range(3)]
    return sp, gs[0], gs[1], gs[2]

def gen_scenes_polar(n, dev_local, full_state=False):
    """增量 polar 链: 车→g1→g2→g3, 侧前40%+侧后40%+正后20%.

    full_state: 初始状态全幅 (v≤V_MAX, δ≤δmax) — 冷启动 (P1) 之后使用.
    """
    n_side = int(n * 0.4); n_behind = n - 2*n_side
    min_d, max_d = 0.5, 6.0
    v_scale = V_MAX if full_state else V_MAX * 0.5
    d_scale = DELTA_MAX if full_state else DELTA_MAX * 0.3
    v = torch.rand(n, device=dev_local) * v_scale
    delta = (torch.rand(n, device=dev_local)*2 - 1) * d_scale
    sp = torch.stack([torch.full((n,), CAR_X, device=dev_local), torch.full((n,), CAR_Y, device=dev_local),
        torch.full((n,), CAR_TH, device=dev_local), v, delta], dim=1)
    theta1 = _polar_angles(n, n_side, n_behind, np.pi/4, dev_local)
    d1 = _polar_dists(n, min_d, max_d, 1.5, dev_local)
    g1x = CAR_X + d1*torch.cos(theta1); g1y = CAR_Y + d1*torch.sin(theta1)
    dth12 = _polar_angles(n, n_side, n_behind, np.pi/4, dev_local)
    d12 = _polar_dists(n, min_d, max_d, 1.5, dev_local)
    h_car_g1 = torch.atan2(g1y - CAR_Y, g1x - CAR_X)
    g2x = g1x + d12*torch.cos(h_car_g1 + dth12); g2y = g1y + d12*torch.sin(h_car_g1 + dth12)
    dth23 = _polar_angles(n, n_side, n_behind, np.pi/4, dev_local)
    d23 = _polar_dists(n, min_d, max_d, 1.5, dev_local)
    h_g1_g2 = torch.atan2(g2y - g1y, g2x - g1x)
    g3x = g2x + d23*torch.cos(h_g1_g2 + dth23); g3y = g2y + d23*torch.sin(h_g1_g2 + dth23)
    for arr in (g1x, g1y, g2x, g2y, g3x, g3y): arr.clamp_(M, FLD - M)
    return sp, torch.stack([g1x, g1y], dim=1), torch.stack([g2x, g2y], dim=1), torch.stack([g3x, g3y], dim=1)

def _ga_mixed(n, dev_local):
    """蒸馏场景角度: 侧前35% + 侧后35% + 正后15% + 直行15%."""
    n_side = int(n*0.35); n_behind = int(n*0.15); n_straight = n - 2*n_side - n_behind
    sigma = np.pi/4
    a = torch.randn(n_side, device=dev_local)*sigma + np.pi/2
    b = torch.randn(n_side, device=dev_local)*sigma - np.pi/2
    c = torch.randn(n_behind, device=dev_local)*sigma*0.6
    sign = torch.randint(0, 2, (n_behind,), device=dev_local).float()*2 - 1
    c = sign*np.pi + c
    d = torch.randn(n_straight, device=dev_local)*sigma*0.4
    th = torch.cat([a, b, c, d], dim=0)
    th = torch.atan2(torch.sin(th), torch.cos(th))
    return th[torch.randperm(n, device=dev_local)]

def _gd_mixed(n, min_d, max_d, dev_local, long_frac=LONG_FRAC, close_frac=CLOSE_FRAC):
    """蒸馏场景距离 — 同 _polar_dists: 超近+长腿双带补足欠采样."""
    n_long = int(n * long_frac); n_close = int(n * close_frac)
    n_short = n - n_long - n_close
    d = torch.abs(torch.randn(max(10, int(n*4)), device=dev_local)*2.0)
    d = d[(d >= min_d) & (d <= max_d)]
    while d.shape[0] < n_short:
        extra = torch.abs(torch.randn(max(10, int(n*4)), device=dev_local)*2.0)
        d = torch.cat([d, extra[(extra >= min_d) & (extra <= max_d)]])
    d = d[:n_short]
    if n_close > 0:
        dc = torch.rand(max(10, int(n_close*4)), device=dev_local) * (CLOSE_MAX - CLOSE_MIN) + CLOSE_MIN
        while dc.shape[0] < n_close:
            extra = torch.rand(max(10, int(n_close*4)), device=dev_local) * (CLOSE_MAX - CLOSE_MIN) + CLOSE_MIN
            dc = torch.cat([dc, extra])
        d = torch.cat([d, dc[:n_close]])
    if n_long > 0:
        dl = torch.rand(max(10, int(n_long*4)), device=dev_local) * (max_d - LONG_MIN) + LONG_MIN
        dl = dl[dl <= max_d]
        while dl.shape[0] < n_long:
            extra = torch.rand(max(10, int(n_long*4)), device=dev_local) * (max_d - LONG_MIN) + LONG_MIN
            dl = torch.cat([dl, extra[extra <= max_d]])
        d = torch.cat([d, dl[:n_long]])
    return d[torch.randperm(n, device=dev_local)]

def gen_mixed(n, dev_local):
    """蒸馏数据场景: 全速全转角."""
    min_d, max_d = 0.3, 6.0
    v = torch.rand(n, device=dev_local)*V_MAX
    delta = (torch.rand(n, device=dev_local)*2 - 1)*DELTA_MAX
    sp = torch.stack([torch.full((n,), CAR_X, device=dev_local), torch.full((n,), CAR_Y, device=dev_local),
        torch.full((n,), CAR_TH, device=dev_local), v, delta], dim=1)
    t1 = _ga_mixed(n, dev_local); d1 = _gd_mixed(n, min_d, max_d, dev_local)
    g1x = CAR_X + d1*torch.cos(t1); g1y = CAR_Y + d1*torch.sin(t1)
    dt12 = _ga_mixed(n, dev_local); d12 = _gd_mixed(n, min_d, max_d, dev_local)
    hh = torch.atan2(g1y - CAR_Y, g1x - CAR_X)
    g2x = g1x + d12*torch.cos(hh + dt12); g2y = g1y + d12*torch.sin(hh + dt12)
    dt23 = _ga_mixed(n, dev_local); d23 = _gd_mixed(n, min_d, max_d, dev_local)
    h2 = torch.atan2(g2y - g1y, g2x - g1x)
    g3x = g2x + d23*torch.cos(h2 + dt23); g3y = g2y + d23*torch.sin(h2 + dt23)
    return (sp,
        torch.stack([torch.clamp(g1x, M, FLD-M), torch.clamp(g1y, M, FLD-M)], dim=1),
        torch.stack([torch.clamp(g2x, M, FLD-M), torch.clamp(g2y, M, FLD-M)], dim=1),
        torch.stack([torch.clamp(g3x, M, FLD-M), torch.clamp(g3y, M, FLD-M)], dim=1))

# ═══════════════════ 优化器 ═══════════════════
def make_opts(actor, critic, lr_scale=1.0):
    opt_a = torch.optim.Adam([
        {'params': actor.state_enc.parameters(), 'lr': 2.5e-5*lr_scale},
        {'params': actor.target_enc.parameters(), 'lr': 2.5e-5*lr_scale},
        {'params': actor.trunk.parameters(), 'lr': 2.5e-5*lr_scale},
        {'params': actor.mean_head.parameters(), 'lr': 5e-5*lr_scale},
        {'params': [actor.log_std], 'lr': 5e-5*lr_scale}])
    opt_c = torch.optim.Adam([
        {'params': critic.state_enc.parameters(), 'lr': 5e-5*lr_scale},
        {'params': critic.target_enc.parameters(), 'lr': 5e-5*lr_scale},
        {'params': critic.trunk.parameters(), 'lr': 5e-5*lr_scale},
        {'params': critic.head.parameters(), 'lr': 1e-4*lr_scale}])
    return opt_a, opt_c

# ═══════════════════ PPO 迭代 (纯 NN, 无 SelWP) ═══════════════════
def do_iteration(actor, critic, opt_a, opt_c, scaler, use_polar=True, tol_val=None,
                 actor_frozen=False, critic_frozen=False, full_state=False):
    if use_polar: sp, g1, g2, g3 = gen_scenes_polar(E, DEV, full_state)
    else:         sp, g1, g2, g3 = gen_scenes_random(E, DEV, full_state)
    targets_t = torch.stack([g1, g2, g3], dim=1)
    phase = torch.zeros(E, dtype=torch.int32, device=DEV)
    alive = torch.ones(E, dtype=torch.bool, device=DEV)
    _tol = tol_val if tol_val else TOL
    iL = []; sr = 0.0
    for _ in range(STEPS):
        if not alive.any(): break
        i0 = torch.clamp(phase.long(), 0, 2); i1 = torch.clamp(phase.long()+1, 0, 2); i2 = torch.clamp(phase.long()+2, 0, 2)
        cg = targets_t[torch.arange(E, device=DEV), i0]
        ng = targets_t[torch.arange(E, device=DEV), i1]
        nng = targets_t[torch.arange(E, device=DEV), i2]
        v2 = (phase < 2).float(); v3 = (phase < 1).float()
        o1, o2, o3 = polar_obs(sp, cg, ng, nng)
        with torch.no_grad(): act, lp, _ = actor.sample(o1, o2, o3, v2, v3)
        with torch.no_grad(): val = critic(o1, o2, o3, v2, v3)
        sn = SIM.simulate(sp, act)
        d_curr = torch.hypot(sp[:, 0]-cg[:, 0], sp[:, 1]-cg[:, 1])
        d_next = torch.hypot(sp[:, 0]-ng[:, 0], sp[:, 1]-ng[:, 1])
        d_curr_new = torch.hypot(sn[:, 0]-cg[:, 0], sn[:, 1]-cg[:, 1])
        d_next_new = torch.hypot(sn[:, 0]-ng[:, 0], sn[:, 1]-ng[:, 1])
        cost = DT + 0.002*(act[:, 0]/A_MAX)**2 + 0.01*(act[:, 1]/O_MAX)**2 + 0.002*torch.abs(act[:, 1])/O_MAX
        reward = -cost + NSR*(d_curr - GAMMA*d_curr_new + d_next - GAMMA*d_next_new)
        atg = check_hit_batch(sp[:, :2], sn[:, :2], cg, _tol)
        iL.append((o1, o2, o3, v2, v3, act, lp, val, reward, alive.clone(), atg & (phase >= 2)))
        sr += reward[alive].mean() if alive.any() else 0.0
        phase = torch.where(atg & alive, phase+1, phase)
        alive = alive & (phase < 3)
        sp = sn
    # GAE
    i0 = torch.clamp(phase.long(), 0, 2); i1 = torch.clamp(phase.long()+1, 0, 2); i2 = torch.clamp(phase.long()+2, 0, 2)
    o1_f, o2_f, o3_f = polar_obs(sp, targets_t[torch.arange(E, device=DEV), i0],
                                  targets_t[torch.arange(E, device=DEV), i1],
                                  targets_t[torch.arange(E, device=DEV), i2])
    with torch.no_grad(): last_val_full = critic(o1_f, o2_f, o3_f, (phase < 2).float(), (phase < 1).float())
    T = len(iL); gae_val = torch.zeros(E, device=DEV); advantages, returns = [], []
    for t_idx in reversed(range(T)):
        _, _, _, _, _, _, _, val_t, reward_t, a_t, done_t = iL[t_idx]
        next_val = last_val_full if t_idx == T-1 else iL[t_idx+1][7]
        next_val = next_val.clone(); next_val[done_t] = 0.0; next_val[~a_t] = 0.0
        reward_t = reward_t.clone(); reward_t[~a_t] = 0.0
        val_t = val_t.clone(); val_t[~a_t] = 0.0
        delta = reward_t + GAMMA*next_val - val_t
        gae_val = delta + GAMMA*LAM*gae_val; gae_val[~a_t] = 0.0
        advantages.insert(0, gae_val[a_t].clone())
        returns.insert(0, (gae_val[a_t] + val_t[a_t]).clone())
    if len(advantages) == 0 or all(a.numel() == 0 for a in advantages): return -1, -1, -1, 0.0
    all_o1 = torch.cat([x[0][x[9]] for x in iL], dim=0); all_o2 = torch.cat([x[1][x[9]] for x in iL], dim=0)
    all_o3 = torch.cat([x[2][x[9]] for x in iL], dim=0); all_v2 = torch.cat([x[3][x[9]] for x in iL], dim=0)
    all_v3 = torch.cat([x[4][x[9]] for x in iL], dim=0); all_act = torch.cat([x[5][x[9]] for x in iL], dim=0)
    all_lp = torch.cat([x[6][x[9]] for x in iL], dim=0)
    all_ret = torch.cat([r.unsqueeze(1) for r in returns], dim=0)
    all_adv = torch.cat([a.unsqueeze(1) for a in advantages], dim=0)
    all_adv = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)
    total_data = all_o1.shape[0]
    if total_data < BS: return -1, -1, -1, 0.0
    perm = torch.randperm(total_data, device=DEV)
    a_tot = torch.tensor(0.0, device=DEV); c_tot = torch.tensor(0.0, device=DEV); nb = 0
    for s0 in range(0, total_data, BS):
        b_idx = perm[s0:s0+BS]
        b_lp_old = all_lp[b_idx]; b_ret = all_ret[b_idx].squeeze(-1); b_adv = all_adv[b_idx].squeeze(-1)
        with torch.amp.autocast('cuda'):
            mean, std = actor.forward(all_o1[b_idx], all_o2[b_idx], all_o3[b_idx], all_v2[b_idx], all_v3[b_idx])
            dist = torch.distributions.Normal(mean, std)
            lp_new = dist.log_prob(all_act[b_idx]).sum(dim=-1)
            ratio = (lp_new - b_lp_old).exp()
            a_loss = -torch.min(ratio*b_adv, torch.clamp(ratio, 1-EPS, 1+EPS)*b_adv).mean()
            v_pred = critic(all_o1[b_idx], all_o2[b_idx], all_o3[b_idx], all_v2[b_idx], all_v3[b_idx])
            v_clipped = b_ret + torch.clamp(v_pred - b_ret, -VC, VC)
            c_loss = MG*torch.max((v_pred - b_ret).pow(2), (v_clipped - b_ret).pow(2)).mean()
        if not actor_frozen:
            opt_a.zero_grad(); scaler.scale(a_loss).backward(); scaler.unscale_(opt_a)
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5); scaler.step(opt_a)
        if not critic_frozen:
            opt_c.zero_grad(); scaler.scale(c_loss).backward(); scaler.unscale_(opt_c)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5); scaler.step(opt_c)
        scaler.update()
        a_tot += a_loss.detach(); c_tot += c_loss.detach(); nb += 1
    return (a_tot/nb).item(), (c_tot/nb).item(), (sr/max(1, len(iL))).item(), float((~alive).float().mean())

LOGSTD_P3 = [(0, -0.5, 0.0), (80, -1.0, -0.5), (160, -1.5, -1.0), (240, -2.0, -1.5)]
def logstd_at(it, tbl=None):
    tbl = tbl if tbl is not None else LOGSTD_P3
    return next((la, lo) for th, la, lo in reversed(tbl) if it >= th)

def run_phase(actor, critic, opt_a, opt_c, scaler, N_iter, label, lr_scale=1.0, use_polar=False,
              explore_boost=False, phase3=False, actor_frozen=False, critic_frozen=False, best_c_ref=None,
              logstd_sched=None, full_state=False):
    if lr_scale != 1.0 or explore_boost: opt_a, opt_c = make_opts(actor, critic, lr_scale)
    print(f'\n{"="*60}\n{label}\n{"="*60}')
    if explore_boost:
        with torch.no_grad(): actor.log_std.copy_(torch.tensor([0.5, 1.5], device=DEV))
    best_r = best_c_ref if best_c_ref is not None else 0.0
    for it in range(N_iter):
        if phase3:
            ls0, ls1 = logstd_at(it, logstd_sched)
            with torch.no_grad(): actor.log_std.copy_(torch.tensor([ls0, ls1], device=DEV))
        a_avg, c_avg, r_avg, rs = do_iteration(actor, critic, opt_a, opt_c, scaler, use_polar=use_polar,
                                               actor_frozen=actor_frozen, critic_frozen=critic_frozen,
                                               full_state=full_state)
        if rs >= best_r: best_r = rs
        if it % 25 == 0 or it < 3 or it == N_iter-1:
            print(f'  it{it:4d}: a={a_avg:.4f} c={c_avg:.4f} r={r_avg:.4f}  rs={rs:.3f}  best={best_r:.3f}')
        elif it % 10 == 0:
            print(f'  it{it:4d}: a={a_avg:.4f} c={c_avg:.4f} r={r_avg:.4f}  rs={rs:.3f}')
    return best_r, opt_a, opt_c

# ═══════════════════ Phase B: frozen base + 100% SelWP → blank 网络 ═══════════════════
def run_p2_bc(frozen_actor, blank_actor, blank_critic, n_iter, near_weight=True, full_state=True):
    """单次 rollout → 就地训练两个 blank 网络. 数据不跨轮积累.

    - frozen_actor: 仅 rollout (eval, log_std=[-2.0,-1.5] 微噪声)
    - SelWP: 100% 条件触发 (in_circle & R_min<0.5 & 每 phase ≤1 次)
    - blank_actor: BC (目标 a_eff, 含 WP 段反打动作)
      near_weight: 按当前目标距离加权 w=1/(d1+0.15) — 反打演示集中在
      d1 小的样本 (WP 触发区), 提升其 BC 权重, 防止 MSE 平均掉反打
      (与 Phase D 距离加权同款, 按批归一)
    - blank_critic: PPO clipped value loss (rollout 中采值, GAE 含 WP 段)
    """
    frozen_actor.eval(); blank_actor.train(); blank_critic.train()
    with torch.no_grad(): frozen_actor.log_std.copy_(torch.tensor([-2.0, -1.5], device=DEV))
    opt_bc = torch.optim.AdamW(blank_actor.parameters(), lr=1e-3, weight_decay=1e-5)
    opt_cr = torch.optim.Adam(blank_critic.parameters(), lr=5e-5)
    scaler_p2 = torch.amp.GradScaler('cuda')
    best_r = 0.0
    for it in range(n_iter):
        sp, g1, g2, g3 = gen_scenes_polar(E, DEV, full_state); B = E
        targets_t = torch.stack([g1, g2, g3], dim=1)
        phase = torch.zeros(B, dtype=torch.int32, device=DEV)
        alive = torch.ones(B, dtype=torch.bool, device=DEV)
        all_a_eff, all_o1, all_o2, all_o3, all_v2, all_v3 = [], [], [], [], [], []
        all_rew, all_val, all_alive, all_done = [], [], [], []
        wp_pos = torch.zeros(B, 2, device=DEV)
        wp_triggered = torch.zeros(B, 3, dtype=torch.bool, device=DEV)
        wp_dead = torch.ones(B, dtype=torch.bool, device=DEV)
        for _ in range(STEPS):
            if not alive.any(): break
            i0 = torch.clamp(phase.long(), 0, 2); i1 = torch.clamp(phase.long()+1, 0, 2); i2 = torch.clamp(phase.long()+2, 0, 2)
            cg = targets_t[torch.arange(B, device=DEV), i0]
            ng = targets_t[torch.arange(B, device=DEV), i1]
            nng = targets_t[torch.arange(B, device=DEV), i2]
            v2 = (phase < 2).float(); v3 = (phase < 1).float()
            o1, o2, o3 = polar_obs(sp, cg, ng, nng)          # 观测永远真实目标
            # SelWP: 100% 条件触发 (速度门 R_min<0.5)
            in_circle, dx_b, dy_b, R_min = unreachable_mask(sp, cg)
            ph_c = torch.clamp(phase, 0, 2)
            trigger_now = in_circle & (R_min < 0.5) & ~wp_triggered[torch.arange(B, device=DEV), ph_c] & alive
            if trigger_now.any():
                wp_new = compute_wp_world(sp[trigger_now], dx_b[trigger_now], dy_b[trigger_now], R_min[trigger_now])
                wp_pos[trigger_now] = wp_new; wp_dead[trigger_now] = False
                wp_triggered[trigger_now, ph_c[trigger_now]] = True
            wp_mask = ~wp_dead & alive
            # 动作: 正常段采样 / WP 段确定性
            act_all = torch.zeros(B, 2, device=DEV)
            normal_mask = alive & ~wp_mask
            if normal_mask.any():
                with torch.no_grad(): act_n, _, _ = frozen_actor.sample(o1[normal_mask], o2[normal_mask], o3[normal_mask], v2[normal_mask], v3[normal_mask])
                act_all[normal_mask] = act_n
            if wp_mask.any():
                o1_wp, o2_wp, o3_wp = polar_obs(sp[wp_mask], wp_pos[wp_mask], cg[wp_mask], ng[wp_mask])
                # 门: WP→cg 段恒 1 (episode 存活 ⇒ 原第一目标 cg 必存在); cg→ng 段用原 v2 门
                with torch.no_grad(): mean_wp, _ = frozen_actor.forward(o1_wp, o2_wp, o3_wp, torch.ones_like(v2[wp_mask]), v2[wp_mask])
                act_all[wp_mask] = mean_wp
            sn = SIM.simulate(sp, act_all)
            a_eff = torch.stack([(sn[:, 3]-sp[:, 3])/DT, (sn[:, 4]-sp[:, 4])/DT], dim=1)
            atg = check_hit_batch(sp[:, :2], sn[:, :2], cg, TOL)
            if wp_mask.any(): wp_dead = wp_dead | (check_hit_batch(sp[:, :2], sn[:, :2], wp_pos, TOL) & wp_mask)
            d_curr = torch.hypot(sp[:, 0]-cg[:, 0], sp[:, 1]-cg[:, 1]); d_next = torch.hypot(sp[:, 0]-ng[:, 0], sp[:, 1]-ng[:, 1])
            d_curr_new = torch.hypot(sn[:, 0]-cg[:, 0], sn[:, 1]-cg[:, 1]); d_next_new = torch.hypot(sn[:, 0]-ng[:, 0], sn[:, 1]-ng[:, 1])
            cost = DT + 0.002*(act_all[:, 0]/A_MAX)**2 + 0.01*(act_all[:, 1]/O_MAX)**2 + 0.002*torch.abs(act_all[:, 1])/O_MAX
            reward = -cost + NSR*(d_curr - GAMMA*d_curr_new + d_next - GAMMA*d_next_new)
            with torch.no_grad(): val = blank_critic.forward(o1, o2, o3, v2, v3)
            all_o1.append(o1); all_o2.append(o2); all_o3.append(o3); all_v2.append(v2); all_v3.append(v3)
            all_rew.append(reward); all_val.append(val); all_alive.append(alive.clone()); all_a_eff.append(a_eff)
            all_done.append(atg & (phase >= 2))
            wp_dead = wp_dead | (atg & alive)
            phase = torch.where(atg & alive, phase+1, phase)
            alive = alive & (phase < 3)
            sp = sn
        suc = (phase >= 3).float().mean().item()
        # GAE (blank_critic 采值, 含 WP 段)
        i0 = torch.clamp(phase.long(), 0, 2); i1 = torch.clamp(phase.long()+1, 0, 2); i2 = torch.clamp(phase.long()+2, 0, 2)
        o1_f, o2_f, o3_f = polar_obs(sp, targets_t[torch.arange(B, device=DEV), i0],
                                      targets_t[torch.arange(B, device=DEV), i1],
                                      targets_t[torch.arange(B, device=DEV), i2])
        with torch.no_grad(): last_val_full = blank_critic(o1_f, o2_f, o3_f, (phase < 2).float(), (phase < 1).float())
        all_val.append(last_val_full)
        T = len(all_rew); gae_val = torch.zeros(B, device=DEV); returns = []
        for t_idx in reversed(range(T)):
            next_val = all_val[t_idx+1].clone()
            next_val[all_done[t_idx]] = 0.0      # 终止步 bootstrap → 0 (与 do_iteration 一致)
            next_val[~all_alive[t_idx]] = 0.0
            reward_t = all_rew[t_idx].clone(); reward_t[~all_alive[t_idx]] = 0.0
            val_t = all_val[t_idx].clone(); val_t[~all_alive[t_idx]] = 0.0
            delta = reward_t + GAMMA*next_val - val_t
            gae_val = delta + GAMMA*LAM*gae_val; gae_val[~all_alive[t_idx]] = 0.0
            returns.insert(0, (gae_val[all_alive[t_idx]] + val_t[all_alive[t_idx]]).clone())
        A_EFF = torch.cat([x[all_alive[i]] for i, x in enumerate(all_a_eff)], dim=0)
        O1 = torch.cat([x[all_alive[i]] for i, x in enumerate(all_o1)], dim=0)
        O2 = torch.cat([x[all_alive[i]] for i, x in enumerate(all_o2)], dim=0)
        O3 = torch.cat([x[all_alive[i]] for i, x in enumerate(all_o3)], dim=0)
        V2 = torch.cat([x[all_alive[i]] for i, x in enumerate(all_v2)], dim=0)
        V3 = torch.cat([x[all_alive[i]] for i, x in enumerate(all_v3)], dim=0)
        RET = torch.cat([r.unsqueeze(1) for r in returns], dim=0)
        total_data = A_EFF.shape[0]
        if total_data >= BS:
            bs_bc = 512; perm = torch.randperm(total_data, device=DEV)
            for s0 in range(0, total_data, bs_bc):
                bi = perm[s0:s0+bs_bc]
                pred_a, _ = blank_actor.forward(O1[bi], O2[bi], O3[bi], V2[bi], V3[bi])
                if near_weight:
                    d1 = O1[bi][:, 2] * 5.0            # O1[:,2] = d1/5.0 → 真实距离
                    w = 1. / (d1 + 0.15)
                    w = w / w.mean().clamp(min=0.1)
                    l_a = (F.mse_loss(pred_a, A_EFF[bi], reduction='none').mean(dim=1) * w).mean()
                else:
                    l_a = F.mse_loss(pred_a, A_EFF[bi])
                opt_bc.zero_grad(); l_a.backward(); opt_bc.step()
                with torch.amp.autocast('cuda'):
                    v_pred = blank_critic.forward(O1[bi], O2[bi], O3[bi], V2[bi], V3[bi])
                    rb = RET[bi].squeeze(-1)
                    v_clip = rb + torch.clamp(v_pred - rb, -VC, VC)
                    l_c = MG*torch.max((v_pred - rb).pow(2), (v_clip - rb).pow(2)).mean()
                opt_cr.zero_grad(); scaler_p2.scale(l_c).backward(); scaler_p2.unscale_(opt_cr)
                torch.nn.utils.clip_grad_norm_(blank_critic.parameters(), 0.5)
                scaler_p2.step(opt_cr); scaler_p2.update()
        if (it+1) % 50 == 0 or it == 0: print(f'  P2 {it+1:4d}/{n_iter}: suc={suc:.3f}')
        if suc > best_r: best_r = suc
    return best_r

# ═══════════════════ 评估 (GPU 批量并行) ═══════════════════
MAX_EVAL_STEPS = 600   # 30s 仿真上限 (实测场景 <20s)

def eval_batch(actor, use_gp, scenes):
    """scenes: [(s0, tgts), ...] — 全部环境并行评估, 无逐步同步.

    返回 (succ[B], times[B]).
    """
    B = len(scenes)
    N = max(len(sc[1]) for sc in scenes)
    s = torch.stack([torch.tensor(sc[0], device=DEV, dtype=torch.float32) for sc in scenes])
    tgts = torch.zeros(B, N, 2, device=DEV)
    n_tgt = torch.zeros(B, dtype=torch.int64, device=DEV)
    for i, sc in enumerate(scenes):
        for j, t in enumerate(sc[1]):
            tgts[i, j] = torch.tensor(t, device=DEV, dtype=torch.float32)
        n_tgt[i] = len(sc[1])
    gi = torch.zeros(B, dtype=torch.int64, device=DEV)
    steps = torch.zeros(B, dtype=torch.int64, device=DEV)
    prev = s[:, :2].clone()
    done = gi >= n_tgt
    with torch.no_grad():
        for _ in range(MAX_EVAL_STEPS):
            if done.all(): break
            i0 = torch.clamp(gi, 0, N-1); i1 = torch.clamp(gi+1, 0, N-1); i2 = torch.clamp(gi+2, 0, N-1)
            cur = tgts[torch.arange(B, device=DEV), i0]
            nxt = tgts[torch.arange(B, device=DEV), i1]
            nnxt = tgts[torch.arange(B, device=DEV), i2]
            v2 = (gi < n_tgt-1).float(); v3 = (gi < n_tgt-2).float()
            if use_gp:
                act = actor.deploy(polar_8d(s, cur, nxt, nnxt), v2, v3)
            else:
                o1, o2, o3 = polar_obs(s, cur, nxt, nnxt)
                act, _ = actor.forward(o1, o2, o3, v2, v3)
            sn = SIM.simulate(s, act)
            hit = check_hit_batch(prev, sn[:, :2], cur, TOL) & ~done
            gi = torch.where(hit, gi+1, gi)
            steps = torch.where(done, steps, steps+1)
            prev = s[:, :2].clone()
            s = sn
            done = gi >= n_tgt
    return (gi >= n_tgt).cpu().numpy(), steps.cpu().numpy() * DT

def eval_model(actor, use_gp, n_scene_trials=6):
    """4 标准场景 + random 100."""
    scenes = [
        ('tight-180', np.array([0., 0., np.pi/4, 0., 0.], dtype=np.float32),
         [np.array([CAR_X+1.5, CAR_Y+1.5]), np.array([CAR_X+0.5, CAR_Y+3.0]), np.array([CAR_X-1.0, CAR_Y+1.5])]),
        ('tight-TR', np.array([0., 0., 0., 0., 0.], dtype=np.float32),
         [np.array([CAR_X+2.0, CAR_Y+0.0]), np.array([CAR_X+2.5, CAR_Y+2.5]), np.array([CAR_X+0.5, CAR_Y+2.0])]),
        ('anti-steer', np.array([CAR_X, CAR_Y, 0., 2.0, 0.], dtype=np.float32),
         [np.array([CAR_X-1.0, CAR_Y+1.5]), np.array([CAR_X+1.0, CAR_Y+4.0]), np.array([CAR_X+3.0, CAR_Y+2.0])]),
        ('straight', np.array([CAR_X, CAR_Y, 0., 0., 0.], dtype=np.float32),
         [np.array([CAR_X+2.0, CAR_Y+0.1]), np.array([CAR_X+4.0, CAR_Y-0.1]), np.array([CAR_X+6.0, CAR_Y+0.0])]),
    ]
    print(f'{"scene":>14s}  {"succ":>6s}  {"time":>6s}')
    print('-'*30)
    for name, s0, tg in scenes:
        succ, times = eval_batch(actor, use_gp, [(s0, tg)] * n_scene_trials)
        s_ok = int(succ.sum())
        ok_times = times[succ]
        ti = f'{ok_times.mean():.1f}s' if len(ok_times) else '-'
        print(f'{name:>14s}  {s_ok}/{n_scene_trials}   {ti}')
    scenes_100 = []
    for i in range(100):
        rng_i = np.random.RandomState(i)
        s0 = np.array([rng_i.uniform(M, FLD-M), rng_i.uniform(M, FLD-M), rng_i.uniform(-np.pi, np.pi),
                       rng_i.uniform(0.5, V_MAX), rng_i.uniform(-DELTA_MAX, DELTA_MAX)], dtype=np.float32)
        tg = [np.array([rng_i.uniform(M, FLD-M), rng_i.uniform(M, FLD-M)], dtype=np.float32) for _ in range(3)]
        scenes_100.append((s0, tg))
    succ, times = eval_batch(actor, use_gp, scenes_100)
    s_ok = int(succ.sum()); ok_times = times[succ]
    tm = f'{ok_times.mean():.1f}s' if len(ok_times) else '-'
    print(f'{"random 100":>14s}  {s_ok/100:.2f}  {tm}')
    return s_ok / 100

def set_teacher_deterministic(actor):
    with torch.no_grad(): actor.log_std.copy_(torch.tensor([-2.0, -1.5], device=DEV))

# ═══════════════════ 反打内化探针 ═══════════════════
def rollout_traj(actor, use_gp, s0, tgts, max_steps=MAX_EVAL_STEPS):
    """单环境回放, 返回 (traj[N,5], gi). 探针/可视化共用."""
    s = torch.tensor(s0, device=DEV, dtype=torch.float32).unsqueeze(0)
    gi = 0; traj = [np.asarray(s0, dtype=np.float64)]
    for _ in range(max_steps):
        if gi >= len(tgts): break
        cur = torch.tensor(tgts[min(gi, len(tgts)-1)], device=DEV, dtype=torch.float32).unsqueeze(0)
        nxt = torch.tensor(tgts[min(gi+1, len(tgts)-1)], device=DEV, dtype=torch.float32).unsqueeze(0)
        nnxt = torch.tensor(tgts[min(gi+2, len(tgts)-1)], device=DEV, dtype=torch.float32).unsqueeze(0)
        v2 = torch.tensor([float(gi < len(tgts)-1)], device=DEV)
        v3 = torch.tensor([float(gi < len(tgts)-2)], device=DEV)
        with torch.no_grad():
            if use_gp:
                act = actor.deploy(polar_8d(s, cur, nxt, nnxt), v2, v3)
            else:
                o1, o2, o3 = polar_obs(s, cur, nxt, nnxt)
                act, _ = actor.forward(o1, o2, o3, v2, v3)
        sn = SIM.simulate(s, act)
        snp = sn[0].cpu().numpy()
        if check_hit_substep_np(traj[-1][:2], snp[:2], tgts[gi], TOL): gi += 1
        traj.append(snp); s = sn
    return np.array(traj), gi

def gen_trigger_scene(rng):
    """车低速 + 首目标落在不可达圆内 (SelWP 触发条件), 后两目标随机可达."""
    th = rng.uniform(-np.pi, np.pi)
    v = rng.uniform(0.6, 1.15)                    # < sqrt(0.5*A_LAT) 满足速度门
    R = v**2 / A_LAT
    sign = 1. if rng.rand() < 0.5 else -1.
    a = rng.uniform(0, 2*np.pi); r0 = rng.uniform(0.3, 0.9) * R
    dx_b = r0 * np.cos(a); dy_b = -sign * R + r0 * np.sin(a)
    c, s = np.cos(th), np.sin(th)
    g1 = np.array([CAR_X + dx_b*c - dy_b*s, CAR_Y + dx_b*s + dy_b*c])
    g2 = np.array([rng.uniform(M, FLD-M), rng.uniform(M, FLD-M)])
    g3 = np.array([rng.uniform(M, FLD-M), rng.uniform(M, FLD-M)])
    s0 = np.array([CAR_X, CAR_Y, th, v, 0.0], dtype=np.float32)
    return s0, [g1, g2, g3]

def probe_countersteer(actor, use_gp, n_scenes=40):
    """反打内化探针: 触发场景成功率 + 反打步数.

    反打判定: |δ|>0.1 且 |θ̇|>0.15 且 sign(δ)≠sign(θ̇) — 车体朝一边转、轮子朝另一边打.
    """
    scenes = [gen_trigger_scene(np.random.RandomState(3000+i)) for i in range(n_scenes)]
    succ, times = eval_batch(actor, use_gp, scenes)
    cs_total = 0; n_steps = 0
    for s0, tg in scenes[:12]:
        traj, gi = rollout_traj(actor, use_gp, s0, tg)
        if len(traj) < 3: continue
        dth = np.diff(traj[:, 2]) / DT
        dth = np.arctan2(np.sin(dth), np.cos(dth))   # 角度环跳安全化
        delta = traj[:-1, 4]
        cs = (np.abs(delta) > 0.1) & (np.abs(dth) > 0.15) & (np.sign(delta) != np.sign(dth))
        cs_total += int(cs.sum()); n_steps += len(delta)
    pct = cs_total / max(1, n_steps) * 100
    print(f'  [counter-steer probe] succ={int(succ.sum())}/{n_scenes}  '
          f'cs_steps={cs_total} ({pct:.1f}% of steps)  mean_time={times.mean():.1f}s')
    return int(succ.sum()), cs_total

# ═══════════════════ Phase D: 蒸馏 ═══════════════════
def collect_bc(teacher, noise_std=0.03):
    all_pol, all_v2, all_v3, all_tgt = [], [], [], []
    for _ in range(20):
        sp, g1, g2, g3 = gen_mixed(64, DEV); B = sp.shape[0]
        tgts = torch.stack([g1, g2, g3], dim=1)
        phase = torch.zeros(B, dtype=torch.int32, device=DEV)
        alive = torch.ones(B, dtype=torch.bool, device=DEV)
        for _ in range(500):
            if not alive.any(): break
            i0 = torch.clamp(phase.long(), 0, 2); i1 = torch.clamp(phase.long()+1, 0, 2); i2 = torch.clamp(phase.long()+2, 0, 2)
            cg = tgts[torch.arange(B, device=DEV), i0]; ng = tgts[torch.arange(B, device=DEV), i1]
            nng = tgts[torch.arange(B, device=DEV), i2]
            o1, o2, o3 = polar_obs(sp, cg, ng, nng)
            v2 = (phase < 2).float(); v3 = (phase < 1).float()
            with torch.no_grad(): t_act, _ = teacher.forward(o1, o2, o3, v2, v3)
            if noise_std > 0:
                t_act = t_act + torch.randn(B, 2, device=DEV) * noise_std
                t_act[:, 0].clamp_(-A_BRAKE, A_LONG); t_act[:, 1].clamp_(-O_MAX, O_MAX)
            t_sn = SIM.simulate(sp, t_act)
            a_eff = torch.stack([(t_sn[:, 3]-sp[:, 3])/DT, (t_sn[:, 4]-sp[:, 4])/DT], dim=1)
            # GPU 累积, 结束后一次性转出 (零逐步同步)
            all_pol.append(polar_8d(sp, cg, ng, nng)[alive]); all_v2.append(v2[alive])
            all_v3.append(v3[alive]); all_tgt.append(a_eff[alive])
            sn = t_sn
            ax, ay = sp[:, 0], sp[:, 1]; bx, by = sn[:, 0], sn[:, 1]; px, py = cg[:, 0], cg[:, 1]
            abx, aby = bx-ax, by-ay; ab2 = abx*abx + aby*aby + 1e-12
            t = torch.clamp(((px-ax)*abx + (py-ay)*aby)/ab2, 0., 1.)
            cx, cy = ax + t*abx, ay + t*aby
            atg = (torch.hypot(bx-px, by-py) < TOL) | ((px-cx)**2 + (py-cy)**2 < TOL**2)
            phase = torch.where(atg & alive, phase+1, phase)
            alive = alive & ~(atg & (phase >= 3))
            sp = sn
    # 单次同步: GPU cat 后一次性转出
    return (torch.cat(all_pol, dim=0).cpu(), torch.cat(all_v2, dim=0).cpu(),
            torch.cat(all_v3, dim=0).cpu(), torch.cat(all_tgt, dim=0).cpu())

def collect_dagger(teacher, student):
    all_pol, all_v2, all_v3, all_tgt = [], [], [], []
    for _ in range(20):
        sp, g1, g2, g3 = gen_mixed(64, DEV); B = sp.shape[0]
        tgts = torch.stack([g1, g2, g3], dim=1)
        phase = torch.zeros(B, dtype=torch.int32, device=DEV)
        alive = torch.ones(B, dtype=torch.bool, device=DEV)
        for _ in range(500):
            if not alive.any(): break
            i0 = torch.clamp(phase.long(), 0, 2); i1 = torch.clamp(phase.long()+1, 0, 2); i2 = torch.clamp(phase.long()+2, 0, 2)
            cg = tgts[torch.arange(B, device=DEV), i0]; ng = tgts[torch.arange(B, device=DEV), i1]
            nng = tgts[torch.arange(B, device=DEV), i2]
            p = polar_8d(sp, cg, ng, nng)
            v2 = (phase < 2).float(); v3 = (phase < 1).float()
            with torch.no_grad(): s_act = student.deploy(p, v2, v3)
            sn = SIM.simulate(sp, s_act)
            o1, o2, o3 = polar_obs(sp, cg, ng, nng)
            with torch.no_grad(): t_act, _ = teacher.forward(o1, o2, o3, v2, v3)
            t_sn = SIM.simulate(sp, t_act)
            a_eff = torch.stack([(t_sn[:, 3]-sp[:, 3])/DT, (t_sn[:, 4]-sp[:, 4])/DT], dim=1)
            # GPU 累积, 结束后一次性转出 (零逐步同步)
            all_pol.append(p); all_v2.append(v2); all_v3.append(v3); all_tgt.append(a_eff)
            ax, ay = sp[:, 0], sp[:, 1]; bx, by = sn[:, 0], sn[:, 1]; px, py = cg[:, 0], cg[:, 1]
            abx, aby = bx-ax, by-ay; ab2 = abx*abx + aby*aby + 1e-12
            t = torch.clamp(((px-ax)*abx + (py-ay)*aby)/ab2, 0., 1.)
            cx, cy = ax + t*abx, ay + t*aby
            atg = (torch.hypot(bx-px, by-py) < TOL) | ((px-cx)**2 + (py-cy)**2 < TOL**2)
            phase = torch.where(atg & alive, phase+1, phase)
            alive = alive & ~(atg & (phase >= 3))
            sp = sn
    # 单次同步: GPU cat 后一次性转出
    return (torch.cat(all_pol, dim=0).cpu(), torch.cat(all_v2, dim=0).cpu(),
            torch.cat(all_v3, dim=0).cpu(), torch.cat(all_tgt, dim=0).cpu())

def train_epochs(student, X_pol, V2, V3, Y, n_epochs, lr):
    X_t = X_pol.to(DEV); V2_t = V2.to(DEV); V3_t = V3.to(DEV); Y_t = Y.to(DEV)
    N = X_t.shape[0]; bs = 512
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    for ep in range(n_epochs):
        perm = torch.randperm(N, device=DEV); tt = torch.tensor(0., device=DEV); n = 0
        for s0 in range(0, N, bs):
            idx = perm[s0:s0+bs]; xb, yb = X_t[idx], Y_t[idx]; v2b, v3b = V2_t[idx], V3_t[idx]
            pred = student.forward(xb, v2b, v3b)
            d = torch.sqrt((xb[:, 2]*D_SCALE)**2 + 1e-8)      # 真实距离 d1 (D_SCALE 归一化)
            w = 1./(d + 0.15); w = w / w.mean().clamp(min=0.1)   # 距离加权: 近点权重大
            loss = (F.mse_loss(pred, yb, reduction='none').mean(dim=1) * w).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tt += loss.detach(); n += 1
        sched.step()
        if ep == 0 or (ep+1) % 50 == 0: print(f'    ep {ep+1:3d}: loss={tt/n:.6f}')

def calc_mean_time(student, seed_off=500, n_random=100):
    """选优指标: 3 个 tight 场景 × 4 trials + n_random 随机场景 (种子 seed_off 起).

    返回 (tight_mean_time, tight_succ, random_succ).
    失败 trial 计 MAX_EVAL_STEPS*DT (30s) 软惩罚.
    随机场景种子与最终评估 (0-99) 错开, 避免对评估集过拟合.
    """
    scenes = [
        (np.array([0., 0., np.pi/4, 0., 0.], dtype=np.float32),
         [np.array([CAR_X+1.5, CAR_Y+1.5]), np.array([CAR_X+0.5, CAR_Y+3.0]), np.array([CAR_X-1.0, CAR_Y+1.5])]),
        (np.array([0., 0., 0., 0., 0.], dtype=np.float32),
         [np.array([CAR_X+2.0, CAR_Y+0.0]), np.array([CAR_X+2.5, CAR_Y+2.5]), np.array([CAR_X+0.5, CAR_Y+2.0])]),
        (np.array([CAR_X, CAR_Y, 0., 2.0, 0.], dtype=np.float32),
         [np.array([CAR_X-1.0, CAR_Y+1.5]), np.array([CAR_X+1.0, CAR_Y+4.0]), np.array([CAR_X+3.0, CAR_Y+2.0])]),
    ]
    trials = [(s0, tg) for s0, tg in scenes for _ in range(4)]
    succ_t, times_t = eval_batch(student, True, trials)
    scenes_r = []
    for i in range(n_random):
        rng = np.random.RandomState(seed_off + i)
        s0 = np.array([rng.uniform(M, FLD-M), rng.uniform(M, FLD-M), rng.uniform(-np.pi, np.pi),
                       rng.uniform(0.5, V_MAX), rng.uniform(-DELTA_MAX, DELTA_MAX)], dtype=np.float32)
        tg = [np.array([rng.uniform(M, FLD-M), rng.uniform(M, FLD-M)], dtype=np.float32) for _ in range(3)]
        scenes_r.append((s0, tg))
    succ_r, _ = eval_batch(student, True, scenes_r)
    return float(times_t.mean()), float(succ_t.mean()), float(succ_r.mean())

# ═══════════════════ checkpoint ═══════════════════
def ckpt_meta(best):
    return {'best': best, 'a_long_max': A_LONG, 'a_brake_max': A_BRAKE, 'a_lat_max': A_LAT,
            'v_max': V_MAX, 'omega_max': O_MAX, 'delta_max': DELTA_MAX}

def save_ckpt(path, actor, critic, best):
    torch.save({'actor_state_dict': actor.state_dict(), 'critic_state_dict': critic.state_dict(), **ckpt_meta(best)}, path)

def load_ckpt(path, actor, critic):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    actor.load_state_dict(ck['actor_state_dict']); critic.load_state_dict(ck['critic_state_dict'])
    return ck.get('best', 0.0)

# ═══════════════════ Main ═══════════════════
def phase_a():
    """TOL 课程 → transition → polar 探索 → anneal.

    P1a 窄初始化 (v≤2.5, δ≤0.3δmax) 热身; P1b 起全状态初始.
    """
    actor = GatedConcatActor().to(DEV); critic = GatedConcatCritic().to(DEV)
    scaler = torch.amp.GradScaler('cuda')
    opt_a, opt_c = make_opts(actor, critic, 1.0)
    best_c = 0.0

    print('\n=== P1a: TOL curriculum (300 iter, random) ===')
    with torch.no_grad(): actor.log_std.copy_(torch.tensor([1.0, 2.0], device=DEV))
    TOL_SCHED = [(0, 2.0), (60, 1.0), (120, 0.5), (180, 0.25), (240, 0.10)]
    def tol_at(it): return next(v for th, v in reversed(TOL_SCHED) if it >= th)
    prev_tol = None
    for it in range(300):
        tp = tol_at(it)
        if prev_tol is not None and tp != prev_tol:
            ls = {1.0: [0.6, 1.6], 0.5: [0.5, 1.5], 0.25: [0.5, 1.5], 0.10: [0.5, 1.5]}.get(tp, [0.8, 1.8])
            with torch.no_grad(): actor.log_std.copy_(torch.tensor(ls, device=DEV))
            opt_a, opt_c = make_opts(actor, critic, 1.0)
        prev_tol = tp
        a_avg, c_avg, r_avg, rs = do_iteration(actor, critic, opt_a, opt_c, scaler, use_polar=False, tol_val=tp)
        if rs >= best_c: best_c = rs
        if it % 25 == 0 or it < 3 or it == 299:
            print(f'  it{it:4d} tol={tp:.2f} a={a_avg:.4f} c={c_avg:.4f} r={r_avg:.4f} rs={rs:.3f} best={best_c:.3f}')

    best_c, opt_a, opt_c = run_phase(actor, critic, opt_a, opt_c, scaler, 100, 'P1b: full-state transition (random)', explore_boost=True, use_polar=False, best_c_ref=best_c, full_state=True)
    best_c, opt_a, opt_c = run_phase(actor, critic, opt_a, opt_c, scaler, 300, 'P1c: explore (polar)', explore_boost=True, use_polar=True, best_c_ref=best_c, full_state=True)
    best_c, opt_a, opt_c = run_phase(actor, critic, opt_a, opt_c, scaler, 300, 'P1d: anneal (polar)', lr_scale=0.4, phase3=True, use_polar=True, best_c_ref=best_c, full_state=True)

    save_ckpt('runs/p1_base.pt', actor, critic, best_c)
    print(f'  Saved: runs/p1_base.pt (best={best_c:.3f})')
    print('\n=== P1 eval ===')
    set_teacher_deterministic(actor)
    eval_model(actor, False)
    probe_countersteer(actor, False)
    return actor, critic

def phase_b(base_actor, base_critic=None):
    """frozen base + 100% SelWP → blank actor BC + blank critic PPO.

    base_critic 占位, P1 critic 不参与 P2 (base 仅用于 rollout).
    """
    print('\n=== P2: Frozen base + 100% SelWP → blank nets (300 iter) ===')
    blank_actor = GatedConcatActor().to(DEV)
    blank_critic = GatedConcatCritic().to(DEV)
    best_p2 = run_p2_bc(base_actor, blank_actor, blank_critic, 300)
    save_ckpt('runs/p2_bc.pt', blank_actor, blank_critic, best_p2)
    print(f'  Saved: runs/p2_bc.pt (best={best_p2:.3f})')
    print('\n=== P2 eval (SelWP 教师) ===')
    set_teacher_deterministic(blank_actor)
    eval_model(blank_actor, False)
    probe_countersteer(blank_actor, False)
    return blank_actor, blank_critic

def phase_c(actor, critic):
    """续训 (替代多次微调): polar/random 交替多循环. 无 SelWP.

    历史经验: 单轮 straight-line 会牺牲 polar 能力, 需 polar refine 回补;
    交替多轮才能把 99% 推满 100%.
    初始状态全幅 (v≤V_MAX, δ≤δmax) — 冷启动之后不再限制.
    """
    scaler = torch.amp.GradScaler('cuda')
    actor.train(); critic.train()
    best_c = 0.0

    # C1: 噪声破死锁 (联合训练 — 实验证明冻结 critic 损害最终质量)
    with torch.no_grad(): actor.log_std.copy_(torch.tensor([-0.7, -0.3], device=DEV))
    opt_a, opt_c = make_opts(actor, critic, 1.0)
    best_c, opt_a, opt_c = run_phase(actor, critic, opt_a, opt_c, scaler, 200,
                                     'C1: noise break-deadlocks (polar, [-0.7,-0.3])',
                                     use_polar=True, best_c_ref=0.0, full_state=True)

    # C2-C4: random/polar 交替 (效率 → 再适应 → 二轮效率)
    for N_iter, use_polar, label in [
        (200, False, 'C2: random straight-line'),
        (200, True,  'C3: polar refine'),
        (150, False, 'C4: random straight-line 2'),
    ]:
        with torch.no_grad(): actor.log_std.copy_(torch.tensor([-1.0, -0.5], device=DEV))
        opt_a, opt_c = make_opts(actor, critic, 1.0)
        best_c, opt_a, opt_c = run_phase(actor, critic, opt_a, opt_c, scaler, N_iter, label, use_polar=use_polar, best_c_ref=best_c, full_state=True)

    # C5: 最终 anneal (250 iter) — 从 C4 结束的 [-1.0,-0.5] 继续收紧 (无噪声反弹),
    # 240 步后收敛到 [-2.0,-1.5] (a5b3l3/sym3 的继续式 anneal 约定)
    LOGSTD_C5 = [(0, -1.0, -0.5), (80, -1.3, -0.8), (160, -1.6, -1.1), (240, -2.0, -1.5)]
    best_c, opt_a, opt_c = run_phase(actor, critic, opt_a, opt_c, scaler, 250, 'C5: final anneal (polar)', lr_scale=0.4, phase3=True, use_polar=True, best_c_ref=best_c, logstd_sched=LOGSTD_C5, full_state=True)

    save_ckpt('runs/kamm533_teacher.pt', actor, critic, best_c)
    print(f'\nSaved: runs/kamm533_teacher.pt (best={best_c:.3f})')
    print('\n=== C eval (最终教师) ===')
    set_teacher_deterministic(actor)
    eval_model(actor, False)
    probe_countersteer(actor, False)
    return actor, critic

def phase_d(teacher):
    """蒸馏: BC 距离加权 + DAgger×12 速度选优."""
    teacher.eval(); set_teacher_deterministic(teacher)
    student = GPSmall().to(DEV)
    n_params = sum(p.numel() for p in student.parameters())
    print(f'\n=== Phase D: distill → GP-Small ({n_params} params) ===')

    print('\n--- D1: BC (200 ep, distance-weighted) ---')
    X_pol, V2, V3, Y = collect_bc(teacher, noise_std=0.03)
    print(f'  {X_pol.shape[0]} samples')
    train_epochs(student, X_pol, V2, V3, Y, 200, 2e-3)
    t_bc, st_bc, sr_bc = calc_mean_time(student)
    print(f'  BC mean time: {t_bc:.2f}s  tight={st_bc:.2f}  rand={sr_bc:.2f}')
    torch.save({'model_state_dict': student.state_dict()}, 'runs/gp_small_kamm533_bc.pt')
    # BC 作为初始最优落盘: 若 DAgger 无一轮超越, 最终加载仍有效 (不再 FileNotFoundError)
    # 两级选优: 先比随机场景成功率 (防速度选优牺牲泛化), 再比 tight 场景速度
    best_time, best_rand, best_round = t_bc, sr_bc, 0
    torch.save({'model_state_dict': student.state_dict(), 'time': t_bc}, 'runs/gp_small_kamm533.pt')

    print('\n--- D2: DAgger ×12 (speed-selected, random-gated) ---')
    for dagger_it in range(1, 13):
        print(f'\n=== DAgger round {dagger_it} ===')
        Xn, V2n, V3n, Yn = collect_dagger(teacher, student)
        print(f'  Dataset: {Xn.shape[0]} samples')
        lr = 5e-4 if dagger_it <= 6 else 2e-4
        train_epochs(student, Xn, V2n, V3n, Yn, 100, lr)
        t, st, sr = calc_mean_time(student)
        print(f'  D{dagger_it} mean time: {t:.2f}s  tight={st:.2f}  rand={sr:.2f}')
        if sr > best_rand or (sr == best_rand and t < best_time):
            best_time, best_rand, best_round = t, sr, dagger_it
            torch.save({'model_state_dict': student.state_dict(), 'time': t}, 'runs/gp_small_kamm533.pt')
            print(f'  -> Best! Saved (D{dagger_it})')
        torch.save({'model_state_dict': student.state_dict(), 'time': t}, f'runs/gp_small_kamm533_d{dagger_it}.pt')

    best_src = f'D{best_round}' if best_round > 0 else 'BC'
    print(f'\nBest: {best_src} with {best_time:.2f}s (rand={best_rand:.2f}) -> runs/gp_small_kamm533.pt')
    print('\n=== Final eval: GP-Small vs Teacher ===')
    student.load_state_dict(torch.load('runs/gp_small_kamm533.pt', map_location=DEV, weights_only=False)['model_state_dict'])
    student.eval()
    print('--- GP-Small (部署) ---')
    eval_model(student, True)
    probe_countersteer(student, True)
    print('--- Teacher ---')
    eval_model(teacher, False)
    probe_countersteer(teacher, False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='stage', default='a', choices=['a', 'b', 'c', 'd'], help='起始阶段')
    args = ap.parse_args()

    print('='*60)
    print(f'Kamm 5/3/3 pipeline | {A_LONG}/{A_BRAKE}/{A_LAT} | A→B→C→D')
    print('='*60)

    actor = critic = None
    if args.stage == 'a':
        actor, critic = phase_a()
        args.stage = 'b'
    if args.stage == 'b':
        if actor is None:
            actor = GatedConcatActor().to(DEV); critic = GatedConcatCritic().to(DEV)
            load_ckpt('runs/p1_base.pt', actor, critic)
            print('Loaded runs/p1_base.pt')
        actor, critic = phase_b(actor, critic)
        args.stage = 'c'
    if args.stage == 'c':
        if actor is None:
            actor = GatedConcatActor().to(DEV); critic = GatedConcatCritic().to(DEV)
            load_ckpt('runs/p2_bc.pt', actor, critic)
            print('Loaded runs/p2_bc.pt')
        actor, critic = phase_c(actor, critic)
        args.stage = 'd'
    if args.stage == 'd':
        teacher = GatedConcatActor().to(DEV)
        if actor is None:
            ck = torch.load('runs/kamm533_teacher.pt', map_location=DEV, weights_only=False)
            teacher.load_state_dict(ck['actor_state_dict'])
            print('Loaded runs/kamm533_teacher.pt')
        else:
            teacher = actor
        phase_d(teacher)

    print('\nPipeline done.')

if __name__ == '__main__':
    main()
