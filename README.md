# rl-pipeline — 航向感知阿克曼小车的四阶段 RL 教学管线

自包含训练管道(`pipeline.py` 单文件可复跑)。目标:从零训练一个能在 fisheye 航向感知阿克曼车(非对称摩擦 Kamm 自行车模型,`R_min = v²/A_LAT`)上导航到任意(可达)目标的策略,并蒸馏成 ~3.7K 参数的 MCU 可部署学生网络。

## 核心设计

把"往哪走"交给冷启动的 RL 基础网络,把"走不到时怎么办"交给训练期的临时脚手架 **SelWP**(Selective Waypoint)——**部署零依赖**:

- **WP 只存在于查询瞬间**:当真实目标落入不可达双圆(最小转弯圆内)时,临时把切点航点伪装成"第一目标"向教师查询反打动作,但存储的训练对永远是 **(真实目标观测, 动作)**,WP 坐标从不进入任何网络输入(action relabeling)。
- **教学 = 行为示范,不是参数**:P2 用全新初始化的空白 actor/critic,从冻结 P1 actor + 100% SelWP 的 rollout 中启动;rollout 里所有观测/价值/GAE 都记录在真实目标下。P1 的唯一痕迹是示范动作本身(教学信号),权重零继承、价值零蒸馏。
- **P3 纯 PPO 续训把 relabel 动作矫正为自洽策略**:反打能力经 critic 长视界价值内化,WP 脚手架退役。

## 四阶段

| 阶段 | 内容 | 产物 |
|---|---|---|
| A | 冷启动 TOL 课程(P1a 窄初始状态)→ 极坐标探索 → 退火到基础策略 | `p1_base.pt` |
| B | 冻结 P1 + 100% SelWP 示范 → 空白 actor/critic 从 rollout 启动(BC + GAE 值) | `p2_bc.pt` |
| C | 纯 PPO 续训(C1–C5 连续退火表,无 WP) | `kamm533_teacher.pt` |
| D | DAgger×12 BC 蒸馏到 GP-Small(3762 参数,两层随机门控选优) | `gp_small_kamm533.pt` |

场景采样为 3 带混合:**超近带** [0.05, 0.65]m @20%(不可达/反打构型)、半正态中段、**长距带** [3.0, 6.0]m @25%(提前减速);P1a 之外全部 full-state 初始化。P2 对近目标(`w = 1/(d+0.15)`)加权以对抗绕圈病态。

## 运行

```bash
python -u pipeline.py            # 全流程 A→D
python -u pipeline.py --from b   # 从 P2 续跑 (需要 p1_base.pt)
python smoke_test.py             # 快速冒烟
python eval_ckpt.py              # 评测随机/反打场景
python countersteer_probe.py     # 反打内化探针
```

产物按设计写入 `runs/`(已 gitignore);`backup_pre_fix/`、`backup_v2/` 为历史对照版本。

## 文件导览

- `pipeline.py` — 唯一训练文件(仿真/PPO/四阶段/蒸馏)
- `eval_ckpt.py` `countersteer_probe.py` `smoke_test.py` — 评测
- `bench_*.py` `viz*.py` `plot_*.py` `render_fix_3008.py` — 素材/可视化(生成物入 `runs/`)
- `ab_p2_weight.py` — 近目标加权的 A/B 实验

## 状态(2026-08-19 v3)

随机 100 场景成功率 1.00;反打探针 40/40(教师 cs 75 / 5.3s,GP 5.9s);N 目标 3–8 = 500/500;超近目标带修复消除绕圈(scene 3008: 24.5s → 6.1s)。
