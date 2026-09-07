# rl-pipeline

给 fisheye 航向感知阿克曼小车训导航策略的自包含 RL 管线。核心是四阶段教学:
冷启动训一个基础网络,再冻结它、用 Select 航点(SelWP, 训练期脚手架, 部署零依赖)示范,
让全新初始化的 actor/critic 从 rollout 里启动,最后纯 PPO 续训、蒸馏成小模型上 MCU。

```bash
python -u pipeline.py            # 四阶段全流程
python -u pipeline.py --from b   # 已有 p1_base.pt, 从 P2 续跑
```
