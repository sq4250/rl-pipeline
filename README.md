# rl-pipeline

阿克曼小车的路径规划器。四阶段训练管线 (冷启动 → SelWP 示范 → 纯 PPO 续训 → 蒸馏),产出可上 MCU 的小网络。

## 文件

| 文件 | 作用 |
|---|---|
| `pipeline.py` | 训练管线,自包含(仅 torch + numpy) |
| `eval_viz.py` | 评测 + 轨迹可视化 (PNG/GIF) |
| `export_mcu.py` | 导出 MCU 用的 C 权重(归一化已折叠进第一层) |
| `runs/kamm533_student.pt` | 训好的学生模型,GP-Medium 8990 参数 |
| `runs/kamm533_teacher.pt` | 教师模型,76.7K 参数(供对比 / 重新蒸馏) |

## 用法

```bash
pip install -r requirements.txt      # 需 NVIDIA GPU + CUDA

python pipeline.py            # 重训全流程 → runs/kamm533_student.pt
python pipeline.py --from d   # 只重跑蒸馏 (需 runs/kamm533_teacher.pt)

python eval_viz.py            # 6 信标路线 GIF
python eval_viz.py --png               # 加输出轨迹/速度 PNG
python eval_viz.py --compare           # GIF 里学生+教师两辆车同时跑
python eval_viz.py --n-target 5 --seed 7   # 换个 5 目标场景
python export_mcu.py          # 导出 C 权重
```

`eval_viz.py --help` / `export_mcu.py --help` 有全部参数。

## 说明

- 场景采样: 三段腿增量 polar 链,每腿距离半正态 σ=1.5 截断 [0.05, 6.5];角度 侧前40%/侧后40%/正后20%。
- 蒸馏学生用 ReLU(MCU 无 GELU),导出 C 时归一化折叠进第一层权重。
- `export_mcu.py` 默认写到 `C:/Users/tsian/Desktop/beta_ackerman/project/code/core`;该目录不存在时回退到当前目录。
