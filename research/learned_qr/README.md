# Learned Q/R：可运行的 RTK 训练与部署

本目录与顶层 `src/rtkpos.c`、`src/rtklib.h` 已集成。**不用再运行 `tools/apply_patch.py`**；该脚本仅保留为最初原型的历史迁移工具及其保护测试。

## 入口

从仓库根目录运行 `python research/learned_qr/run_qr.py`。子命令：

| 子命令 | 功能 |
|---|---|
| `train` | 读取训练/验证路线，使用原始 RINEX 与参考位置 loss 训练 Q/R |
| `replay` | 加载原生权重，C 解算，保留配置中的整数固定；可同时输出评估 |
| `evaluate` | 将回放 CSV 与独立参考轨迹对齐并评估 |
| `convert-reference` | 将明确为 GPST、XYZ/LLH 的独立参考 `.pos` 转为监督 CSV |
| `export` | 从安全的张量 checkpoint 导出纯数值 C 模型 |

[完整使用说明](docs/TRAINING.md)含编译命令、数据清单、参考坐标格式、训练、回放、纯 C 示例和公平对比步骤。

## 结构

- `qrlearn/native.py`：ctypes 所有权管理，调用真实 C RINEX 处理；不把参考轨迹传给 C。
- `qrlearn/live.py`：原生实时事件的可微 FLOAT 递推，不读取固定的历史残差文件。
- `qrlearn/training.py`：按路线训练、验证、Q/R 交替优化、截断反向传播、权重保存。
- `qrlearn/reference.py`：显式 GPST/ECEF 数据、有效掩码、ENU 损失、评估。
- `qrlearn/model.py`：10 历元窗口的双分支时序 MLP；不是论文 Transformer。
- `src/learned_qr*.h`：有界方差倍率推理、因果特征、独立缓存、安全模型加载。
- `src/qr_bridge.*`、`src/qr_trace.h`：纯 C 原始观测会话及事件接口。
- `src/qr_filter.h`：可选择的 Cholesky/Joseph 更新核。
- `examples/replay_native.c`：无 Python/PyTorch 的 C 回放示例。

Q 只缩放原有加速度驱动噪声的水平/垂直方差，不学习模糊度 Q，也不输出完整 9×9 矩阵。R 按卫星、频点、码/相位输出站间单差方差倍率；原 `ddcov()` 保留参考卫星相关项。不要对方差倍率再平方。

[梯度范围](docs/REFERENCE_LOSS.md)和[验证边界](docs/VERIFICATION.md)是本实现的一部分。真实权重需自行训练。默认不允许将合成/未训练模型当作实际推理模型。
