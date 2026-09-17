# 用参考坐标训练 Q/R：可行性与当前边界

## 原理

参考坐标是监督目标；loss 是估计坐标与参考坐标的差，而不是参考坐标本身。例如：

```text
e_k = ECEF_to_ENU_rotation(reference_k) @ (position_float_k - reference_ecef_k)
L_pos = sum_k valid_k * (e_E^2 + e_N^2 + w_U * e_U^2) / sum_k valid_k
```

这是拟采用的工程目标示例，不声称逐式复现原论文。参考轨迹需要与观测时间系统、采样时刻、坐标框架和天线参考点一致；参考无效的历元不参与监督。

所需梯度链是：网络参数 -> Q/R -> 跨历元滤波状态/协方差 -> 估计位置 -> loss -> 网络参数更新。只输出 Q/R 并计算位置误差，不足以形成这条梯度链。

## 建议采用的训练/部署分离

训练端用 PyTorch 实现与 C 端对应的连续 FLOAT 递推。观测残差和 Jacobian 必须在每次当前估计状态上重算，不能一直复用基线解算导出的固定残差。训练结束导出网络权重，C 端只做前向推理并将 Q/R 输入原有 RTK 解算。

参考轨迹只用于训练损失和独立评估，不作为推理特征，也不能在部署中将参考坐标反馈给滤波器。

第一版不对 LAMBDA 的整数搜索和离散接受/拒绝判断求普通梯度。保留这些代码用于部署和完整回放，分别评估 FLOAT 和 FIX。FLOAT loss 降低不能自动证明固定率或错误固定指标改善。

## C 语言不是障碍，缺少导数接口才是问题

可保留 C 计算，通过自定义 PyTorch 运算提供 forward/backward，将所需导数传回网络；这需要实际实现和数值梯度检查。普通 C 函数调用、文件读写、ctypes 调用本身不会自动为任意 C 程序生成反向传播。

也可在 C 外层使用无梯度优化或数值差分，以参考位置误差迭代少量 Q/R 超参数。这属于黑箱调参，不等同于论文的可微端到端网络训练；不能混称。

PyTorch 官方接口说明：
- https://docs.pytorch.org/docs/stable/notes/extending.html
- https://docs.pytorch.org/docs/stable/library.html#torch.library.register_autograd

## 当前仓库具备与缺失的内容

具备：双分支网络、可微滤波数学原语、原生推理和合成训练测试。
缺失：真实 GNSS 观测适配器、与完整 RTKLIB 对齐的 FLOAT 递推训练链、实际 RINEX/星历/参考轨迹加载与质量掩码、真实训练与独立验证。

因此，当前不能仅提供一个参考坐标文件就开始训练真实 RTK。`examples/train_synthetic.py` 只验证合成问题的梯度和权重导出，不能把其中的观测模型当作真实 GNSS。
