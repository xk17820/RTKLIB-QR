# RTKLIB-QR：参考轨迹监督的 Q/R 学习与 C 部署

研究分支：`research/learned-qr`。`main` 保留上游。**原有 LAMBDA、固定管理、fix-and-hold 和双差协方差函数不改；不增加 bias head、不回归坐标。**

现在顶层 C 解算器已接入 Q/R；不是必须先运行补丁的原型目录。提供原始 RINEX → 在线原生 FLOAT 递推 → 参考位置损失 → 网络更新 → 权重导出 → C 回放的可执行流程。

- [完整训练、数据格式和部署说明](research/learned_qr/docs/TRAINING.md)
- [梯度定义及与原论文的区别](research/learned_qr/docs/REFERENCE_LOSS.md)
- [验证记录及性能结论边界](research/learned_qr/docs/VERIFICATION.md)
- [数据清单示例](research/learned_qr/configs/dataset.example.json)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
python -m pip install -r research/learned_qr/requirements.txt
python research/learned_qr/run_qr.py --help
python research/learned_qr/run_qr.py train --manifest /path/dataset.json \
  --config research/learned_qr/configs/short-baseline.conf --output outputs/run01
```

**方法边界：**训练是条件一阶 EKF 求导，局部 H、坐标旋转、因果特征及离散选择停止梯度；不是对全部 C 指令和整数搜索做自动微分。训练/桥接回放使用 Cholesky/Joseph 数值更新，必须同时比较“稳定核关闭学习”的基线，不能把数值核变化带来的增益算成学习增益。原生 `rnx2rtkp` 默认关闭学习且不启用稳定核时保留旧路径。

验证使用仓库附带的真实格式 RINEX，训练链路测试中的参考标签是人工构造的；没有随仓库提供真实训练权重或定位改善结论。用户需在自己的独立训练、验证、测试路线中评估。
