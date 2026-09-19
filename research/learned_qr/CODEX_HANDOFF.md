# 当前开发状态

顶层C集成、实时原生事件接口、参考轨迹监督训练、模型导出、原生回放和命令行已经实现；不再停留于最初原型。不要重新运行历史 `tools/apply_patch.py` 或覆盖顶层文件。

先读 `docs/TRAINING.md`、`docs/REFERENCE_LOSS.md` 和 `docs/VERIFICATION.md`。构建全库后运行 CTest 与本目录pytest。保留 `upstream_protected.json` 的整数算法保护测试。不得将仅有合成标签的测试当成精度证据。

扩展其他profile时，必须补齐相应原生状态操作事件、同路正向对齐、重置/间断测试，并重新验证条件梯度范围。尤其不能在未添加事件前开放电离层/对流层估计、IFLC、GLO autocal或固定基线约束。使用独立研究分支，不改main、不加bias head。
