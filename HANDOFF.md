# 新聊天从这里继续：RTKLIB-QR / SCL-RTK

更新：2026-09-20。目的：按用户要求保存当前源码、结论和未完成工作。**这是研究工作快照，不是性能通过或可部署版本。**

仓库 `xk17820/RTKLIB-QR`；只在 `research/learned-qr` 工作。`main` 应保持 `06e8644287ff07efc4c53bbf3e7f9dafb0355605`。不得强推、覆盖已有研究提交，或再询问已经确认的技术路线。

## 1. 当前状态，先看证据

进入本次归档时，研究分支为 `1940876021724d86036ee31e3c184d66c2bc8fb4`。此前 SCL 修改已以 `.scl-publication` 的4段压缩补丁保存到远端，但没有展开为最终的普通源码提交：上次 CI 在学习测试失败后跳过了发布。

已直接读取的远端运行：
- [运行 35497731869](https://github.com/xk17820/RTKLIB-QR/actions/runs/35497731869)，job `106043760195`。
- 27个源文件的补丁校验及应用成功；原生程序构建成功。
- CTest：55/55通过，断言开启。
- Python：**74通过、1失败**，不是75项全部通过。
- 失败：`tests/test_guard_policy.py::test_identity_learns_but_no_shrinking_or_clean_reweighting`。
- 现象：模型输出倍率为1，但 `m.r(h).backward()` 后 `m.r.fc2.bias.grad.abs().sum()` 为0；测试要求大于0。
- 该次环境：Python3.11.16、torch2.14.0+cpu、numpy2.4.6、pytest9.1.1、Ubuntu24.04.5。根因尚未最终确定，不能仅凭版本差异断言原因。

本次快照会将校验过的27个源文件展开到正常路径，保存补丁、清单及新运行日志；即使测试失败也按用户要求保存WIP，并明确保留失败状态。**这不是绕过测试批准发布。**

实际快照结果以 [SNAPSHOT_STATUS.json](research/learned_qr/handoff/2026-09-20/SNAPSHOT_STATUS.json) 和其中的运行链接为准。若该文件不存在或 `source_materialized=false`，只表示交接文档已提交，不得宣称展开源码成功。原始压缩补丁可在进入归档时的提交历史中恢复；归档后证据保存在 [evidence/](research/learned_qr/handoff/2026-09-20/evidence/)。

## 2. 用户已确认的目标和约束

方法统一称 **SCL-RTK — Structured Covariance Learning for RTK（结构化协方差学习RTK）**。源自参考位置监督滤波随机模型的思路，但原参考论文是JUB-KF伪距DGNSS；本工程是码/载波相位RTK迁移，不复制其bias分支。

必须保留：原有整数搜索、固定管理和fix-and-hold的算法行为；双差共同参考卫星相关项；原生C部署。只学习Q/R，不增加观测偏差回归、不直接回归或修正坐标。参考轨迹仅用于loss/评估，不作网络特征、滤波观测或位置初值。

训练是原始RINEX重新遍历、当前状态下重算观测的条件一阶滤波求导；局部H、坐标旋转、离散选择等停止梯度。不能称为全部C指令或LAMBDA的精确自动微分。区分原数值核基线和Cholesky/Joseph稳定核基线，不能把稳定核收益全部算成学习收益。

## 3. 已保存的代码与下一步入口

常规代码在 `src/` 与 `research/learned_qr/`。后者包含 `run_qr.py`、`qrlearn/`、`tests/`、`src/`镜像和说明。SCL增量包括：
- 原生候选事件/观测接口及条件整数候选位置计算；
- V3因果观测一致性协方差策略；
- `qrlearn/scl.py`、`sparse_training.py`、模型/CLI/训练更新；
- FLOAT/DGPS监督处理修正与新增回归测试。

这些条目描述源码保存范围，不表示完整科学验证完成。阅读 [SCL_RTK.md](research/learned_qr/docs/SCL_RTK.md)、[REFERENCE_LOSS.md](research/learned_qr/docs/REFERENCE_LOSS.md)、[TRAINING.md](research/learned_qr/docs/TRAINING.md)，并与源码核对；旧说明中的历史测试数字不能覆盖当前失败证据。

先在独立干净工作树构建、复现失败：

```bash
git clone --branch research/learned-qr https://github.com/xk17820/RTKLIB-QR.git
cd RTKLIB-QR
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS_RELEASE='-O2 -UNDEBUG' -DCMAKE_CXX_FLAGS_RELEASE='-O2 -UNDEBUG'
cmake --build build -j2
ctest --test-dir build --output-on-failure
cd research/learned_qr
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
python -m pytest -q tests/test_guard_policy.py::test_identity_learns_but_no_shrinking_or_clean_reweighting
```

依赖/输入profile见TRAINING.md。**不要删除失败测试、放宽断言或把synthetic权重改标签来制造成功。**先检查边界导数约定、训练与C前向一致性及版本差异，再以失败复现→修正→完整测试闭环处理。修正前后均保留日志。

## 4. 结论必须分版本

详细历史见 [CONCLUSIONS.md](research/learned_qr/handoff/2026-09-20/CONCLUSIONS.md)。核心结论不是“已全面超过原RTK”：
1. 原自由倍率学习在首次5条留出模拟轨迹上退化；该负结果已存在于仓库报告。
2. 码/相位分开与保护性门控曾在小型同分布模拟上显示明显收益，但存在低C/N0干净数据和固定解位置超限反例。
3. 扩大到完整长序列后，RMS提升与固定解位置超限增加同时出现；四轮候选均被接受规则拒绝。
4. 因而提出SCL新设计：观测一致性特征、结构约束、条件整数候选位置辅助loss、完整RTK验证与独立重训消融。当前代码部分已恢复，但新大规模实验和新版正式论文仍未完成。

“末轮诊断候选”仅指最后一轮保存、未通过接受条件、用于失败分析的权重，不是方法名称，不是验证选中的部署模型。身份回退不能记作学习成功。所有FIX位置阈值超限在未核验真整数前均称代理指标。

## 5. 新聊天的优先顺序

先修复当前确定的梯度回归并核对C/Python前向；再完成条件候选导数、索引/单位、触发频率及原有AR不受影响的验证；随后冻结数据划分和模型选择规则，按 [NEXT_EXPERIMENT_PLAN.md](research/learned_qr/handoff/2026-09-20/NEXT_EXPERIMENT_PLAN.md) 实施。

88条序列、180000个预定时间槽、至少3个训练初始化种子是**已同意的计划，不是已经完成的实验**。不要继续把旧候选改名当作SCL结果；不要只扩大随机种子而仍共用全部几何条件。已有反复分析的测试集转为开发/回归集。

论文目标仍为完整研究论文，不是开发日志：I引言、II方法、III数据与结果（随机模型分析→完整轨迹比较→独立重训消融→场景/泛化）、IV结论。失败检查和源码哈希进附录，关键反例必须保留正文。不能承诺任意数据都更好。

## 6. 文件保存边界

本次保留可从远端恢复的最新代码、实际CI失败记录、历史结论转录及后续协议。**本会话的运行容器和Files检索未找回旧的大体积实验ZIP、逐历元CSV、生成论文Word。**因此它们没有被假装重新上传，名称与恢复线索在 [ASSET_INVENTORY.md](research/learned_qr/handoff/2026-09-20/ASSET_INVENTORY.md)。原始未发表参考稿 `lcymanuscript.docx` 不在本次公开上传范围。

## 复制给新聊天

> 请继续 xk17820/RTKLIB-QR 的 research/learned-qr 分支工作，首先读取根目录 HANDOFF.md、其中链接的 SNAPSHOT_STATUS.json、CONCLUSIONS.md 和 NEXT_EXPERIMENT_PLAN.md，并核对最新提交/CI。先修复初始倍率1时梯度为0的失败测试，核验SCL条件整数候选训练和因果观测特征，保留原有整数算法、不加bias分支、不改main。新大规模实验与新版论文未完成；不要把旧诊断候选或身份回退当作成功结果。按已确认的独立训练/验证/盲测、完整长序列、重训消融计划继续，无需重新讨论或再次确认方案。
