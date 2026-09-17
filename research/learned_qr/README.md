# RTKLIB-QR：Q/R 学习研究原型

本目录包含 C99 推理模块、PyTorch 可微滤波基础模块、接入脚本和 26 项组件测试。
它是研究原型，不是已经训练完成的 RTK 系统；不包含 bias 分支。

## 本次提交的边界

原型源码位于 `research/learned_qr/`。本次提交没有修改仓库顶层 `src/rtkpos.c`、`src/rtklib.h` 或 LAMBDA：**接入补丁尚未应用到上游源码**。直接编译仓库顶层程序仍是原始 RTKLIB。

没有完成完整 RTKLIB 编译、真实 RINEX 回放、真实 GNSS 参考轨迹训练或定位精度验证。这里的 26 项测试只验证新增组件，不代表整库或真实 RTK 实验通过。

## 已有代码

- `src/learned_qr.h`：C99 模型加载、前向推理、独立历史缓存，无 PyTorch 运行依赖。
- `src/learned_qr_adapter.h`：RTK 特征提取和 Q/R 接入入口，需安装补丁并重编译后才生效。
- `qrlearn/model.py`：10 历元窗口、16 隐藏单元的时序 MLP；输出有界正方差倍率，不输出坐标或 bias。
- `qrlearn/features.py`：与 C 端对应的特征和窗口格式。
- `qrlearn/filtering.py`：可微预测、双差协方差构造、Joseph 形式 EKF 更新。
- `examples/train_synthetic.py`：仅用于验证反向传播与导出的非 GNSS 合成训练。
- `tools/apply_patch.py`：源码指纹检查、接入安装和保护函数一致性检查。
- `tests/`：C/Python 推理一致性、协方差、梯度、缓存、回退、补丁和合成训练测试。

Q 只缩放原有 ENU 加速度驱动噪声的水平/垂直项，不学习模糊度噪声。
R 分别预测每颗卫星、频点、码/相位的站间单差方差倍率，再由原始 `ddcov()` 保留双差相关项。完整特征定义见 [docs/FEATURES.md](docs/FEATURES.md)。

## 运行组件测试

在 Linux/WSL、Python 3.10+、具有 C99 编译器 `cc` 的环境中，从仓库根目录执行：

```bash
cd research/learned_qr
python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
```

发布前复测：`26 passed in 3.64s`。范围见 [docs/VERIFICATION.md](docs/VERIFICATION.md)。

合成训练仅用于软件测试，不能产生真实 RTK 模型：

```bash
python -m examples.train_synthetic --epochs 20 --length 40 --output outputs/synthetic.qr
```

输出明确标为 `synthetic`，C 端默认拒绝使用。这里不附带真实训练权重。

## 在独立本地工作树中应用接入补丁

不要直接对当前包含本目录的工作树运行安装器。安装器需要代码包在目标工作树外部，目标干净且没有已有 `research/learned_qr` 目录。以下命令从仓库根目录执行，使用全新的目录和分支名；如已存在，应停止检查，不要删除或强制覆盖：

```bash
cp -R research/learned_qr ../RTKLIB-QR-kit-install
# 不移动 main，也不覆盖本研究分支。
git worktree add -b research/learned-qr-integration ../RTKLIB-QR-integration 06e8644287ff07efc4c53bbf3e7f9dafb0355605
python ../RTKLIB-QR-kit-install/tools/apply_patch.py --repo ../RTKLIB-QR-integration --check
python ../RTKLIB-QR-kit-install/tools/apply_patch.py --repo ../RTKLIB-QR-integration
```

安装器要求 `rtkpos.c` blob 为 `edf17ff4e507687b597a0cfc2232db70099b8527`，不匹配就拒绝修改。安装后必须重编译全部相关二进制：`rtk_t` 增加上下文指针，不能混用旧库。安装器本身不提交或推送。

安装后的环境变量：`RTKLIB_QR_MODE=off/q/r/qr`、`RTKLIB_QR_MODEL`、可选 `RTKLIB_QR_LOG`。默认关闭；无有效模型时回退。`RTKLIB_QR_ALLOW_TEST_MODEL=1` 仅供受控测试，不代表训练完成。Q 限正向动态 RTK；R 不支持 PPP、DGPS、IFLC 或反向解算。

## 参考坐标能否用于 loss？

可以将估计位置与参考位置之差作为训练损失，但当前 C 代码只有推理，不含滤波反向传播。不能仅把 C 输出坐标读进 Python 就自动得到网络梯度。详见 [docs/REFERENCE_LOSS.md](docs/REFERENCE_LOSS.md)；尚缺的实现见 [CODEX_HANDOFF.md](CODEX_HANDOFF.md)。

## 许可证

新增代码使用 [LICENSE-QR](LICENSE-QR)。上游 RTKLIB 保留自身许可证与版权声明。
