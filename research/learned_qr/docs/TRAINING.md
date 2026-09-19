# 从自己的观测与参考轨迹开始训练

本实现是对论文“用最终位置误差优化滤波随机模型”的 RTK 工程迁移，不是对原文 DGNSS/bias/Transformer 的逐项复现。训练是**条件一阶 FLOAT 递推求导**，详见 `REFERENCE_LOSS.md`。不要把代码测试通过理解为自己的定位精度必然提升。

## 1. 编译与环境

已验证 Linux、GCC/CMake、CPU、Python/PyTorch。Windows DLL 导出路径没有完整平台验证；Windows 用户可在 WSL2 按下列步骤运行。不能把旧的 RTKLIB DLL 与本分支混用：`rtk_t` 新增了上下文字段，所有依赖模块必须重编译。

```bash
git clone --branch research/learned-qr https://github.com/xk17820/RTKLIB-QR.git
cd RTKLIB-QR
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
python -m venv .venv
. .venv/bin/activate
python -m pip install -r research/learned_qr/requirements.txt
ctest --test-dir build --output-on-failure
(cd research/learned_qr && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q)
python research/learned_qr/run_qr.py --help
```

默认库路径为仓库 `lib/librtklib.so`；安装到其他位置时传入 `--library /absolute/path/librtklib.so`。推荐先用 CPU。训练要求 float64，不能为了速度直接改为 float32。`--device cuda` 接口保留，但没有做 GPU 端全链路验收；C 核心始终在 CPU。

## 2. 支持的首版解算配置

使用 `configs/short-baseline.conf`，按自己的基站坐标、系统、频数等作明确修改。桥接入口对不支持的选项直接报错，不悄悄忽略。

当前 profile：短基线、kinematic、dynamics=on、forward、niter=1；广播星历；电离层 off/brdc，对流层 off/saas；不启用潮汐、残差时间插值、IFLC、电离层/对流层状态估计、GLO autocal、固定基线长度约束和外部天线/ERP/DCB/站点文件。默认 `pos1-navsys=41` 即 GPS+Galileo+BDS，双频。可用频数由编译的 NFREQ 限定。

基站位置可来自 RINEX header，或通过原生 `ant2-postype` 与 `ant2-pos1/2/3` 明确给出 XYZ/LLH。真实基站坐标不准会影响结果；**不能用流动站参考轨迹充当基站坐标**。

训练临时关闭 AR；回放不带 `--float` 时使用配置原有 AR。原 LAMBDA、固定管理和 hold 代码不改。Q/R 会改变进入整数搜索的 FLOAT 状态/协方差，因此固定结果仍可能变化。

## 3. 每条路线需要的文件

流动站 RINEX 观测、基站 RINEX 观测、一个或多个广播星历 RINEX、独立高精度参考轨迹 CSV。RINEX 由 RTKLIB 原解析器读取，不是另写一个仅支持玩具数据的解析器。对于压缩/Hatanaka 等外部压缩格式，先转换为普通 RINEX 再使用本入口。

参考 CSV 支持两种明确格式，任选一种：

```csv
gps_week,tow_s,x_m,y_m,z_m,valid
```

或

```csv
t_gpst_s,x_m,y_m,z_m,valid
```

`gps_week` 是完整 GPS 周，`tow_s` 范围 `[0,604800)`；`t_gpst_s` 是从 **1980-01-06 GPS 时标零点起的秒数，不是 Unix 秒数，不是 UTC 秒数**。XYZ 单位米、ECEF 地固系、GNSS 天线参考点。`valid=1` 才可监督，`valid=0` 不参与；无效行可留空坐标。时间必须严格递增，不允许重复。

参考与观测必须统一时间系统、坐标框架、天线点和高度基准。参考来自 GNSS/INS 时，先完成杆臂和时间延迟处理；代码不会自动猜测这些改正。无参考时仍可 C 回放，但不能监督训练。

插值仅在两端都有效且间隔不大于 `reference_max_gap_s` 的区间进行；不外推，不跨失效区补真值。`reference_time_offset_s` 明确**加在参考时间上**，默认0；不要靠调整偏移去拟合测试结果。

### 从独立参考 `.pos` 转换

```bash
python research/learned_qr/run_qr.py convert-reference \
  --input /data/independent_truth.pos --output /data/reference.csv \
  --coordinate-format xyz --time-format week-tow --quality 1
```

经纬高参考使用 `--coordinate-format llh`，纬/经为十进制度，高度必须是椭球高。GPST 日历格式用 `--time-format calendar-gpst`。工具拒绝标注 UTC 的文件；必须先明确进行 UTC→GPST 转换。`--quality` 指可信的**参考解**质量，不是被评估算法的质量。不要把本算法自己的解当作独立真值。

## 4. 路线清单与分组

复制 `configs/dataset.example.json` 到自己的工作目录并改为真实路径。该示例里的文件名只是输入位置示例，不是随仓库提供的数据。

所有路线文件路径相对 **manifest 所在目录** 解析，也支持绝对路径。每条路线包含 `id, rover, base, nav, reference`；`nav` 是数组。可以为某一路线另设 `config`、`reference_max_gap_s`、`reference_time_offset_s`。每个 id 必须唯一。

默认要求独立 validation 路线，拒绝 train/validation 中相同流动站文件路径。程序无法识别重命名或复制后的重复路线；日期/路线/接收机的数据隔离仍由实验设计保证。测试路线不能用于选模型或调超参数。

## 5. 训练

```bash
python research/learned_qr/run_qr.py train \
  --manifest /data/dataset.json \
  --config research/learned_qr/configs/short-baseline.conf \
  --output outputs/run01 \
  --epochs 20 --mode qr --sequence-length 16 --warmup-epochs 5 \
  --learning-rate 0.001 --gradient-clip 1 \
  --max-gap 30 --pair-age 0.05 --reference-max-gap 1 \
  --up-weight 1 --huber-delta 1 --nll-weight 0.001 --reg-weight 0.0001
```

这些是可运行的起始参数，不是已经为你的数据优化的最优参数。低频观测要调整 `--max-gap`：仓库测试为30秒采样，使用31秒；1Hz 路线可从30秒开始。该值既控制特征连续性，也控制桥接会话的整段重置，并写入模型；部署必须一致。

`--pair-age` 是流动站和最近基站历元的最大时间差，不得超过配置 `pos2-maxage`；没有可用基站历元时保留失败历元，不拿旧坐标伪装成新解。

`--mode qr` 按完整数据遍历交替训练：第1轮 R，第2轮 Q，然后重复。`q`/`r` 用于消融。窗口10历元是网络特征窗口；`sequence-length` 是截断反向传播长度，两者不是同一参数。训练遍历中每个块后更新权重，滤波数值继续递推，计算图在块边界截断。跨观测长中断时重置会话；参考无效仅屏蔽 loss，不把它作为滤波观测。

loss 主项为 ENU 分量的 Huber 位置误差，U 分量乘 `up-weight`；`--huber-delta 0` 切换到平方误差。另可加创新 NLL 和 log 方差倍率正则。Q/R 方差倍率都有正值上下界。验证只做前向；根据验证集加权位置 RMSE 保存 `best`。不会对 LAMBDA 离散搜索反传。

输出目录必须尚不存在，避免覆盖旧实验：

| 文件 | 内容 |
|---|---|
| `best.pt` / `last.pt` | 张量 checkpoint；加载使用 `weights_only=True` |
| `best.qr` / `last.qr` | C 可读纯数值权重，无 Python 推理依赖 |
| `history.jsonl` | 每轮 train/validation RMSE、梯度、原生/tape 一致性、有效样本数 |
| `run.json` | 参数、数据清单、梯度近似、选择规则、数值核和来源声明 |

`--init old/best.pt` 仅初始化网络权重，不是恢复 Adam 动量或断点状态。验证集不会参与反传。

`--allow-training-only` 只供流程调试；没有独立验证时，不能从训练 loss 下降推出泛化。人工标签/合成试验必须加 `--provenance synthetic`，C 默认拒绝此类模型。`trained` 是调用者的来源声明，不是自动的真实性或精度认证。零有效参考、无可监督 FLOAT 历元、非有限梯度、原生/tape 明显不一致等都会报错，不导出假成功模型。

## 6. 在自己的测试路线回放

```bash
python research/learned_qr/run_qr.py replay \
  --config research/learned_qr/configs/short-baseline.conf \
  --rover /data/test/rover.obs --base /data/test/base.obs --nav /data/test/brdc.nav \
  --model outputs/run01/best.qr --mode qr --max-gap 30 \
  --output outputs/test_qr.csv --reference /data/test/reference.csv
```

不加 `--float` 就保留配置的 AR。CSV 输出所有尝试的流动站历元，失效状态用0和 NaN坐标；不能只导出成功固定的历元再计算误差。参考仅用于生成同名 `.metrics.json`，不传入 C 会话。

公平比较至少包括：相同参数下 `--mode off --legacy`（原更新核）、`--mode off`（稳定核）、`--mode qr --model ...`（稳定核+学习）。分离比较 Q-only、R-only 也可使用 `--mode q/r`。要严格消融训练效果，分别用相应模式独立训练，不只是临时关闭一个已经联合训练的分支。

```bash
# 稳定核、不学习：用于分离学习增益
python research/learned_qr/run_qr.py replay --config research/learned_qr/configs/short-baseline.conf \
  --rover /data/test/rover.obs --base /data/test/base.obs --nav /data/test/brdc.nav \
  --mode off --output outputs/test_stable.csv --reference /data/test/reference.csv
# 原更新核基线：同上命令额外加 --legacy，并换输出文件名
```

报告同时检查覆盖率、FLOAT/FIX 分组误差、RMS H/U/3D、P95/P99、最大误差及固定解大位置误差事件。`fixed_large_error_proxy` 是参考位置阈值超限的代理指标，**不是知道每个真整数后的错误固定率**。参考覆盖率不一致时，不要直接比较两个总 RMS。

也可单独评估：

```bash
python research/learned_qr/run_qr.py evaluate --solution outputs/test_qr.csv \
  --reference /data/test/reference.csv --output outputs/test_metrics.json
```

## 7. 纯 C 部署，不依赖 Python/PyTorch

训练权重导出后，可编译示例 C 客户端；它直接调用原生桥接会话，使用与训练相同的基站配对和长间隔重置策略：

```bash
cc -std=c99 -Isrc research/learned_qr/examples/replay_native.c \
  -Llib -lrtklib -lm -Wl,-rpath,"$PWD/lib" -o bin/qrreplay
bin/qrreplay research/learned_qr/configs/short-baseline.conf \
  /data/test/rover.obs /data/test/base.obs outputs/run01/best.qr outputs/c_replay.csv 30 \
  /data/test/brdc.nav
```

示例参数中的模型路径用 `-` 表示稳定核关闭学习基线。输出独占创建，不覆盖原文件。回放本身不需要参考坐标。`qr_bridge.h` 提供 `qr_open/qr_step/qr_restart/qr_close`；callback数组只在回调期间有效。除 INFER 事件的倍率输出外，不允许修改借用数组。示例 C 程序中学习权重已经加载，因此回调仅导出结果，不做 Python 推理。

原 `rnx2rtkp` 也支持通过环境变量加载模型：

```bash
RTKLIB_QR_MODE=qr RTKLIB_QR_MODEL=outputs/run01/best.qr \
  bin/rnx2rtkp -k research/learned_qr/configs/short-baseline.conf \
  -o outputs/rtk.pos /data/test/rover.obs /data/test/base.obs /data/test/brdc.nav
```

这个入口保留 RTKLIB 原有的完整后处理读入/间隔处理策略；跨长间隔时不等同于桥接入口的整段重置。需要与训练严格一致时使用桥接 C 示例或 Python `replay`；使用原入口时按同样的连续段切分，并保留其模式限制。

原入口默认学习关闭且未设置 `RTKLIB_QR_STABLE=1` 时走旧的 `filter()`；学习开启且有效模型加载后走 Cholesky/Joseph 核。无效模型回退并打印警告，不能忽视该警告然后声称已使用学习。桥接命令行对模型错误直接失败。

## 8. 已验证与尚需自行验证

软件测试和短段基线回归见 `VERIFICATION.md`。提供了真实 RINEX 的链路测试，但参考标签是人为构造，仅验证可训练性和部署一致性。没有在你的真实路线训练，也没有提供声称有效的训练权重。最少应在独立路线检查误差尾部和错误固定代理，不能只看固定率或训练 loss。参数、接收机、频数、采样间隔改变后应重新验证。
