# 原生断言与 CI 验证

默认 CMake Release 可能带 `-DNDEBUG`，这会关闭上游 C 测试中的 `assert()`。因此常规 GitHub Actions 验证明确使用 `-O3 -UNDEBUG`，保留优化但打开断言；不是仅依赖 Release 下的冒烟测试。

2026-09-19 本地实际补充执行：

```bash
cmake -S . -B build_assert -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS_RELEASE='-O3 -UNDEBUG' \
  -DCMAKE_CXX_FLAGS_RELEASE='-O3 -UNDEBUG'
cmake --build build_assert -j4
ctest --test-dir build_assert --output-on-failure
```

结果：55/55通过，0失败，1.91秒。使用这次重编译的原生库再次运行学习模块测试：47通过，8.57秒。

CI 安装完整上游测试要求的 `libblas-dev` 和 `liblapack-dev`，之后重新构建、运行原生断言测试和真实 RINEX 训练链路测试。原始输出存储在每次 workflow run 的 `qr-validation` artifact；最终状态以该提交对应的 run 为准。

这些是软件验证；人工参考标签不构成真实定位精度或泛化证据。常规 CI 只读，不会提交、覆盖或合并任何分支。完整用法及算法梯度边界见 `TRAINING.md` 与 `REFERENCE_LOSS.md`。
