# RTKLIB-QR 研究分支

本分支保存 Q/R 学习研究原型，代码位于 [research/learned_qr](research/learned_qr/README.md)。

**本次仅导入原型源码、测试和接入脚本；未将补丁应用到顶层 RTKLIB。直接编译顶层源码仍运行原始 RTKLIB。** main 不作修改，原有整数固定算法不作修改，无 bias 分支。

- [代码、组件测试与安装说明](research/learned_qr/README.md)
- [用参考坐标作为训练损失：实现方式与当前缺口](research/learned_qr/docs/REFERENCE_LOSS.md)
- [发布前验证范围](research/learned_qr/docs/VERIFICATION.md)
- [后续集成与真实训练任务](research/learned_qr/CODEX_HANDOFF.md)

组件测试 26 项通过不代表整库编译或真实定位验证。当前没有完整 RTK 端到端训练器、真实训练权重或定位提升结论。原始项目说明见 readme.txt，上游许可证见 license.txt。
