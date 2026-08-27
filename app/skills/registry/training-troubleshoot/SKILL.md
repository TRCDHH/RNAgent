---
name: training-troubleshoot
description: 训练过程中的异常信号诊断规则。
---

## 异常信号
- CUDA out of memory → 减半 batch_size 重试
- loss NaN/Inf → 学习率过大
- 显存持续 >95% → 可能 OOM，告警
- EarlyStopping 触发 → 正常结束
