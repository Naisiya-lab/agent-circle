# 四分片停滞诊断

`select_after_read_N`表示读取工具后的第N次选择调用。N不断增加而样本不变，需要比较模型的inspect请求、工具返回和下一轮输入；这不等同于HTTP重试或核查返工。

建议先在测试调度终端按Ctrl+C，让测试停止，保留模型服务和全部结果。无需清除锁或样本文件。独立采集脚本也可读取仍在运行的文件，会记录缺失、超大或尚未写完整的文件。

```bash
cd /nc1test1/lwj/agent
git pull --ff-only origin main
python scripts/collect_stall_diagnostics.py --experiment qwen25_7b_full4gpu_v2 --images
```

脚本打印输出压缩包绝对路径。默认只采集每个分片当前样本，最近8个模型阶段文件、初始及首轮工具交互、核查返工反馈、配置和提示词快照、相关代码快照。阶段文件中的input保留上一轮请求与工具返回，raw_response保留模型响应。每个日志仅取末尾128KiB。只导出当前样本的清单条目，可选附原图；不会打包完整清单、数据集、模型、全部样本或海量attempts/streams/tools子目录。

单文件上限2MiB、归档内容上限32MiB，各分片共享预算；超限和缺失记录在manifest.json中。只扫描当前样本目录的文件名来选择最新阶段，不读取全部历史阶段。不会调用模型API、重跑阶段、修改实验状态或删除文件。密钥字段和Bearer值脱敏，不收集环境变量。

如自动识别不到卡住样本，可明确指定（2026-11-05截图）：

```bash
python scripts/collect_stall_diagnostics.py \
  --experiment qwen25_7b_full4gpu_v2 --images \
  --sample s0=TABMWP_2444 \
  --sample s1=TABMWP_33750 \
  --sample s2=TABMWP_15687 \
  --sample s3=TABMWP_16606
```

已知主流程在action=inspect时持续查询，每次回传上一轮请求和结果，没有累计完整查询历史。仅凭截图尚不能区分重复相同位置、反复无效位置、丢失已查信息或模型未遵循工具反馈；应先查看上述证据，不能将核查返工次数与工具查询次数混为一谈。
