# Qwen2.5-VL-7B 四卡全量运行

这是部署和任务调度脚本。现有主流程、工具、提示词、输出上限和最多一次返工规则均保持原样。

## 固定运行安排

使用用户已经准备的清单：

    /nc1test1/lwj/data/MMTab_eval/prepared/mmtab_eval.json

模型权重：

    /nc1test1/lwj/Models/Qwen2.5-VL-7B-Instruct

四张卡各运行一个完整模型副本，tensor-parallel-size=1。按原清单顺序交错分片，保留全部字段；运行前核对ID无重复、无遗漏。

|分片|GPU|端口|49,726条清单对应数量|
|---|---|---|---:|
|s0|0|8000|12,432|
|s1|1|8001|12,432|
|s2|2|8002|12,431|
|s3|3|8003|12,431|

顺序固定为：四份循环并行 → 等全部处理完 → 四份单模型基线并行。两阶段读取完全相同的四个清单文件。

## 远程执行

先激活现有 agent 环境。若旧的单卡全量测试仍在跑，先在其测试终端按Ctrl+C停止旧测试，保留已有模型服务。不要让旧全量进程同时占用8000服务。

~~~bash
cd /nc1test1/lwj/agent
git pull --ff-only

# 可选：只准备并核对分片，不启动模型或测试
bash scripts/full_qwen25_7b_4gpu.sh prepare

# 在tmux会话中运行，一条命令启动服务和两个测试阶段
bash scripts/full_qwen25_7b_4gpu.sh run
~~~

脚本不读取旧的 scripts/env.local.sh，避免旧单卡配置覆盖新端口。默认路径已经按上述远程目录填写。若需调整，复制并编辑专用配置：

~~~bash
cp -n scripts/full_qwen25_7b_4gpu.env.example.sh scripts/full_qwen25_7b_4gpu.env.local.sh
~~~

启动前以/v1/models检查服务名、服务返回的权重目录与上下文长度（若返回这些字段）。已有8000服务匹配时复用；其单卡GPU0设置应与之前的启动命令一致，HTTP接口本身不能证明GPU归属。其余服务通过CUDA_VISIBLE_DEVICES分别绑定一张卡。端口被其他模型占用时停止并说明原因，不终止已有服务。

## 结果与监控

默认控制目录：

    /nc1test1/lwj/agent/runs/qwen25_7b_full4gpu_v1/

其中包含plan.json、services.json、各服务日志、四份循环日志、四份基线日志、阶段结束记录和合并导出。

~~~bash
# 新开一个终端，同样激活agent环境
cd /nc1test1/lwj/agent
bash scripts/full_qwen25_7b_4gpu.sh status

# 查看某份循环的实时输出
tail -f runs/qwen25_7b_full4gpu_v1/circle_s0.console.log
~~~

八个原始运行目录分别为：

    runs/qwen25_7b_full4gpu_v1_circle_s0/   （以及s1、s2、s3）
    runs/qwen25_7b_full4gpu_v1_baseline_s0/ （以及s1、s2、s3）

完整预测按原始清单顺序合并到：

    runs/qwen25_7b_full4gpu_v1/exports/circle_all.jsonl
    runs/qwen25_7b_full4gpu_v1/exports/baseline_all.jsonl

导出调用现有agent_circle.py export，不删除失败或空答案。此处只核对样本ID和任务处理状态，不评分、不检查答案内容、不改变智能体判断。

## 失败与续跑

- “处理完”表示样本已有终止结果或不可重试错误，不代表答对。tool_failed、needs_revision等仍保留原状。
- 单个模型错误可以让原脚本返回1；调度器不会仅因返回1就误判整批中断，而是检查是否还有未处理样本。
- 如果任一循环分片提前中断、仍有未处理样本或可重试的通信错误，基线阶段不会启动。查看对应日志后，用同一条run命令续跑。
- 源清单、分片、模型配置、核心代码、提示词和部署参数被固定。续跑时不要更改它们。同名实验不接受这些设置发生变化。
- Ctrl+C会通知本脚本启动的测试进程退出，模型服务保持运行。强制kill可能留下run.lock；必须确认对应进程已经退出，再处理遗留锁。
- 不会自动重跑已保存的tool_failed结果，也不会增加返工次数。
- 模型服务在测试结束后保留，便于继续使用；服务PID记录在控制目录。不要通过通配杀进程误停其他任务。

## 本地验证边界

使用假模型服务检查、合成清单和假测试子进程验证分片、阶段屏障、续跑和合并导出；Bash脚本做语法检查。实际A800/vLLM启动需要在远程执行，没有在当前电脑上声称已启动GPU测试。
