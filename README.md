# agent-circle

整理日期：2026-10-24。保留当前四角色工具循环的最终实现，支持**同一基础模型M直接回答**与**同一M承担全部角色的循环**。旧`loop_agent`、`muti_agent`和所有历史结果均保留。

- [模型与论文调研、4点提升的解释](research/model_survey_20261024.md)
- [框架与通信](docs/FRAMEWORK.md)
- [迁移范围与验证状态](docs/MIGRATION.md)
- [评估来源及导出](evaluation/README.md)

## 目录

```text
agent_circle.py                 唯一实验入口
tool_loop_runtime/              流程、通信、解析、网络、直接回答基线
  prompts/                      当前英文与中文提示词
table_tools/                    索引、合并格保留、区域提取、坐标映射
configs/runtime.json            通用运行参数，无机器路径
configs/models/                 同模型API/生成参数
configs/legacy_mixed_A,B.json    历史混合模型配置，仅供追溯
scripts/                        Linux环境、模型服务、准备与运行.sh
tests/                          离线工具、返工、解析、通信测试
evaluation/official/            保留的官方评估源码与来源记录
research/                       调研与公开元数据核查
prepared/                       生成的无答案测试清单
runs/                           独立运行输出；不含旧实验结果
```

## 远程Linux开始方式

只需把此目录上传远程，数据与模型权重单独放置。客户端只需要Python>=3.10标准库；模型服务请在独立环境安装`requirements-server.txt`，先核对CUDA与驱动。以下尚未在A800实机执行。

```bash
cd /你的目录/agent-circle
cp scripts/env.example.sh scripts/env.local.sh
# 编辑env.local.sh：MMTAB_ANNOTATIONS、MMTAB_IMAGES、MODEL_WEIGHTS等远程路径
bash scripts/check_offline.sh
bash scripts/prepare_eval.sh
```

服务终端：

```bash
CUDA_VISIBLE_DEVICES=0 PORT=8000 bash scripts/serve.sh qwen25_vl_7b
```

运行终端（同一个M，同一份清单）：

```bash
# 先少量烟测，不代表已完成性能测试
RUN_NAME=qwen25_7b_direct_smoke bash scripts/run.sh qwen25_vl_7b baseline --max-samples 3
RUN_NAME=qwen25_7b_circle_smoke bash scripts/run.sh qwen25_vl_7b circle --max-samples 3
# 相同命令去掉--max-samples 3后，续跑清单剩余样本
```

默认B：理解不编码颜色、选择同时看原图与HTML；A：理解保留颜色、选择只看HTML。通过`ROUTE=A`切换，必须使用独立RUN_NAME。新同模型A与历史GLM/Qwen混合A不是同一次实验。

API模型使用相同入口，例如`glm_flash`；密钥由环境变量或交互隐藏输入，不写入.sh、配置或运行文件。`qwen_vl_plus`是视觉模型，与文本`qwen-plus`不同。

配置优先顺序：`scripts/env.local.sh` → `scripts/models/<模型>.sh`的默认值 → Python读取环境变量。机器路径通过环境或CLI给出，Python中没有D盘/服务器数据路径。`configs/runtime.json`保存可复查的算法运行参数，.sh不承担手工拼JSON。

模型版本请设置`MODEL_REVISION`为checkpoint commit；闭源别名另固定可用快照/日期版。生成选项在同一M的两组间保持一致；直接基线与求解使用相同输出上限，但循环总预算更大，不能称等计算量比较。

## 清单和分片

默认全49,726条，保留T2T和OOD。排除T2T时另设MANIFEST，保留完整清单：

```bash
MANIFEST="$PWD/prepared/mmtab_eval_no_t2t.json" bash scripts/prepare_eval.sh --exclude-tasks T2T
```

4卡4副本时，分别用`--shards 4 --shard-index 0/1/2/3`准备清单；不同终端设不同PORT、MANIFEST和RUN_NAME。先做远程烟测再全量，不会自动启动付费或GPU任务。

## 保存与恢复

每次运行固定配置、源码、提示词、清单及图像哈希。同名运行不允许换配置续跑；格式失败重试一次，内容返工最多一轮，网络重试独立计数。普通样本失败继续后续样本，鉴权等致命错误停止。

```bash
bash scripts/run.sh qwen25_vl_7b circle --retry-errors
python3 agent_circle.py report --run 你的运行名
python3 agent_circle.py export --run 你的运行名 --output predictions.jsonl
```

导出采用官方LLaVA式`question_id/text/model_id`；未完成样本导出空答案并保留分母，不伪造正确性。当前未运行新的模型实验，`passed`只表示核查模型接受，不是评测得分。
