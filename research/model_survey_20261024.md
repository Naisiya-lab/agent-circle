# 表格图像模型调研与实验选择

日期：2026-10-24。范围：表格图像理解、结构理解和推理。采用用户确认的**官方型号3–7B**口径；整模总参数另列。闭源模型不按未公开的参数量归类。

## 1. 先给选择结论

建议第一批使用 **Qwen2.5-VL-7B-Instruct、InternVL2_5-4B、LLaVA-v1.6-Vicuna-7B**，分别覆盖Qwen、InternVL、LLaVA三个视觉模型系列；增加 **Qwen2.5-VL-3B-Instruct** 观察规模变化。若旧LLaVA的短上下文导致大量接口/长度失败，可保留失败率并把 **Qwen3-VL-4B-Instruct** 作为更新模型补充。不要删除旧模型失败后只报告成功样本。

其中Qwen2.5-VL-7B有直接且强的表格论文使用依据；LLaVA-1.6-7B是DiSCo使用的基础模型之一。InternVL2.5-4B是为家族多样性选的合规候选，**本次没有查到它被以下表格论文作为主要训练骨干的直接证据**；不能把论文使用InternVL2.5-8B说成使用了4B。Qwen3-VL-4B同样是更新模型补充，不能把InterTab的8B实验证据转给4B。

所有正式对照都应为：同一个视觉模型M、同一个checkpoint/版本、同一份MMTab-eval清单，分别执行直接图像回答和四角色循环。求解可以只输入文本，但仍调用同一M；不能换成对应系列的纯文本LLM。当前历史GLM/Qwen混合路线的约4点提升，尚不是这种严格同模型对照的证据。

## 2. 开源候选：型号、参数和依据

模型卡的“Model size”是取整显示，不是精确参数审计。此次Hugging Face元数据API直连失败，以下“整模”只记录官方卡片的显示值；实际部署后应保存checkpoint commit及精确参数计数，不能把8B显示伪装成精确8.000B。

| 可运行的多模态checkpoint | 官方型号规模 | 模型卡整模显示 | 表格论文依据／定位 | 建议 |
|---|---:|---:|---|---|
| [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | 7B | 约8B | TableMix实际训练骨干；CoReTab实际训练骨干；多篇表格论文基线 | 第一优先 |
| [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | 3B | 约4B | MM-Table-R1实际基础模型；TableMix比较基线 | 规模泛化 |
| [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) | 7B | 约8B | SynTab-LLaVA比较基线，较早一代高分辨率视觉模型 | 可选历史对照 |
| [OpenGVLab/InternVL2_5-4B](https://huggingface.co/OpenGVLab/InternVL2_5-4B) | 4B | 约4B | 跨家族验证；本次未证实其为经典表格论文主要骨干 | 第一批，标明证据边界 |
| [llava-hf/llava-v1.6-vicuna-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf) | 7B | 约7B | DiSCo使用LLaVA-v1.6-7B；本checkpoint选Vicuna版本 | 第一批；注意短上下文 |
| [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) | 7B | 约7B | Table-LLaVA的多模态基础架构/基础模型系列 | 经典旧基线，可选 |
| [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | 4B | 约4B | 更新模型补充；不是InterTab实际使用的8B骨干 | 第二批 |

### 不混入候选的类别

- InternVL2.5-8B、InternVL3.5-8B、MiniCPM-V-2.6-8B、Qwen3-VL-8B：官方型号8B，超出用户指定范围。
- Kimi-VL-A3B：A3B表示激活规模，不等于总参数3B；不以激活参数冒充小规模稠密模型。
- Gemma3n-E4B：E4B是有效规模口径，不能直接当作整模4B；本次不作为主要候选。
- Vicuna-7B、Qwen2.5-7B、TableGPT2-7B、文本Table-R1：不能直接接收表格图片；与本实验的M定义不同。
- Table-LLaVA-7B、SynTab-LLaVA-7B、MM-Table-R1：是表格专门训练的模型，可另作专用模型实验；不得与原始通用骨干混为一组。SynTab的高低分辨率编码器是定制架构，不能承诺普通vLLM服务直接加载。

## 3. 哪些论文实际用了哪些模型

“经典”与“较新代表工作”分开；2026新工作用于补充当前方法背景，不冒称已成为经典。下表只保留影响本次选择的事实。

| 工作 | 时间／出处 | 实际基础模型 | 训练还是推理改造 | 对本实验的价值 |
|---|---|---|---|---|
| [Multimodal Table Understanding / Table-LLaVA](https://aclanthology.org/2024.acl-long.493/) | ACL 2024 | LLaVA-1.5架构，语言部分Vicuna-1.5-7B/13B | 表格预训练与指令微调 | MMTab-eval源头；选择7B多模态版本，不单独选Vicuna |
| [SynTab-LLaVA](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_SynTab-LLaVA_Enhancing_Multimodal_Table_Understanding_with_Decoupled_Synthesis_CVPR_2025_paper.pdf) | CVPR 2025 | Vicuna-1.5-7B＋定制双分辨率视觉编码器 | 合成数据训练、架构改动 | 说明表格专用checkpoint与通用7B并不等价；比较过Qwen2-VL |
| [MM-Table-R1](https://ojs.aaai.org/index.php/AAAI/article/view/37042) | AAAI 2026 | Qwen2.5-VL-3B | 难度与课程驱动的RL后训练 | 3B也是有直接表格研究依据的规模 |
| [CoReTab](https://aclanthology.org/2026.eacl-long.306/) | EACL 2026 | Qwen2.5-VL-7B | 代码驱动推理数据训练 | 支持7B候选；需区别训练提升和免训练循环收益 |
| [DiSCo / Table-GLS](https://arxiv.org/html/2602.03491v2) | ICML 2026；核对v2 | DiSCo包含LLaVA-v1.6-7B、Gemma3-12B、Gemma3n-E4B、Qwen3-VL-8B；GLS推理重点为后两者 | DiSCo训练对齐；Table-GLS为推理框架 | 有同模型推理改造对照，适合参考收益口径 |
| [TableMix](https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_TableMix_Enhancing_Multimodal_Table_Reasoning_in_MLLMs_from_a_Data-Centric_CVPR_2026_paper.pdf) | CVPR 2026 | Qwen2.5-VL-7B | 混合感知、表格推理、文本数学数据的RL训练 | 当前最明确的7B选择依据之一 |
| [InterTab](https://arxiv.org/html/2609.32660v1) | 2026-09预印本，未据此声称已录用 | Qwen3-VL-8B-Instruct | SFT＋RL，推理中调用结构对齐裁剪工具 | 骨干超范围，但可参考同模型及机制对照 |

来源补充：SynTab [作者仓库](https://github.com/bang123-box/SynTab-LLaVA)；MMTab [作者仓库](https://github.com/SpursGoZmy/Table-LLaVA)。

### 经典论文还比较过哪些3–7B语言骨干？

Table-LLaVA原论文Table 2还包括BLIP2（Flan-T5 3B）、MiniGPT-4（Vicuna 7B）、Qwen-VL（Qwen 7B）、InternLM-XComposer（InternLM 7B）、mPLUG-Owl（Llama 7B）、mPLUG-Owl2（Llama-2 7B）和Monkey（Qwen 7B）。这里原表列名是LLM，**这些数字首先表示语言骨干，不保证整模的官方型号也是3–7B**。它们构成历史候选清单，但不宜全部直接当现代可部署7B模型。

你举的“intern7B”需要区分两个系列：[InternLM-XComposer-VL-7B](https://huggingface.co/internlm/internlm-xcomposer-vl-7b)确实是经典表格论文比较过的系列；[InternLM-XComposer2.5-7B](https://huggingface.co/internlm/internlm-xcomposer2d5-7b)是后续版本，能看高分辨率图，但不能冒称Table-LLaVA当时比较的就是2.5版。它们不同于InternVL2.5-8B。XComposer自定义模型代码及纯文本阶段服务兼容性尚需独立适配核对，本包未承诺vLLM一键加载它。相较之下，InternVL2.5-4B在当前vLLM支持表中有明确条目，因此先作为工程上较稳妥的跨家族候选。

InternVL2.5-4B内部语言部分为Qwen2.5-3B；它与Qwen-VL属于不同视觉模型家族，但不是完全独立的语言骨干家族。LLaVA/Vicuna能补充另一种语言骨干，这也是保留LLaVA候选的原因。

## 4. 闭源模型单列

历史论文用过的闭源版本不保证今天仍可调用。参数量未公开时只写“未公开”，不能称3B/7B。账号权限、地域、配额和快照可用性都需要运行前实际测试。

| 模型／系列 | 论文依据 | 当前实验建议 |
|---|---|---|
| GPT-4V | Table-LLaVA、SynTab-LLaVA以及TableVQA-Bench历史对照 | 作为历史依据，不安排复现已过时的接口 |
| GPT-4o-mini | Table-GLS主表闭源基线 | 官方模型页已标Deprecated；若账号仍可用才作历史复现，不作为新实验默认 |
| GPT-5.4-mini | Table-GLS v2扩展实验 | 可作为闭源候选；需要另核对账号、具体快照和费用，本包尚未调用或配置它 |
| Gemini-Pro Vision | [TableVQA-Bench](https://arxiv.org/abs/2404.19205)历史图像基线 | 历史版本不等于今天可重现；不把Gemini-2.5-Flash当原论文模型替换后仍称复现 |
| Qwen-VL-Plus / Max | Qwen-VL作者官方提供API系列；表格实验可作补充 | 已留Plus配置；默认别名可能漂移，正式测试固定可用日期版。它不同于纯文本qwen-plus |
| GLM-5.3-Flash | 当前项目实际使用，非上述经典论文统一骨干 | 已留同一GLM承担全部角色的配置，避免继续混Qwen；型号、实际接口能力以已有可用账号为准 |

OpenAI状态依据：[GPT-4o-mini模型页](https://developers.openai.com/api/docs/models/gpt-4o-mini)、[弃用说明](https://developers.openai.com/api/docs/deprecations)。Qwen依据：[官方VL仓库](https://github.com/QwenLM/Qwen-VL)、[模型列表/定价](https://help.aliyun.com/en/model-studio/model-pricing)。闭源只完成调研，不调用付费API。

## 5. 提升多少才算有效？

**这些论文没有给出通用的“超过X个百分点才有效”判据。**只能列出相同模型或明确比较对象下的实证差值，并注明数据集、指标、是否训练。

| 比较 | 数据与指标口径 | 原值→新值 | 绝对差值 | 可比性 |
|---|---|---:|---:|---|
| Table-GLS：固定Qwen3-VL-8B-DiSCo，DA→GLS | 8个表格推理benchmark平均；Table 3 | 56.22→59.48 | +3.26点 | 固定已对齐checkpoint，推理方法改变；不是裸模型→全部方法 |
| Table-GLS：GPT-5.4-mini→加GLS | 4个推理benchmark平均；Table 4 | 56.64→66.16 | +9.52点 | 闭源同模型推理流程收益，数据范围不同于本项目 |
| InterTab：匹配训练的文本推理→交替区域推理 | 9个benchmark、LLM评审ACC；Table 2 | 68.89→73.17 | +4.28点 | 两者均训练，非免训练agent循环 |
| InterTab：实际Qwen3-VL-8B→完整InterTab | 同论文Table 1九项平均 | 67.18→73.17 | +5.99点 | 包含训练与工具机制，不能只归因于工具或循环 |
| TableMix：Qwen2.5-VL-7B→TableMix | held-out TableVQA-Bench；论文Table 2所报AVG | 70.2→82.3 | +12.1点 | RL后训练，非同成本推理；沿用作者AVG，不自行换成四列算术平均 |

注意：InterTab摘要把68.28写成骨干水平，但正文Table 1中68.28属于InternVL3.5-8B，实际Qwen3-VL-8B为67.18。本报告按正文同模型对照计算，不重复摘要的归因错误。

### 对目前约4点提升的判断

- **可称为值得验证的正向结果。**若在冻结的MMTab-eval、多个M、相同评分与预先确定的任务范围下稳定出现约4点提升，完全可能构成有意义的有效性证据。不能因不到某个固定分数就判无效。
- 目前约4点来自历史混合模型路线、开发样本和自定义跨任务宏平均。它不是同一M的配对结果，也不是4%的样本准确率提升，暂不能直接作论文主结论。
- 保留MMTab各benchmark官方指标；另行明确如何从benchmark归并到9类任务。去除T2T的8类平均是预先声明的补充汇总，仍保留全任务结果。FeTaQA属于TQA但按官方使用BLEU，不能因排除T2T就把它改成ACC。
- 以同一图片为簇进行配对bootstrap更合适：一张图片可能对应多道题，不能把每道题都视作完全独立。ACC任务还可报告配对胜负与McNemar检验。F1/TEDS/BLEU需要在每次重采样中按原定义重算汇总值，不能把所有逐样本值随意平均。
- 同时报告耗时、模型调用数、token和失败率。相同M不代表相同计算预算；先完成M与循环的基础对照，再增加预算可比的多次采样/自我修正对照以归因收益。
- 失败样本留在主分母，另列成功运行子集。不得按每个模型分别删除API失败再比较不同样本集合。

## 6. 4×80GB A800怎么安排

本阶段为推理，不是训练。四个角色可复用同一份模型服务，**不需要给每个角色各加载一份权重**。

优先试1卡1个BF16服务、4卡4副本，按同一清单做互斥分片；同一M的直接基线与循环必须使用相同的分片并最终按ID合并。服务端可能同时接收文本和图像请求。7B权重通常约十几GB，但图像编码、长HTML、KV cache、并发和CUDA工作区会额外占显存；80GB足以作为合理起点，不是对任意长度/并发不OOM的承诺。

不要先上4卡张量并行；小模型用4副本通常更适合独立样本吞吐，实际仍需在远程测量。低并发起步，记录吞吐与首token时间。LLaVA的短上下文和低分辨率属于模型限制，不可通过静默截断HTML“修复”。

服务接口参考：[vLLM支持模型](https://docs.vllm.ai/en/v0.20.1/models/supported_models/)、[OpenAI兼容服务](https://docs.vllm.ai/en/v0.20.1/serving/openai_compatible_server/)。脚本提供部署入口，当前Windows环境没有A800/vLLM实机，尚未完成GPU服务烟测。

## 7. MMTab-eval与远程数据准备

数据集由用户在远程另行下载，本包不附带数据，也不依赖当前电脑的数据目录。通过`scripts/env.local.sh`配置标注、图片和模型路径，再生成没有参考答案的推理清单；原始标注保留在独立评估流程中。

[作者仓库](https://github.com/SpursGoZmy/Table-LLaVA)将MMTab-eval描述为约49K问答、23K表格图像，涵盖held-in与held-out评估。任务汇总为TQA、TFV、T2T、TR、TCE、TCL、RCE、TSD、MCD九类；这不等于只有九个benchmark。保留dataset_name与OOD标签，具体样本数以下载版本生成的清单为准。

本包默认准备所有任务；若主实验排除T2T，使用独立清单预先固定，保留完整任务结果。旧200条MMTab-instruct开发集没有迁入正式测试清单。对不同模型采用同一清单和图片版本。
