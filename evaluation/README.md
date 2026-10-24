# 评估来源与预测导出

`official/`保留Table-LLaVA作者公开评估notebook、IBM TEDS实现及相关辅助文件。原始来源：[Table-LLaVA/MMTab-eval_evaluation](https://github.com/SpursGoZmy/Table-LLaVA/tree/main/MMTab-eval_evaluation)。

新鲜直连下载失败，因此使用此前已核对的官方源码缓存，来源与哈希见`official/PROVENANCE.json`，不声称与当前远程HEAD完全相同。源码未修改；未把历史200条的答案修正规则、人工标签或临时评分补丁当成官方公式。

导出：

```bash
python3 agent_circle.py export --run RUN_NAME --output predictions.jsonl
```

按清单输出`question_id`、`text`和`model_id`，空答案保留缺失/失败项。两个模型必须按完全相同ID集合比较。

数据集在远程另行下载。按官方notebook组织预测、`MMTab-eval_test_data_49K.json`与`MMTab-eval_test_tables_23K.json`后评分；图片路径由远程配置指定。本包不依赖当前电脑的数据目录，也不附带测试数据。不要从预测HTML反造gold表。此任务没有实际执行新评分。

官方指标边界：TQA/TFV按各benchmark规定，FeTaQA是BLEU；T2T采用BLEU；TSD分行列ACC；TCE/TCL为ACC；RCE分行/列F1；MCD为F1；TR分HTML/Markdown/LaTeX的TEDS。先保存逐benchmark结果，再声明如何汇总为9类任务及去T2T的8类任务平均。

可补充统一答案等价评估，但必须与官方严格指标并列、规则预先固定，不能只对循环结果宽松处理。`answer`是外层协议字符串，不自动意味着其中的任务答案符合官方表示。
