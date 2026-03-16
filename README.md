# Tianchi 2026 AI Deep Research Agent Solution

> 本仓库是**QSeek团队**在2026天池Deep Research Agent比赛中的**季军🥉**解决方案。

题目背景：题目要求参赛者基于qwen系列模型构建一个能够自主搜索互联网、推理并回答复杂问题的Research Agent。比赛提供了一组难度从「简单」到「困难」的问题，Agent 需要在有限时间内通过多轮搜索、交叉验证、逐步推理来推理出最终答案。

核心思路是ReAct单循环架构，整体方案基于 MiroThinker的Agent框架思路继续迭代，并针对天池赛题的多跳检索、证据验证、跨语言实体规范化、长上下文稳定性和时限约束做了专项优化。

## 目录
- [项目简介](#项目简介)
- [整体方案](#整体方案)
- [核心优化](#核心优化)
  - [1. 分层上下文隔离](#分层上下文隔离)
  - [2. 主动上下文监控与熔断](#主动上下文监控与熔断)
  - [3. 基于回滚的脏数据清洗](#基于回滚的脏数据清洗)
  - [4. 重复查询过滤](#重复查询过滤)
  - [5. 基于近因的工具结果压缩](#基于近因的工具结果压缩)
  - [6. 天池场景下的额外适配](#天池场景下的额外适配)
- [仓库结构](#仓库结构)
- [环境安装](#环境安装)
- [数据与配置](#数据与配置)
- [运行方式](#运行方式)
- [实验结果](#实验结果)
- [当前主推配置](#当前主推配置)


## 整体方案

整体流程如下：

```text
问题理解
  -> 拆解缺失证据
  -> 搜索候选来源
  -> 页面级抓取与抽取
  -> 多轮验证与候选排除
  -> 本地化/标准译名检查
  -> 最终短答案生成
```

## 核心优化

### 1. 分层上下文隔离

这是 Orchestrator 最关键的内存管理机制。系统采用主智能体和子智能体分层执行：

- 当主 Agent 遇到复杂子问题时，不在当前 `message_history` 中硬撑，而是调用 `run_sub_agent` 开启子任务。
- 子 Agent 拥有一份全新的独立上下文，只携带子任务描述，不继承主 Agent 的冗余历史。
- 子 Agent 可能执行几十轮搜索、抓取和推理，但返回给主 Agent 的只有最终结果，而不是完整过程日志。

这样做的本质是把“大量中间推理 token”压缩成“一个高价值结论”，从而显著降低主上下文膨胀速度。对于长链路、多页面、多候选实体的问题，这个机制非常重要。

### 2. 主动上下文监控与熔断

我们没有被动等待模型因上下文溢出而失败，而是在每轮结束后主动检查剩余预算：

- 结合模型 `max_context_length` 做上下文空间估计。
- 在接近预算上限时触发压缩或提前进入收束阶段。
- 为最终总结和本地化校验预留独立时间窗口，避免“前面搜索太久，最后来不及输出”。

这套机制的目标是把失败前移为可控退化，让 Agent 在有限时间和上下文预算内优先保证“有证据的可交付答案”。

### 3. 基于回滚的脏数据清洗

Agent 执行过程中不可避免会出现工具调用格式错误、模型幻觉、页面抽取失败或超时等问题。如果把这些错误轨迹长期保留在上下文中，会同时造成两个问题：浪费 token，以及诱导模型重复犯同类错误。

因此我们在 Orchestrator 中引入了基于 rollback 的清洗策略：

- 当某一轮输出明显异常时，当前轮次不进入稳定记忆。
- 通过 `consecutive_rollbacks` 记录连续回滚次数，避免无限重试。
- 在大多数情况下，模型看不到自己刚刚产生的脏轨迹，只保留恢复后的有效上下文。

这相当于把错误尝试视为“事务失败”，默认不提交到长期记忆，从而保持上下文尽可能干净。

### 4. 重复查询过滤

Research Agent 很容易在复杂问题上陷入搜索死循环，例如反复搜索同一组关键词、只做轻微改写、或者围绕错误候选不断打转。

为此系统加入了查询去重与重复检测：

- 使用 `used_queries` 记录已经调用过的查询及其参数。
- 对重复或近重复搜索做拦截，迫使 Agent 改变检索角度。
- 配合 prompt 中的“每轮必须重写 query list”规则，避免追加式、惯性式搜索。

这个机制直接减少了无效调用，也让有限的工具预算更多用于真正能带来新证据的搜索路径。

### 5. 基于近因的工具结果压缩

长网页抓取结果是上下文膨胀的主要来源。我们的做法不是粗暴截断整个历史，而是优先保留“推理链条”，压缩“过时原始观察”：

- 通过 `keep_tool_result` 控制保留最近多少条工具结果，当前常用设置为 `5`。
- 在构造发送给 LLM 的消息时，保留思考与动作轨迹。
- 对更早的工具返回内容，用类似 `Tool result is omitted to save tokens.` 的占位文本替换。

背后的假设是：如果较早的页面信息确实重要，它应该已经被模型吸收并体现在后续 thought 中；因此没有必要无限保留原始长文本。

### 6. 天池场景下的额外适配

除以上通用机制外，我们还针对天池赛题做了几项定制：

- 强化 evidence-first prompt，要求先明确答案类型、语言和最小证据集，再发起搜索。
- 限制每轮搜索 query 数量，并强制从不同角度重写，减少无效枚举。
- 加入 localization gate，在最终总结前专门检查人名、机构名、标题等实体的中文/英文标准形式。
- 增加 web tool hard limit、超时预留和最终总结保底逻辑，提升整体完赛率。
- jina爬取网页内容中，让LLM按照线索主动对网页内容进行寻找和总结，很大程度增强了网页内容的利用度，也减少了无效的网页内容在context中的占比

### 7. 总结：

- 物理隔离：通过子 Agent 承担高成本子任务。
- 动态修剪：通过 rollback 清洗错误轨迹。
- 近因保留：保留最近有效观察，压缩早期长文本。
- 重复抑制：防止搜索行为陷入循环。
- 安全底线：通过上下文和时间预算检查避免任务在最后阶段崩溃。

## 仓库结构

```text
.
├── apps/miroflow-agent
│   ├── conf/                  # LLM、Agent、Benchmark 配置
│   ├── src/                   # Agent 核心实现
│   ├── jsonl_inference/       # 批量推理与结果回填
│   ├── scripts/               # 天池验证集运行脚本
│   └── tests/                 # 单测
├── libs/miroflow-tools        # 搜索、抓取等工具层
├── logs/                      # 实验日志与推理结果
└── README.md
```

几个关键文件：

- `apps/miroflow-agent/src/core/orchestrator.py`：主/子 Agent 协调、回滚、上下文管理、localization gate。
- `apps/miroflow-agent/conf/agent/mirothinker_v1.5_keep5_max200_tianchi.yaml`：当前天池主配置。
- `apps/miroflow-agent/scripts/run_jsonl_inference_tianchi-validation.sh`：批量推理入口脚本。

## 环境安装

建议使用 `uv` 和 Python `3.12+`。

```bash
cd apps/miroflow-agent
uv sync
```

常用环境变量按实际 provider 配置，例如：

```bash
export API_KEY=YOUR_LLM_API_KEY
export BASE_URL=YOUR_LLM_BASE_URL
export SERPER_API_KEY=YOUR_SERPER_KEY
export JINA_API_KEY=YOUR_JINA_KEY
```

如果使用脚本中的默认参数，也可以额外指定：

```bash
export LLM_CONFIG=qwen-3
export LLM_PROVIDER=qwen
export LLM_MODEL=qwen3.5-plus
export AGENT_SET=mirothinker_v1.5_keep5_max200_tianchi
```

## 数据与配置

默认天池验证配置位于：

- `apps/miroflow-agent/conf/benchmark/tianchi-validation.yaml`
- `apps/miroflow-agent/conf/agent/mirothinker_v1.5_keep5_max200_tianchi.yaml`

默认输入数据路径为：

```text
data/tianchi/standardized_data.jsonl
```

天池配置中的一些关键设置：

- `max_turns: 200`
- `web_tool_call_hard_limit: 100`
- `keep_tool_result: 5`
- `context_compress_limit: 5`
- `localization_gate_enabled: true`

## 运行方式

### 运行天池验证集

```bash
cd apps/miroflow-agent
bash scripts/run_jsonl_inference_tianchi-validation.sh
```

### 常见自定义参数

```bash
cd apps/miroflow-agent
INPUT_JSONL=../../data/tianchi/standardized_data.jsonl \
LLM_CONFIG=qwen-3 \
LLM_PROVIDER=qwen \
LLM_MODEL=qwen3.5-plus \
BASE_URL=YOUR_BASE_URL \
API_KEY=YOUR_API_KEY \
AGENT_SET=mirothinker_v1.5_keep5_max200_tianchi \
MAX_CONTEXT_LENGTH=262144 \
MAX_CONCURRENT=10 \
bash scripts/run_jsonl_inference_tianchi-validation.sh
```

结果默认输出到：

```text
logs/tianchi-validation/<provider>_<model>_<agent_set>/<run_id>/
```

其中通常包含：

- `final_answers.jsonl`
- `benchmark_results.jsonl`
- `task_runtimes.jsonl`
- `summary_time_cost.json`
- 每题对应的 `task_*.json` 日志

## 实验结果

| # | Model | Tool(s) | Score | Improvement | Comment |
|---|---|---|---:|---|---|
| 1 | deepseek-chat | google-search, jina scrape | 0.6162 | baseline from previous validation setting | |
| 2 | qwen3-max | none | 0.2929 | no external tools | |
| 3 | qwen3-max | google-search | 0.3535 | basic search integration | |
| 4 | qwen3-max | google-search (search-only) | 0.5051 | search-only tool policy, MCP tool-call parsing hardening, qwen thinking log | local run, 97/100 completed |
| 5 | qwen3.5-plus | google-search (Serper only) | 0.5152 | stage-1 efficiency prompt, tool routing auto-repair, web tool hard-limit, rollback/minimal-context retry, timeout/final-summary reserve | full run, 100/100 completed |
| 6 | qwen3.5-plus | google-search, jina scrape | 0.5758 | stronger page-level extraction and evidence verification | full run, 100/100 completed |

对应日志目录：

1. `logs/gaia-validation/deepseek_deepseek-chat_mirothinker_v1.5_keep5_max200/run_20260206_110146`
2. `logs/tianchi-validation/qwen_qwen3-max_mirothinker_v1.5_keep5_max200`
3. `logs/tianchi-validation/qwen_qwen3-max_mirothinker_v1.5_keep5_max200_tianchi`
4. `apps/miroflow-agent/logs/tianchi-validation/qwen_qwen3-max_mirothinker_v1.5_search_only_keep5_max200/fg_full_probe_20260216_154936`
5. `logs/tianchi-validation/qwen_qwen3.5-plus_mirothinker_v1.5_keep5_max200/v5_serper_newkey_full_20260219_082009`
6. `logs/tianchi-validation/qwen_qwen3.5-plus_mirothinker_v1.5_keep5_max200_tianchi/run_20260220_104009_score0.57`

## 当前主推配置

如果你想直接复现当前仓库的主线方案，建议优先使用：

- Model: `qwen3.5-plus`
- Agent config: `mirothinker_v1.5_keep5_max200_tianchi`
- Tools: `google_search + jina scrape`

这套配置相对平衡了：
- 页面级证据获取能力
- 长任务的上下文稳定性
- 中文实体规范化表现
- 完赛率与平均耗时