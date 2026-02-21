# Deep Research Agent for Tianchi2026 Challenge

## Tricks
### Done
**Prompt优化：**
- WIP...
- 
  
**Context/Memory优化：**
- 上下文隔离：当主Agent遇到复杂问题时，希望不要自己在当前的上下文里死磕，而是新开一个sub_agent。
- 主动监控上下文状态：每一轮结束后主动检查剩余context，防止上下文爆炸。
- Rollback机制：Agent在运行过程中出错（如工具调用格式错误、幻觉、网络超时）信息自动回退，保证上下文干净。
- 重复查询过滤：防止 Agent 陷入死循环（反复搜索同一个词），引入记忆去重机制，避免了无效的重复信息占用上下文。
- 智能压缩：保留推理链条但丢弃过时数据。

**其他优化：**
- WIP...
- 
### TODO
- 后处理：json-repair防止输出为非json结构，LLM最后的输出检查
- 时间限制优化：限制10min内，时间优化（越短越好）
- Ensemble策略：运行多次进行ensemble，但是要考虑时间问题！！
- tool calling优化：什么时候用google，什么时候用国内搜索api，fallback机制/策略


## Experiments

| # | Model         | Tool(s)                     | Score  | Improvement | Comment |
|---|---------------|-----------------------------|--------|-------------|---------|
| 1 | deepseek-chat | google-search, jina scrape  | 0.6162 | -           |         |
| 2 | qwen3-max     | None                        | 0.2929 | -           |         |
| 3 | qwen3-max     | google-search               | 0.3535 | -           |         |
| 4 | qwen3-max     | google-search (search-only) | 0.5051 | search-only tool policy (blacklist sogou_search); MCP tool-call parsing hardening (`_normalize_mcp_name` + malformed `<use_mcp_tool>` fallback); qwen `enable_thinking` + reasoning log | local run, 97/100 completed (last 3 manually stopped) |
| 5 | qwen3.5-plus  | google-search (Serper only) | 0.5152 | stage-1 efficiency prompt skill; tool routing auto-repair; web tool hard-limit; data-inspection rollback/minimal-context retry; timeout/final-summary reserve | full run, 100/100 completed |
| 6 | qwen3.5-plus  | google-search, jina scrape  | 0.5758 | enhance jina scrape usage by prompts | full run, 100/100 completed |

Log Path：
1. logs/gaia-validation/deepseek_deepseek-chat_mirothinker_v1.5_keep5_max200/run_20260206_110146
2. logs/tianchi-validation/qwen_qwen3-max_mirothinker_v1.5_keep5_max200
3. logs/tianchi-validation/qwen_qwen3-max_mirothinker_v1.5_keep5_max200_tianchi
4. apps/miroflow-agent/logs/tianchi-validation/qwen_qwen3-max_mirothinker_v1.5_search_only_keep5_max200/fg_full_probe_20260216_154936
5. logs/tianchi-validation/qwen_qwen3.5-plus_mirothinker_v1.5_keep5_max200/v5_serper_newkey_full_20260219_082009
6. logs/tianchi-validation/qwen_qwen3.5-plus_mirothinker_v1.5_keep5_max200_tianchi/run_20260220_104009_score0.57

Run Note:
- local run completed 97 tasks before manual stop
- average runtime from task_runtimes.jsonl: 00:17:44
