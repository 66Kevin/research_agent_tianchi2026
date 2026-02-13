# Deep Research Agent for Tianchi2026 Challenge

## Experiments
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
