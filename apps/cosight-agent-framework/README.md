# Co-Sight Agent Framework (Sanitized)

This folder contains the core framework files used for the Co-Sight based Tianchi run, with secrets removed.

## Included Files
- `run_tianchi_first2_with_agent_logs.py`
- `run_tianchi_full_parallel10_guarded.py`
- `export_sequence_mermaid.py`
- `app/manus/tool/search_toolkit.py`
- `config/config.py`
- `.env.example`

## Current Strategy Snapshot
- Enforce per-question time limit (`--timeout 600` in benchmark runs).
- Use Serper for search retrieval.
- Use Jina Reader to enrich top Serper results with page-level details.
- Keep guard counters for Google/Jina failures and runtime monitoring.

## Run (example)
```bash
cd apps/cosight-agent-framework
python run_tianchi_full_parallel10_guarded.py \
  --serper-api-key "<YOUR_SERPER_API_KEY>" \
  --jina-api-key "<YOUR_JINA_API_KEY>" \
  --parallel 20 \
  --timeout 600
```

## Security Note
- No real API key is stored in this directory.
- Do not commit populated `.env` files.
