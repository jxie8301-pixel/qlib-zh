# rd-agent 20 轮因子挖掘脚本

- `scripts/run_rdagent_20_rounds.sh`: 在容器内运行 20 轮 `rdagent fin_factor`，并在完成后收集结果。
- `scripts/collect_rdagent_rounds.py`: 从 `log/<run>/Loop_*` 中提取每轮假设、反馈和优化方向，生成独立 round 文件夹。

输出位置：
- `DATA/analysis_outputs/rdagent_20round_<timestamp>/round_01` ... `round_20`
- 每轮包含：
  - `analysis_report.md`
  - `summary.json`
  - `hypothesis_strings.txt`
  - `feedback_strings.txt`
  - `runner_strings.txt`
- 顶层包含：
  - `leaderboard.md`
  - `container_run.log`
  - `bigmodel_full_requests.log`
  - `bigmodel_debug.log`
