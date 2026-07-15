# Paper-v1 数值实验矩阵

`examples/paper-v1-load-matrix.json` 是本实验的参数记录；它不是 `run-numerical-study --config` 的输入。生成器参数通过 `generate-numerical-study` 的命令行选项传入，随后每个冻结研究包由 `run-numerical-study` 生成配对的独立环境。

## 设计边界

- 三个负载条件均固定 24 个任务、12 s horizon、同一 profile/evidence seed 和 0.35 隐私阈值；因此改变的是到达密度，而不是任务数、必然预处理失败次数或隐私预算。
- 低、中、高负载的 arrival window 分别是 4.0、1.0、0.6 s，中心均为 6.0 s。它们对应名义总到达率 6、24、40 tasks/s。
- `arrival_jitter_fraction=0.15` 是相对相邻名义到达间隔的比例，不是秒；它用于去除完全规则的到达并保持任务顺序。
- `preprocessing_failure_mode=none` 使主负载实验不会因“最后一个任务固定失败”而把 task count 混入失败率。预处理故障鲁棒性应另立实验。
- `local_service_scale=1.0` 是未修改的基线。若机制校准选择了预先定义的 1.5 或 2.0，本矩阵中的三个条件必须**一起**更新为同一数值，并在运行正式种子前冻结该 JSON 文件。

## 先做机制校准（不进入正式统计）

在中负载下，以独立环境种子 `201,202,203,204,205` 分别测试 `local_service_scale=1.0,1.5,2.0`。只检查 hard-mask 审计、失败完整性和 `edge_done_rate`；不比较性能优劣。选择第一个使 ESL-SMPC 出现非零且不过度集中的 EDGE 完成比例（建议 10%--80%）并且没有安全违规的尺度。校准种子和输出不得混入正式结果。

例如，生成一个候选中负载研究包：

```powershell
$py = ".\.venv\Scripts\python.exe"

& $py -m privacy_edge_sim.cli generate-numerical-study `
  --output-root artifacts\paper-v1-calibration\medium-scale-1p5 `
  --seed 20260713 `
  --profile-subjects 256 `
  --test-subjects 64 `
  --scenario-subjects 48 `
  --tasks 24 --horizon 12 `
  --arrival-center-s 6 --arrival-window-s 1 `
  --arrival-jitter-fraction 0.15 `
  --privacy-threshold 0.35 `
  --preprocessing-failure-mode none `
  --local-service-scale 1.5

& $py -m privacy_edge_sim.cli run-numerical-study `
  --base-study-root artifacts\paper-v1-calibration\medium-scale-1p5 `
  --environment-seeds 201,202,203,204,205 `
  --policies all_local,esl_smpc `
  --baseline all_local `
  --metrics edge_done_rate,pipeline_attempt_rate,pipeline_to_edge_rate,pipeline_to_local_rate `
  --load-level calibration-medium `
  --family-id paper-v1-calibration `
  --output-root results\paper-v1-calibration\medium-scale-1p5
```

校准完成后，重新生成正式包；不要把校准包复制或重命名为正式包。

## 冻结三个正式研究包

以下示例以校准选出的 `local_service_scale=1.5` 为例。若选择了其他尺度，只替换三个命令中的该值，保持其他参数完全相同。

```powershell
$py = ".\.venv\Scripts\python.exe"

& $py -m privacy_edge_sim.cli generate-numerical-study `
  --output-root artifacts\paper-v1\low `
  --seed 20260713 --profile-subjects 256 --test-subjects 64 --scenario-subjects 48 `
  --tasks 24 --horizon 12 `
  --arrival-center-s 6 --arrival-window-s 4 --arrival-jitter-fraction 0.15 `
  --privacy-threshold 0.35 --preprocessing-failure-mode none `
  --local-service-scale 1.5

& $py -m privacy_edge_sim.cli generate-numerical-study `
  --output-root artifacts\paper-v1\medium `
  --seed 20260713 --profile-subjects 256 --test-subjects 64 --scenario-subjects 48 `
  --tasks 24 --horizon 12 `
  --arrival-center-s 6 --arrival-window-s 1 --arrival-jitter-fraction 0.15 `
  --privacy-threshold 0.35 --preprocessing-failure-mode none `
  --local-service-scale 1.5

& $py -m privacy_edge_sim.cli generate-numerical-study `
  --output-root artifacts\paper-v1\high `
  --seed 20260713 --profile-subjects 256 --test-subjects 64 --scenario-subjects 48 `
  --tasks 24 --horizon 12 `
  --arrival-center-s 6 --arrival-window-s 0.6 --arrival-jitter-fraction 0.15 `
  --privacy-threshold 0.35 --preprocessing-failure-mode none `
  --local-service-scale 1.5
```

## 正式主要检验

主要假设只比较 `esl_smpc` 与 `all_local` 的 `all_task_loss`，在中、高负载条件上进行。30 个 environment seeds 为 `301..330`，统计单位始终是 environment。两个研究报告用同一个 family id，最后统一做 Holm 校正。

```powershell
$seeds = (301..330) -join ','

& $py -m privacy_edge_sim.cli run-numerical-study `
  --base-study-root artifacts\paper-v1\medium `
  --environment-seeds $seeds `
  --policies all_local,esl_smpc --baseline all_local `
  --metrics all_task_loss `
  --bootstrap-resamples 5000 --permutations 20000 `
  --load-level medium --family-id paper-v1-primary `
  --output-root results\paper-v1-primary\medium

& $py -m privacy_edge_sim.cli run-numerical-study `
  --base-study-root artifacts\paper-v1\high `
  --environment-seeds $seeds `
  --policies all_local,esl_smpc --baseline all_local `
  --metrics all_task_loss `
  --bootstrap-resamples 5000 --permutations 20000 `
  --load-level high --family-id paper-v1-primary `
  --output-root results\paper-v1-primary\high

& $py -m privacy_edge_sim.cli aggregate-statistical-families `
  --study-reports results\paper-v1-primary\medium\study_report.json results\paper-v1-primary\high\study_report.json `
  --output results\paper-v1-primary\family-report.json
```

低负载用于负对照和机制呈现，不纳入上述主优效性检验。

## 次要结果、基线与审计

在每个负载条件上另行运行全部六种策略，并注册延迟、任务归因能耗、失败率、覆盖率及四个机制比例。将三个报告以 `paper-v1-secondary` family id 聚合；这些结果不能替代主要检验。

```powershell
$metrics = 'all_task_loss,failure_rate,coverage,latency_p95_s,energy_j.task_attributed.total,edge_done_rate,pipeline_attempt_rate,pipeline_to_edge_rate,pipeline_to_local_rate'
$policies = 'all_local,fixed_safe_lowest_link_cost,fixed_safe_shortest_visible_queue,safe_greedy,safe_lyapunov_h1,esl_smpc'

foreach ($level in 'low','medium','high') {
  & $py -m privacy_edge_sim.cli run-numerical-study `
    --base-study-root "artifacts\paper-v1\$level" `
    --environment-seeds $seeds `
    --policies $policies --baseline all_local `
    --metrics $metrics `
    --bootstrap-resamples 5000 --permutations 20000 `
    --load-level $level --family-id paper-v1-secondary `
    --output-root "results\paper-v1-secondary\$level"
}

& $py -m privacy_edge_sim.cli aggregate-statistical-families `
  --study-reports results\paper-v1-secondary\low\study_report.json results\paper-v1-secondary\medium\study_report.json results\paper-v1-secondary\high\study_report.json `
  --output results\paper-v1-secondary\family-report.json
```

对每个 ESL-SMPC 输出再运行 `audit-hard-mask`、`audit-two-stage` 和 `audit-failure-integrity`；审计失败时不应解释性能统计。隐私阈值 0.10 在当前冻结 evidence 下可作为“无可行匿名路径”的边界报告，不应作为与 0.35 的优效性比较。
