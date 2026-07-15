# Paper-v1 数值仿真实验手册

本手册给出从机制 pilot 到正式多环境统计的可执行顺序。参数的机器可读唯一记录是：

- `examples/paper-v1-load-matrix.json`：主实验规模、负载、容量压力和统计族；
- `examples/paper-v1-sensitivity.json`：独立、非确认性的 OFAT 敏感性计划；
- `examples/sweep-*.json`：可直接交给 `sweep` 的控制器扫描网格。

所有结果都是 `numerical_simulation/frozen_numerical_model`，不是硬件实测或真实道路结论。隐私阈值 0.35 是预注册研究边界，不得根据性能结果调大或调小后仍并入主统计族。

## 1. 实验矩阵

| scale | regime | tasks | horizon | arrival window | 名义负载 | local FER scale | RSU 容量 |
|---|---|---:|---:|---:|---:|---:|---|
| pilot | default | 48 | 12 s | 4.0 s | 12 tasks/s | 1 | 默认有限容量 |
| pilot | burst | 48 | 12 s | 1.6 s | 30 tasks/s | 1 | 默认有限容量 |
| pilot | compute_pressure | 48 | 12 s | 1.6 s | 30 tasks/s | 3 | 默认有限容量 |
| pilot | capacity_pressure | 48 | 12 s | 1.6 s | 30 tasks/s | 3 | 每 RSU 一个保守 edge reservation |
| formal | default | 240 | 20 s | 16.0 s | 15 tasks/s | 1 | 默认有限容量 |
| formal | burst | 240 | 20 s | 8.0 s | 30 tasks/s | 1 | 默认有限容量 |
| formal | compute_pressure | 240 | 20 s | 8.0 s | 30 tasks/s | 3 | 默认有限容量 |
| formal | capacity_pressure | 240 | 20 s | 8.0 s | 30 tasks/s | 3 | 每 RSU 一个保守 edge reservation |

`compute_pressure` 与 `burst` 只差本地 FER 工作量及其配对动态能耗；`capacity_pressure` 与 `compute_pressure` 只差已登记的 RSU override。pilot seeds 为 201–205，formal seeds 为 301–330，敏感性 seeds 为 401–420，三者互不重叠。

30 tasks/s 是小规模机制 probe 后冻结的工程压力点：60 tasks/s 导致大量 buffer rejection，无法区分控制策略；30 tasks/s 已观测到有限队列等待、匿名化路径和 edge completion，但仍必须先通过下面的 pilot gate。

## 2. 安装、测试和源码冻结门禁

在仓库根目录执行：

```powershell
python -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -e ".[test,parquet]"
if ($LASTEXITCODE -ne 0) { throw 'dependency installation failed' }
& $py -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'test suite failed' }
& $py -m ruff check src tests
if ($LASTEXITCODE -ne 0) { throw 'Ruff failed' }
```

正式 `run-numerical-study` 和 `sweep` 默认要求 `src/privacy_edge_sim` 与当前 Git commit 完全一致，并在每次运行结束时再次检查。study 还会逐文件验证 `paper-v1-load-matrix.json` 已提交；正式 sweep 同时验证 sensitivity 登记文件，并把登记内容哈希写入报告。批处理会在注册时冻结 base config/profile/evaluation/scenario/evidence 的字节身份，逐 environment/case 和发布前复核；复制后的 profile/scenario/evidence 语义及允许变动的 RNG seed 集也会再次比较。完成本轮代码审查后先提交，再执行：

```powershell
$py = ".\.venv\Scripts\python.exe"
git status --short
& $py -m privacy_edge_sim.cli source-preflight
if ($LASTEXITCODE -ne 0) { throw 'source preflight failed' }
```

若门禁失败，不要开始正式实验。`--allow-dirty-source` 只允许用于本地调试或废弃的 pilot；带该开关的结果不得进入论文统计。

每个输出目录应使用新名称。不要在正式批次上使用 `--overwrite`；中断的 `run-numerical-study` 当前不能原位续跑，应保留故障目录取证并换新目录重跑。单次 `run` 的 replay checkpoint 能恢复逻辑前缀，但会重新计算该前缀，不能代替正式批次级断点续跑。

## 3. 生成四个 pilot 冻结包

先定义一个只封装已冻结公共选项的 PowerShell 函数：

```powershell
$py = ".\.venv\Scripts\python.exe"

function New-PaperBundle {
  param(
    [string]$OutputRoot,
    [int]$Tasks,
    [double]$Horizon,
    [double]$Center,
    [double]$Window,
    [double]$LocalScale
  )
  & $py -m privacy_edge_sim.cli generate-numerical-study `
    --output-root $OutputRoot `
    --seed 20260713 `
    --profile-subjects 256 --test-subjects 64 --scenario-subjects 48 `
    --tasks $Tasks --horizon $Horizon `
    --arrival-center-s $Center --arrival-window-s $Window `
    --arrival-jitter-fraction 0.15 `
    --privacy-threshold 0.35 `
    --preprocessing-failure-mode none `
    --local-service-scale $LocalScale `
    --anon-time-variability-scale 1 `
    --output-size-variability-scale 1
  if ($LASTEXITCODE -ne 0) { throw "bundle generation failed: $OutputRoot" }
}

New-PaperBundle artifacts\paper-v1\pilot\default 48 12 6 4.0 1
New-PaperBundle artifacts\paper-v1\pilot\burst 48 12 6 1.6 1
New-PaperBundle artifacts\paper-v1\pilot\compute_pressure 48 12 6 1.6 3
New-PaperBundle artifacts\paper-v1\pilot\capacity_pressure 48 12 6 1.6 3
```

只对容量压力包应用已登记 override。`derive-config` 拒绝修改 profile/trace/evidence 路径、拒绝未知点路径，并在原子替换前运行完整配置校验：

```powershell
$capacityConfig = 'artifacts\paper-v1\pilot\capacity_pressure\configs\numerical_default.json'
& $py -m privacy_edge_sim.cli derive-config `
  --config $capacityConfig `
  --overrides examples\paper-v1-load-matrix.json `
  --section capacity_pressure_overrides.config_values `
  --output $capacityConfig --overwrite
if ($LASTEXITCODE -ne 0) { throw 'pilot capacity config derivation failed' }

Get-ChildItem artifacts\paper-v1\pilot -Recurse -Filter numerical_default.json |
  ForEach-Object {
    & $py -m privacy_edge_sim.cli validate --config $_.FullName
    if ($LASTEXITCODE -ne 0) { throw "invalid config: $($_.FullName)" }
  }
```

## 4. 运行 pilot

一批一批运行，先观察耗时再继续。四策略 pilot 同时覆盖本地参照、edge-using 固定基线、H=1 理论核心和 H>1 主方法：

```powershell
$py = ".\.venv\Scripts\python.exe"
$pilotSeeds = '201,202,203,204,205'
$pilotPolicies = 'all_local,fixed_safe_shortest_visible_queue,safe_lyapunov_h1,esl_smpc'
$pilotMetrics = 'all_task_loss,failure_rate,coverage,latency_p95_s,energy_j.task_attributed.total,edge_done_rate,pipeline_attempt_rate,pipeline_to_edge_rate,pipeline_to_local_rate'

foreach ($regime in 'default','burst','compute_pressure','capacity_pressure') {
  & $py -m privacy_edge_sim.cli run-numerical-study `
    --base-study-root "artifacts\paper-v1\pilot\$regime" `
    --environment-seeds $pilotSeeds `
    --policies $pilotPolicies --baseline all_local `
    --metrics $pilotMetrics `
    --bootstrap-resamples 1000 --permutations 5000 `
    --load-level $regime --family-id paper-v1-pilot-mechanism `
    --experiment-registration examples\paper-v1-load-matrix.json `
    --registration-scale pilot --registration-regime $regime `
    --registration-family pilot `
    --output-root "results\paper-v1-pilot\$regime"
  if ($LASTEXITCODE -ne 0) { throw "pilot failed: $regime" }
}
```

在资源有限时，可先运行 `default` 和 `compute_pressure`，再运行 `capacity_pressure`，最后运行 `burst`。不要并行启动多个 ESL-SMPC 批次，除非已确认 CPU、内存和磁盘吞吐有余量。

## 5. Pilot gate 与失败成本审计

先检查所有 run manifest 的持续不变量：

```powershell
$manifests = Get-ChildItem results\paper-v1-pilot -Recurse -Filter manifest.json
$actions = Get-ChildItem results\paper-v1-pilot -Recurse -Filter actions.jsonl
$studyFiles = Get-ChildItem results\paper-v1-pilot -Recurse -Filter study_report.json
if (@($manifests).Count -ne 80) { throw "expected 80 pilot manifests, found $(@($manifests).Count)" }
if (@($actions).Count -ne 80) { throw "expected 80 pilot action logs, found $(@($actions).Count)" }
if (@($studyFiles).Count -ne 4) { throw "expected 4 pilot reports, found $(@($studyFiles).Count)" }
$bad = foreach ($file in $manifests) {
  $m = Get-Content $file.FullName -Raw -Encoding utf8 | ConvertFrom-Json
  if (-not $m.invariants.passed -or $m.invariants.failure_count -ne 0) {
    $file.FullName
  }
}
if ($bad) { $bad; throw 'invariant gate failed' }
```

再检查 `study_report.json`。至少一个压力 regime 必须同时出现实际等待和 edge completion：

```powershell
$qualifiedPressureRegimes = foreach ($regime in 'burst','compute_pressure','capacity_pressure') {
  $path = "results\paper-v1-pilot\$regime\study_report.json"
  $report = Get-Content $path -Raw -Encoding utf8 | ConvertFrom-Json
  $waitValues = @()
  $edgeValues = @()
  foreach ($policy in $report.mechanism_diagnostics.by_policy.PSObject.Properties) {
    $waitValues += $policy.Value.metrics.'resources.max_waiting_jobs'.max
    $edgeValues += $policy.Value.metrics.edge_done_rate.max
  }
  $maxWait = ($waitValues | Measure-Object -Maximum).Maximum
  $maxEdge = ($edgeValues | Measure-Object -Maximum).Maximum
  if ($maxWait -gt 0 -and $maxEdge -gt 0) { $regime }
}
if (-not $qualifiedPressureRegimes) {
  throw 'no single pressure regime exercised both waiting and edge completion'
}
```

聚合失败成本审计默认验证每个 manifest 自哈希，以及 `tasks.csv`、`actions.jsonl`、`events.jsonl` 的 manifest 绑定哈希。容量压力必须实际观测到 admission reject；若没有，正确做法是报告机制未激活并修改下一版预注册压力设计，不能把它写成已验证：

```powershell
$py = ".\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force results\paper-v1-pilot-audits | Out-Null

& $py -m privacy_edge_sim.cli audit-failure-coverage `
  --study-roots results\paper-v1-pilot `
  --require-categories uplink,admission_accept,rsu_ingress,edge_execution,rsu_attributed_energy,downlink `
  --output results\paper-v1-pilot-audits\core-path-coverage.json
if ($LASTEXITCODE -ne 0) { throw 'pilot core-path coverage gate failed' }

& $py -m privacy_edge_sim.cli audit-failure-coverage `
  --study-roots results\paper-v1-pilot\capacity_pressure `
  --require-categories admission_reject,edge_execution,rsu_attributed_energy `
  --output results\paper-v1-pilot-audits\capacity-failure-coverage.json
if ($LASTEXITCODE -ne 0) { throw 'capacity failure coverage gate failed' }

$capacityAudit = Get-Content results\paper-v1-pilot-audits\capacity-failure-coverage.json -Raw -Encoding utf8 | ConvertFrom-Json
$observedReasons = @($capacityAudit.observed_coverage.admission_reject.reason_counts.PSObject.Properties.Name)
$capacityReasons = @('RSU_DESCRIPTOR_CAPACITY','RSU_VRAM_CAPACITY','RSU_WORKLOAD_CAPACITY')
if (-not @($observedReasons | Where-Object { $_ -in $capacityReasons })) {
  throw "admission rejects were not caused by registered capacity limits: $($observedReasons -join ',')"
}
```

对每个 actions 文件运行 hard-mask 审计：既重算反事实成本，也把 RAW/READY 的实际策略动作及执行时修复与同次执行前的 `EXECUTION_RECHECK` mask 配对；默认还验证相邻 manifest、actions 哈希、clean source 和零失败不变量。故障后的 `FROZEN_LOCAL_FALLBACK` 是状态机自动后继，不属于 RAW/READY 动作集合：它由 `_local_fallback_feasible`、运行期不变量和专项测试验证，并在报告中单独计数，不能把本审计解读为对所有自动后继的独立重放证明。输出放在独立目录，避免污染 run leaf：

```powershell
$auditRoot = New-Item -ItemType Directory -Force results\paper-v1-pilot-audits\hard-mask
$i = 0
$pilotActions = Get-ChildItem results\paper-v1-pilot -Recurse -Filter actions.jsonl
if (@($pilotActions).Count -ne 80) { throw "expected 80 hard-mask inputs, found $(@($pilotActions).Count)" }
$pilotActions | ForEach-Object {
  $i++
  & $py -m privacy_edge_sim.cli audit-hard-mask `
    --actions $_.FullName `
    --output (Join-Path $auditRoot.FullName ('audit-{0:D4}.json' -f $i))
  if ($LASTEXITCODE -ne 0) { throw "hard-mask audit failed: $($_.FullName)" }
}
```

只有以下条件全部成立才进入 formal：

1. 所有 manifest 的 invariant failure 为 0；
2. hard-mask 审计全部成功，RAW/READY 策略动作与执行修复没有绕过安全门；自动 failure successor 的专项测试和不变量同时通过；
3. 至少一个压力 regime 的等待非零；
4. 至少一个压力 regime 的 edge completion 非零；
5. `capacity_pressure` 中 edge-using 策略实际出现原子 admission reject；
6. admission reject 的 reason code 确实属于 descriptor、VRAM 或 workload 容量；
7. `capacity_pressure` 与 `compute_pressure` 的差异只包含矩阵登记的 override（正式 study 启动前会重建 canonical config 并自动拒绝额外修改）；
8. pilot 与 formal 的 seeds、目录和统计 family 完全分离。

失败审计对 admission 只证明“观测到结构完整的 accept/reject 记录”，原子无副作用本身由运行期不变量与单元测试验证；有限 pilot 不能构成一般数学证明。

## 6. 生成 formal 冻结包

Pilot gate 通过后，重新生成而不是复制 pilot 包：

```powershell
$py = ".\.venv\Scripts\python.exe"
function New-PaperBundle {
  param([string]$OutputRoot,[int]$Tasks,[double]$Horizon,[double]$Center,[double]$Window,[double]$LocalScale)
  & $py -m privacy_edge_sim.cli generate-numerical-study `
    --output-root $OutputRoot --seed 20260713 `
    --profile-subjects 256 --test-subjects 64 --scenario-subjects 48 `
    --tasks $Tasks --horizon $Horizon `
    --arrival-center-s $Center --arrival-window-s $Window `
    --arrival-jitter-fraction 0.15 --privacy-threshold 0.35 `
    --preprocessing-failure-mode none --local-service-scale $LocalScale `
    --anon-time-variability-scale 1 --output-size-variability-scale 1
  if ($LASTEXITCODE -ne 0) { throw "bundle generation failed: $OutputRoot" }
}

New-PaperBundle artifacts\paper-v1\formal\default 240 20 10 16 1
New-PaperBundle artifacts\paper-v1\formal\burst 240 20 10 8 1
New-PaperBundle artifacts\paper-v1\formal\compute_pressure 240 20 10 8 3
New-PaperBundle artifacts\paper-v1\formal\capacity_pressure 240 20 10 8 3

$capacityConfig = 'artifacts\paper-v1\formal\capacity_pressure\configs\numerical_default.json'
& $py -m privacy_edge_sim.cli derive-config `
  --config $capacityConfig `
  --overrides examples\paper-v1-load-matrix.json `
  --section capacity_pressure_overrides.config_values `
  --output $capacityConfig --overwrite
if ($LASTEXITCODE -ne 0) { throw 'formal capacity config derivation failed' }

Get-ChildItem artifacts\paper-v1\formal -Recurse -Filter numerical_default.json |
  ForEach-Object {
    & $py -m privacy_edge_sim.cli validate --config $_.FullName
    if ($LASTEXITCODE -ne 0) { throw "invalid formal config: $($_.FullName)" }
  }
```

提交最终实验记录与配置后再次执行 `source-preflight`，并保存 commit ID：

```powershell
git rev-parse HEAD
& $py -m privacy_edge_sim.cli source-preflight
```

## 7. 正式主要比较

主要假设固定为四个 regime 中 `esl_smpc` 相对 `all_local` 的 `all_task_loss`。统计单位是独立 environment，30 个 seeds 为 301–330；同一 environment 内任务不能当作独立重复。

```powershell
$py = ".\.venv\Scripts\python.exe"
$formalSeeds = (301..330) -join ','

foreach ($regime in 'default','burst','compute_pressure','capacity_pressure') {
  & $py -m privacy_edge_sim.cli run-numerical-study `
    --base-study-root "artifacts\paper-v1\formal\$regime" `
    --environment-seeds $formalSeeds `
    --policies all_local,esl_smpc --baseline all_local `
    --metrics all_task_loss `
    --statistics-seed 91001 `
    --bootstrap-resamples 5000 --permutations 20000 `
    --load-level $regime --family-id paper-v1-primary-v2 `
    --experiment-registration examples\paper-v1-load-matrix.json `
    --registration-scale formal --registration-regime $regime `
    --registration-family primary `
    --output-root "results\paper-v1-formal\primary\$regime"
  if ($LASTEXITCODE -ne 0) { throw "formal primary failed: $regime" }
}

& $py -m privacy_edge_sim.cli aggregate-statistical-families `
  --study-reports `
    results\paper-v1-formal\primary\default\study_report.json `
    results\paper-v1-formal\primary\burst\study_report.json `
    results\paper-v1-formal\primary\compute_pressure\study_report.json `
    results\paper-v1-formal\primary\capacity_pressure\study_report.json `
  --output results\paper-v1-formal\primary\family-report.json
if ($LASTEXITCODE -ne 0) { throw 'formal primary family aggregation failed' }
```

不要根据 pilot 中“看起来最好”的 regime 删除其他主假设；跨 load、metric、policy 的检验族由 family report 统一 Holm 校正。

## 8. 全部基线与次要指标

次要 family 与主要 family 分开，仍使用相同 formal seeds、冻结包、hard mask 和外生随机性：

```powershell
$py = ".\.venv\Scripts\python.exe"
$formalSeeds = (301..330) -join ','
$policies = 'all_local,fixed_safe_lowest_link_cost,fixed_safe_shortest_visible_queue,safe_greedy,safe_lyapunov_h1,esl_smpc'
$metrics = 'failure_rate,coverage,latency_p95_s,energy_j.task_attributed.total,edge_done_rate,pipeline_attempt_rate,pipeline_to_edge_rate,pipeline_to_local_rate'

foreach ($regime in 'default','burst','compute_pressure','capacity_pressure') {
  & $py -m privacy_edge_sim.cli run-numerical-study `
    --base-study-root "artifacts\paper-v1\formal\$regime" `
    --environment-seeds $formalSeeds `
    --policies $policies --baseline all_local `
    --metrics $metrics `
    --statistics-seed 92001 `
    --bootstrap-resamples 5000 --permutations 20000 `
    --load-level $regime --family-id paper-v1-secondary-v2 `
    --experiment-registration examples\paper-v1-load-matrix.json `
    --registration-scale formal --registration-regime $regime `
    --registration-family secondary `
    --output-root "results\paper-v1-formal\secondary\$regime"
  if ($LASTEXITCODE -ne 0) { throw "formal secondary failed: $regime" }
}

& $py -m privacy_edge_sim.cli aggregate-statistical-families `
  --study-reports `
    results\paper-v1-formal\secondary\default\study_report.json `
    results\paper-v1-formal\secondary\burst\study_report.json `
    results\paper-v1-formal\secondary\compute_pressure\study_report.json `
    results\paper-v1-formal\secondary\capacity_pressure\study_report.json `
  --output results\paper-v1-formal\secondary\family-report.json
if ($LASTEXITCODE -ne 0) { throw 'formal secondary family aggregation failed' }
```

主要与次要命令会重复计算 `all_local`/`esl_smpc`，这是为了保持两个预注册统计族独立且不把次要指标混入主要多重检验。若算力不足，先完成主要 family，再分 regime 完成次要 family。

## 9. 参数扫描与敏感性

`examples/sweep-small.json` 是 2×2×2 的机制压力 factorial smoke，不是正式 OFAT：

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m privacy_edge_sim.cli sweep `
  --config artifacts\paper-v1\formal\compute_pressure\configs\numerical_default.json `
  --policy esl_smpc `
  --grid examples\sweep-small.json `
  --output-root results\paper-v1-sensitivity\factorial-smoke `
  --allow-dirty-source
```

上面的 factorial 命令显式标记为开发机制 smoke，不进入正式敏感性结果。正式控制器 OFAT 必须每次只使用一个网格文件，以 `burst` 为登记参考，并使用与主实验分离的 401–420 环境族。CLI 会同时核对已提交 sensitivity 计划、主矩阵、factor levels、参考配置和 evaluation seed；临时网格、错误 regime 或未登记配置改动会在运行前拒绝。

```powershell
$py = ".\.venv\Scripts\python.exe"
$matrix = 'examples\paper-v1-load-matrix.json'
$sensitivity = 'examples\paper-v1-sensitivity.json'
$factors = @(
  @{ Name='privacy_policy_boundary'; Grid='examples\sweep-privacy-boundary.json' },
  @{ Name='lyapunov_tradeoff'; Grid='examples\sweep-lyapunov-v.json' },
  @{ Name='mpc_horizon'; Grid='examples\sweep-mpc-horizon.json' },
  @{ Name='scenario_count'; Grid='examples\sweep-scenario-count.json' },
  @{ Name='rsu_admission_concurrency'; Grid='examples\sweep-rsu-admission-concurrency.json' }
)

foreach ($seed in 401..420) {
  $replication = "artifacts\paper-v1\sensitivity\env-$seed"
  & $py -m privacy_edge_sim.cli generate-numerical-replication `
    --base-study-root artifacts\paper-v1\formal\burst `
    --output-root $replication --environment-seed $seed
  if ($LASTEXITCODE -ne 0) { throw "sensitivity replication failed: $seed" }

  foreach ($factor in $factors) {
    & $py -m privacy_edge_sim.cli sweep `
      --config "$replication\configs\numerical_default.json" `
      --policy esl_smpc --grid $factor.Grid `
      --sensitivity-registration $sensitivity `
      --registration-factor $factor.Name `
      --experiment-registration $matrix `
      --output-root "results\paper-v1-sensitivity\registered\env-$seed\$($factor.Name)"
    if ($LASTEXITCODE -ne 0) { throw "sensitivity sweep failed: seed=$seed factor=$($factor.Name)" }
  }
}

& $py -m privacy_edge_sim.cli aggregate `
  --inputs results\paper-v1-sensitivity\registered `
  --output results\paper-v1-sensitivity\registered-aggregate.csv --parquet
if ($LASTEXITCODE -ne 0) { throw 'sensitivity aggregation failed' }

# 正式探索性统计：每个 environment 只计一个独立单位；同一 seed 内各 level 配对。
& $py -m privacy_edge_sim.cli analyze-sensitivity `
  --sweep-roots results\paper-v1-sensitivity\registered `
  --sensitivity-registration examples\paper-v1-sensitivity.json `
  --metric all_task_loss --statistics-seed 92001 `
  --bootstrap-resamples 2000 --permutations 20000 `
  --output results\paper-v1-sensitivity\registered-analysis.json
if ($LASTEXITCODE -ne 0) { throw 'paired sensitivity analysis failed' }
```

`registered-aggregate.csv` 仅用于逐行浏览，不是敏感性推断入口。`analyze-sensitivity`
要求每个 factor 恰好包含登记的 401–420 环境、每个环境具有完全相同且无缺失的
level、同一环境内 evaluation/scenario trace 严格配对、所有 manifest/summary 哈希有效，
并且运行代码和登记文件均来自同一个 clean commit。报告以登记的 `reference_value`
为基线，给出 level 减 reference 的环境级均值、environment-cluster bootstrap CI、
双侧 sign-flip 检验以及 factor 内 Holm 校正。不同 factor 之间不再做一次确认性 Holm；
它们始终标记为 exploratory、non-confirmatory，且不进入论文主/次统计 family。

每个 sweep 输出 `sweep.json` 和 `sweep_diagnostics.json`。后者记录配置坐标是否产生可观察的路径、性能或资源变化；“未观察到变化”是描述性结果，不证明参数在一般情形下无关。每个坐标必须有至少两个不同 level。中断会保留 `sweep.in_progress.json`；聚合与失败审计还会校验完成索引、case 集和 manifest 一一对应，删除一个 case 或索引同样会被拒绝。使用新目录重跑或明确取证后再决定是否 `--overwrite`。

401–420 是 20 个独立 environment；同一 environment 内的多个 level 是配对 case，不能当作独立重复。该 sensitivity family 明确为探索性、非 confirmatory，扁平 aggregate 用于按 `parameter.*` 和 evaluation seed 分组报告全部 level，不并入主/次 Holm family。

隐私阈值在本计划中是已登记的策略边界扫描：每个 level 仍使用同一冻结 profile，并重新执行完整 hard mask；它不是为了追求更好性能而调参。到达 burst 与本地计算压力已经由主矩阵中的 `default`/`burst`/`compute_pressure` 配对对照覆盖，因此不再重复放入这一 OFAT family。不得在线修改 profile、跨 pipeline 插值或复用测试标签。H=1 始终由 `safe_lyapunov_h1` 策略比较，不把 ESL 的 `horizon_events` 设为 1 后冒充理论版本。

## 10. 汇总与查看结果

机器可读扁平汇总：

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m privacy_edge_sim.cli aggregate `
  --inputs results\paper-v1-formal\primary results\paper-v1-formal\secondary `
  --output results\paper-v1-formal\aggregate.csv --parquet
if ($LASTEXITCODE -ne 0) { throw 'flat aggregation failed' }
```

该扁平文件只用于逐行或按 `analysis_family_id`/family 输出目录分组诊断。主要和次要 family 都包含 `all_local` 与 `esl_smpc`，因此禁止对整张合并表直接计算总均值、总样本数或推断统计；论文结论应分别读取两个 `family-report.json`。

聚合器会验证每个 manifest 自哈希和 summary 哈希，输出同名 CSV/JSON，并在安装 `pyarrow` 时输出 Parquet。常用检查：

```powershell
Import-Csv results\paper-v1-formal\aggregate.csv |
  Select-Object policy,all_task_loss,coverage,failure_rate,latency_p95_s

Get-ChildItem results\paper-v1-formal -Recurse -Filter tasks.csv |
  ForEach-Object { Import-Csv $_.FullName } |
  Group-Object failure_reason | Sort-Object Count -Descending

Get-ChildItem results\paper-v1-formal -Recurse -Filter actions.jsonl |
  Select-String 'PRIVACY_RISK|OOD|VERSION_MISMATCH|UNSUPPORTED'
```

关键结果以 `study_report.json`/`family-report.json` 的环境级效应量、cluster bootstrap CI、双侧 sign-flip p 值和 Holm 校正为准，不以单个 seed、任务级标准误或 controller wall-clock 作推断。

## 11. 预计成本与运行边界

实测 48-task H=1 pilot 的 controller diagnostic wall-clock 合计约 105 s；四个轻量策略 probe 约 175 s。包含 ESL 的高压 probe 在 244 s 内未完成，因此正式运行应按 regime 分批。240-task 单环境 H=1/ESL 粗略估计约 9–15 分钟或更久；30 seeds 的一个主要 regime 可能需要 8–15 小时，四个 regime 加全部基线可能持续数日。以上只是当前机器上的工程估算，不进入仿真时间或论文性能指标。

有限数值实验可以验证代码路径、物理单调性、指定数值总体下的经验效果和所观测失败成本；不能证明绝对匿名、未知攻击器安全、conformal 单任务必然覆盖、真实道路 Slater 条件、无限时域稳定性、H>1 理论继承或全局最优。
