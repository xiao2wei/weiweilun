# 最终实现、验证与合理性审查

审查日期：2026-07-13。本文只报告当前源码和修复后重新生成的 `artifacts/final-v11` 实际结果；`final-v10` 及更早目录均为历史产物，不作为当前源码证据。`synthetic_fixture_only` 仅用于工程测试；`numerical_simulation/frozen_numerical_model` 可用于指定冻结数值总体下的论文仿真实验，但不是实体硬件、真实道路或真实人脸攻击测量。

## 最终结论

在用户允许以数值模型替代缺失车辆、RSU、车载计算单元、无线环境、FER 模型和攻击器的边界内，`方案1.md` 要求的研究仿真代码已经闭合：连续真实时间 DES、集中状态机、有限资源、联合事务与配对 FER、双向净无线服务、失败/重试/回退、原子 admission、三风险 hard mask、确定性修复、H=1 安全 Lyapunov、H>1 ESL-SMPC、六个公平策略、数值证据生成、审计、指标、manifest、恢复、批处理和统计入口均有可运行实现与测试。

当前 manifest 记录 31 个内容寻址源码文件，源码树 SHA-256 为 `110d9f98a97876871526552bb635350fc05d7a677b4944e67a339f20da4d5b01`。当前目录不是 Git 工作树，因此没有虚构 Git commit。

`final-v11` 冻结输入身份为：

- profile：`7a9176b997ff4f2d2a4908eb560bf80c3916b4ecca4dfe2c3702ee5628583c44`
- evaluation trace：`c0f89985a6866f199e0fbe5b74f095585a9b9c626525360e79cb496a95494ea5`
- scenario trace：`2a68c640310d0cb031324dc05ac0f2af9994e4ecc9c74698b04cdfdee9caaab5`
- 冻结 evidence：`9eb6c79730fec09f781c255f91c9da6deb02b04c8bdb2a239031916437afa04a`
- 规范化配置：`9e3ae296e860682ae186e02b59612363ac19ca62b0c5034dc4db9a1918e982d1`

## 实际执行的验证命令

```powershell
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m compileall -q src tests

python -m privacy_edge_sim.cli generate-numerical-study --output-root artifacts/final-v11/numerical-base --seed 714 --profile-subjects 256 --test-subjects 24 --scenario-subjects 24 --tasks 12 --horizon 12 --privacy-threshold 0.35 --anon-time-variability-scale 1 --output-size-variability-scale 1
python -m privacy_edge_sim.cli validate-profile --profile artifacts/final-v11/numerical-base/profiles/numerical_profile.json
python -m privacy_edge_sim.cli validate-trace --trace artifacts/final-v11/numerical-base/traces/numerical_evaluation_trace.json --profile artifacts/final-v11/numerical-base/profiles/numerical_profile.json
python -m privacy_edge_sim.cli validate-trace --trace artifacts/final-v11/numerical-base/traces/numerical_scenario_trace.json --profile artifacts/final-v11/numerical-base/profiles/numerical_profile.json
python -m privacy_edge_sim.cli validate --config artifacts/final-v11/numerical-base/configs/numerical_default.json

python -m privacy_edge_sim.cli run-all --config artifacts/final-v11/numerical-base/configs/numerical_default.json --output-root artifacts/final-v11/run-all
python -m privacy_edge_sim.cli smoke --config configs/default.json --output-root artifacts/final-v11/smoke-default
python -m privacy_edge_sim.cli smoke --config artifacts/final-v11/numerical-base/configs/numerical_default.json --output-root artifacts/final-v11/smoke-numerical-h1

python -m privacy_edge_sim.cli run --config artifacts/final-v11/numerical-base/configs/numerical_default.json --policy esl_smpc --output artifacts/final-v11/replay-source --checkpoint artifacts/final-v11/replay-checkpoint.json --checkpoint-every 20
python -m privacy_edge_sim.cli run --config artifacts/final-v11/numerical-base/configs/numerical_default.json --policy esl_smpc --output artifacts/final-v11/replay-resumed --resume-checkpoint artifacts/final-v11/replay-checkpoint.json

python -m privacy_edge_sim.cli audit-hard-mask --actions artifacts/final-v11/run-all/fixed_safe_lowest_link_cost/actions.jsonl --output artifacts/final-v11/hard-mask-audit.json
python -m privacy_edge_sim.cli build-one-shot-commitments --actions artifacts/final-v11/run-all/fixed_safe_lowest_link_cost/actions.jsonl --plan examples/one-shot-ready-priority-numerical.json --output artifacts/final-v11/one-shot-commitments.json
python -m privacy_edge_sim.cli audit-two-stage --actions artifacts/final-v11/run-all/fixed_safe_lowest_link_cost/actions.jsonl --commitments artifacts/final-v11/one-shot-commitments.json --output artifacts/final-v11/two-stage-audit.json
python -m privacy_edge_sim.cli audit-failure-integrity --tasks artifacts/final-v11/run-all/fixed_safe_lowest_link_cost/tasks.csv --actions artifacts/final-v11/run-all/fixed_safe_lowest_link_cost/actions.jsonl --events artifacts/final-v11/run-all/fixed_safe_lowest_link_cost/events.jsonl --output artifacts/final-v11/failure-integrity-audit.json
python -m privacy_edge_sim.cli numerical-evidence-report --evidence artifacts/final-v11/numerical-base/evidence/numerical_study_evidence.json --subject-counts 16,32,64,128 --seed 714 --resamples 100 --output artifacts/final-v11/numerical-evidence-report-r100.json

python -m privacy_edge_sim.cli multi-seed --config configs/default.json --policy safe_lyapunov_h1 --seeds 41,42 --output-root artifacts/final-v11/multi-seed-small
python -m privacy_edge_sim.cli sweep --config configs/default.json --policy safe_lyapunov_h1 --grid examples/sweep-small.json --output-root artifacts/final-v11/sweep-small
python -m privacy_edge_sim.cli run-numerical-study --base-study-root artifacts/final-v11/numerical-base --environment-seeds 901,902 --policies all_local,safe_lyapunov_h1 --baseline all_local --metrics all_task_loss,failure_rate,coverage --bootstrap-resamples 100 --permutations 100 --output-root artifacts/final-v11/paired-study-small
python -m privacy_edge_sim.cli aggregate --inputs artifacts/final-v11/run-all artifacts/final-v11/multi-seed-small artifacts/final-v11/sweep-small artifacts/final-v11/paired-study-small --output artifacts/final-v11/aggregate-all.csv
```

## 测试与 smoke 结果

- 全量测试：`382 passed in 149.82s`，0 failed。修复前的同轮第一次全量回归曾发现“无限暂停被当成非有限 expiry”这一处失败（380 passed / 1 failed）；修复后新增明确回归并重新执行全套，最终结果才计为通过。
- Ruff lint：通过；Ruff format check：57 files already formatted；`compileall`：通过。
- profile、evaluation trace、scenario trace 和 config 跨对象校验全部通过；两份 trace 各含 12 个 arrival 和 288 条联合匿名事务，evidence 绑定通过。
- 六策略共 72 个任务、66 DONE、6 FAIL，1,122 次运行中不变量检查、0 失败。每个策略均为 11 DONE / 1 `PREP_FAILED`；这是小规模入口验证，不代表策略等价或优劣结论。
- 默认 synthetic smoke：`all_local` 为 7/8 DONE；H=1 主方法两次均为 3/8 DONE，核心 digest 均为 `8087679285f9f9e556ae68fdb56c789ac02ac8dbfda70c3299e6e7371678947f`。
- numerical smoke：`all_local` 为 11/12 DONE；H=1 两次均为 11/12 DONE，核心 digest 均为 `fb05245f60f30175a74a291bdb6be992011850361f1bab404ed22f8cf86b92b3`。
- ESL-SMPC checkpoint 原始运行与恢复运行均为 11/12 DONE、140 次不变量检查，核心 digest 同为 `f8cff08d85f52a869f47e6f0c3560f3a3b77091a9fcd2869f1b8012bf2c53794`。checkpoint 含 140 个复合事件，`contains_raw_or_aligned_payload=false`。
- hard-mask 离线审计覆盖 22 次 mask、91 个拒绝动作，其中 31 个在忽略安全时看似成本更低；实际执行不安全动作数为 0，hard mask 未绕过。
- 两阶段审计比较 11 个 task pair，只比较 allowed action；失败完整性审计状态为 `COMPLETE`，覆盖 12 个任务、102 条动作和 318 条事件，并量化失败 attempt、retry、下行和 RSU 能耗。
- 小规模批处理实际完成 2 个普通 seed、8 个扫描 case、2 个独立环境 × 2 个策略的 4 次配对运行；最终聚合器复验并汇总 20 个 run。两环境统计只验证入口，样本过小，不作显著性或算法优势结论。

## 数学对象与程序对象

完整映射见 `docs/ARCHITECTURE_AND_ASSUMPTIONS.md`。主要对应为：宏事件 $t_\kappa$ → `EventQueue`/`DiscreteEventSimulator`；任务状态 $x_n$ → `TaskRecord`/`TaskStateMachine`；有限物理队列 → `ResourcePool`；联合匿名事务 $\Xi_{n,p}$ → `AnonTraceRow`；硬安全集合 → `HardMaskEngine`；修复算子 → `DeterministicRepair`；长期队列 → `VirtualQueueBank`；H=1 → `SafeLyapunovPolicy`；H>1 → `ESLSMPCPolicy`、`ScenarioLibrary` 和隔离预测状态。

## 模型一致性审查

- `RAW_BUF/PREP_WAIT/PREP_RUN/RAW`、本地 FER、匿名化/guard/编码、`READY/UL/EDGE_WAIT/EDGE_RUN/DL/DONE/FAIL` 均由集中状态机改变。
- completion 优先于同刻 deadline；deadline 恰好得到有效本地或完整下行结果可成功，晚结果不能成功。
- EventQueue、生产 DES、H>1 分支和 ScenarioLibrary anchor 统一使用有限 8 ULP 的复合时刻与严格未来加法；`0.1+0.2`/`0.3` 合并，而相差 `1e-12` 的物理不同时刻不合并。任务构造、DONE 转移和 DONE invariant 也使用同一判据，因此在 `t=1e9 s` 的大绝对时钟下不会与事件队列矛盾。
- retry 在实际开始 attempt 时计数，最大值闭合；失败前 work、时间和能量保留，所有失败均有重试、本地回退或 FAIL 后继。
- 终态取消活动作业/传输/pending decision，释放车辆和 RSU 资源、删除传输及敏感句柄，并由 `EventQueue.cancel_task()` 物理删除堆中未来 task-owned 事件。同一复合批中已经弹出的同刻事件仍由吸收态和 stale-token 检查无副作用。
- 同刻旧 completion token 不会压制已归零当前版本的 compute/transfer completion；materialization 只复用 current token，否则合成当前版本事件，避免零残余对象残留为 RUNNING。
- `ingress_failed` 是配对 edge trace 的组成部分：上传、admission 和已执行入口成本保留，绝不提交 GPU，随后释放 reservation 并确定性回退或失败；H>1 使用相同语义。
- 未发现未定义后继、无法终止的活动状态、retry off-by-one、晚成功或边缘 GPU 完成即提前 DONE 的路径。

## 物理合理性审查

- 内部单位统一为 s、busy-second、bit、byte、W、J、bit/s 和 `[0,1]`；控制器 wall-clock 仅作诊断。
- 车辆 CPU/accelerator/encoder 与 RSU ingress/GPU 均为有限非抢占服务器；等待来自 enqueue/start/end，不使用“队列长度 ÷ 平均速率”。
- 车辆 reservation 按阶段原子 reconcile：PREP 后释放计算 token、保留可信原始缓冲；READY 后释放 CPU/encoder，只保留匿名包和合法本地回退所需容量。同刻 planned shadow 防止双重承诺。
- 无线仅积分外生应用层 goodput；临时同 RSU 中断暂停恢复，永久失联/切换删除 partial packet；不重复计算无线等待，双侧能量均保留。
- admission 请求的 descriptor/VRAM/work 来自 profile 中对全部 conformal 候选区取上界的 deployment resource certificate；simulator-only 真质量区和 evaluation 实现值不参与请求。完整包到达后才以实际状态原子提交，拒绝零副作用。
- `MODEL_VERSION`/`MODEL_CACHE` 事件入队正 work/energy 的 taskless 非抢占 RSU GPU maintenance；同一 `(RSU, model)` 维护链按先后版本串行，不同模型仍可使用多 GPU 并行；完成时校验 old-version 后才提交 cache/version，能量进入系统物理总能耗而不归因给任务。`final-v11` 每个 run 的 maintenance 为 18 J。
- 联合 trace 的 stage energy 只按 `served_work / total_work` 归因，热降频改变完成时间、idle/hold 能耗和瞬时服务功率，但不再次乘 `dynamic_power_multiplier`；ScenarioLibrary anchor 与 H>1 采用同一配对能耗语义。
- 无限暂停限制表示“不产生 expiry 事件”，不会把 `inf` 作为事件时刻；有限暂停仍由严格未来候选和同刻优先级处理。
- 单调性、热降频、容量收紧、bit/goodput、失败 attempt、下行失败、确定性重放和 admission 零副作用均有受控测试。

## 安全合理性审查

- raw/aligned handle 无 bytes、pickle、JSON 或网络消息序列化路径；唯一匿名请求构造链要求匿名化、guard、编码、profile/model/protocol 证据闭合。
- identity、verification、link 风险分别以主体级同时 UCB 检查，并对所有 conformal 质量候选区取交集；风险只删除动作，不能由时延、能耗或 FER 抵消。
- OOD、unsupported、版本/协议/profile 不匹配、证据/配对缺失一律保守拒绝；repair 重新调用同一 hard mask，不能恢复不安全动作。
- 策略 observation 使用正向白名单，不含身份、标签、攻击真值、evaluation 未来 trace、未来损失或环境 seed。
- evaluation `artifact_key` 和完整配对索引仅在 simulator 可信域；policy、HardMaskEngine 和 estimator 只能看到 task-scoped opaque token 及脱敏的 `(RSU, model, pipeline)` capability。token 跨任务重放被拒绝，RSU 原子 admission 再由 simulator 以真实任务键私下精确复核；策略可达对象图测试包含恶意主体字符串与未来工件键。
- 不变量对原图禁传、容量、电池、版本、终态资源、retry、partial packet、时间、能耗、传输守恒、stale completion 和 admission 副作用 fail-fast。

## 软件与公平性审查

- 模拟器与策略解耦，六策略共享同一任务、物理环境、hard mask、repair、指标、trace 和外生随机性；基线没有额外可见字段。
- 环境、scenario 和统计 RNG 显式分离；稳定排序和内容寻址避免遍历顺序改变外生实现。
- H>1 各候选/场景的可变状态完全隔离，包含 live anchors、入口失败、maintenance GPU 竞争、资源、电池、admission、无线和虚拟队列，只执行第一步。
- 非零 controller overhead 后重新构造当前 hard mask；若原动作失效，H=1/ESL 用各自当前安全动作分数（含刚变为可行的动作）做确定性修复，并记录 `repair_score_source`，不退化成可能改变算法语义的通用排序。
- manifest 绑定代码、配置、profile/evidence/protocol、两份 trace、来源、资源、网络、负载、deadline、控制器、全部 seed、软件环境、仿真时段和不变量；aggregate 会复验 manifest 与 summary hash。
- 没有核心 `TODO`、`NotImplementedError`、固定成功值或为测试关闭安全检查的路径。`safety.py` 中的 `NotImplemented` 是 Python 比较协议的标准返回值，不是未实现功能。

## 默认物理数量级与来源

| 项目 | 默认范围/值 | 单位 | 来源类别 |
|---|---:|---|---|
| 车辆预处理 | 0.021–0.04484 / 0.2415–0.3363 | s / J | engineering_assumption |
| 匿名化 | 0.02987–0.13790 / 0.3616–1.2403 | s / J | engineering_assumption |
| guard / 编码 | 0.008–0.010 / 0.0082–0.00909 | s | engineering_assumption |
| 匿名包 | 54,000–101,000 | byte | engineering_assumption |
| 本地 FER | 0.033–0.0671 / 0.396–0.52338 | s / J | engineering_assumption |
| RSU ingress / GPU | 0.0032 / 0.015–0.02352 | busy-s | engineering_assumption |
| synthetic / numerical maintenance | 0.18 / 0.22；12.6 / 18 | busy-s；J | engineering_assumption |
| goodput | 3.48192–28.32 | Mbit/s | engineering_assumption |
| 无线发/收功率 | 3.2–4.6 / 1.15–1.4 | W | engineering_assumption |
| 车辆 idle / RSU idle | 6.5–8 / 58–72 | W | engineering_assumption |
| deadline | 0.62–1.35 | s | engineering_assumption |
| 热服务倍率 | 0.68 | 1 | stress_test_boundary |

数值研究中的 40 ms telemetry 延迟、0.001 GPU-work-s 量化、每 7 次一次确定性 telemetry 丢失、网络 burst、中断、故障和热变化均是有界工程假设或压力边界，不是虚构实测或文献值。

## 实现假设、差异和证明边界

当前没有实体车辆/RSU、真实攻击器、真实人脸 FER/质量模型或真实测量 trace；按用户确认，它们由冻结数值总体、预注册数值攻击器、数值 FER/质量函数和联合物理 trace 替代。这不是代码缺口，所有替代物都标记 `numerical_simulation`，`real_hardware_measurement=false`，并可通过相同 schema/hash/adapter 接入未来实测对象。

用户明确排除的正式论文规模长时间、多 seed 和大扫描未在本轮执行；本轮已经实际运行小规模多 seed、扫描和独立环境配对统计，证明入口可工作。正式扩容只改变任务量、环境 seed 列表和扫描网格，不需要补写核心代码。

仍不能由有限代码或数值实验严格证明的只有：未知真实总体和未知攻击器上的绝对匿名；单个未来任务 conformal 必然覆盖；真实道路过程满足 Slater、平稳/遍历等无限时域前提；H>1 自动继承 H=1 全部理论保证或全局最优；有限数值 seed 无条件外推到所有真实设备、城市和人群；Python API 边界抵抗恶意 OS、解释器、被篡改源码或侧信道。这些属于严格数学证明、系统形式化安全或外部有效性边界。公平性与观察隔离保证覆盖仓库注册的六个内置策略和官方 CLI；任意外部自定义策略若自行携带额外外部数据，不能由框架自动证明公平。

## 规范文本审计

`论文1.md` 与 `方案1.md` 当前均存在、非空、严格 UTF-8 且无异常控制字符。初始 `方案1.md` 曾含 2 个 U+0008 和 5 个 U+000C；已依据局部 LaTeX 上下文最小、可审计地修复为括号和 `\frac`。修复前后 hash、位置和解释见 `docs/SPEC_AUDIT.md`。

## 用户下一步运行入口

以下命令应从仓库根目录执行，并为每次运行使用新的输出目录；默认拒绝覆盖已有结果。

```powershell
# 单算法：H=1；将 safe_lyapunov_h1 换成 esl_smpc 可运行多步主方法
python -m privacy_edge_sim.cli run --config artifacts/final-v11/numerical-base/configs/numerical_default.json --policy safe_lyapunov_h1 --output results/h1-run-001

# 全部六个策略，共享同一冻结环境
python -m privacy_edge_sim.cli run-all --config artifacts/final-v11/numerical-base/configs/numerical_default.json --output-root results/all-policies-001

# 同一冻结环境下改变控制随机流
python -m privacy_edge_sim.cli multi-seed --config artifacts/final-v11/numerical-base/configs/numerical_default.json --policy esl_smpc --seeds 101,102,103,104,105 --output-root results/esl-multiseed-001

# 独立 environment seed 的正式数值配对研究；关键论文结果建议至少 10，最终宜 20 个环境 seed
python -m privacy_edge_sim.cli run-numerical-study --base-study-root artifacts/final-v11/numerical-base --environment-seeds 101,102,103,104,105,106,107,108,109,110 --baseline all_local --output-root results/numerical-study-10env

# 参数扫描
python -m privacy_edge_sim.cli sweep --config artifacts/final-v11/numerical-base/configs/numerical_default.json --policy esl_smpc --grid examples/sweep-small.json --output-root results/esl-sweep-001

# 汇总；安装 pyarrow 后可追加 --parquet
python -m privacy_edge_sim.cli aggregate --inputs results/all-policies-001 results/esl-multiseed-001 results/esl-sweep-001 --output results/aggregate-001.csv
```

`multi-seed` 只改变控制/采样随机流，不等价于独立物理环境重复；正式论文比较优先使用 `run-numerical-study`。完整 profile/trace 导入、失败定位、checkpoint 恢复、Parquet 和大实验成本说明见根目录 `README.md`。
