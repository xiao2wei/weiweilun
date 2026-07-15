# 架构、数学映射与实现边界

本文描述当前代码的实际语义。实现优先级为 `方案1.md > 论文1.md > 明确记录的数值仿真工程假设`。`numerical_simulation` 是冻结数值总体上的研究仿真，不是实测硬件、真实人脸模型或绝对匿名证据。

## 方案数学对象到代码对象

| `方案1.md` 数学/语义对象 | 主要代码对象 | 已实现语义 |
|---|---|---|
| 连续宏事件时刻、复合事件 $t_\kappa$ | `events.EventQueue`、`simulator.DiscreteEventSimulator` | 时钟直接跳至下一事件；同刻事件按 `(priority, seq)` 稳定处理，不使用固定时间步 |
| 完整 SMDP 状态 $S_\kappa$ | `state.SimulationState`、`TaskRecord`、`VehicleRuntime`、`RSURuntime` | 任务、资源、传输、电池、无线/热状态、RSU admission 和虚拟队列均进入可审计状态 |
| 策略可见信息 $O_\kappa$ | `safety.Observation`、`ObservationBuilder` | allow-list 构造；不包含真实身份、真实 artifact key、表情标签、攻击真值、实际 FER 损失、真实质量区或 evaluation 未来 trace；配对仅暴露任务作用域 opaque token |
| 任务状态 $x_n(t),\sigma_n(t)$ | `state.TaskRecord`、`TaskStateMachine`、`LEGAL_SUCCESSORS` | 状态只由集中状态机改变；`DONE/FAIL` 吸收；合法前驱、后继和终态清理闭合 |
| RAW/READY 动作 $a_n^{\rm RAW},a_n^{\rm READY}$ | `safety.Action`、`HardMaskEngine` | RAW 支持 local/pipeline/fail，READY 支持 edge/local/fail；所有删除均带结构化 reason code |
| 硬安全集合 $\mathcal A_\kappa^{\rm hard}$ | `profiles.FrozenProfileBundle.query_privacy`、`safety.HardMaskEngine` | 隐私、OOD、支持、版本、联合配对、资源、电池、连接、快照和 deadline 先硬过滤，隐私不进入可权衡的软奖励 |
| 执行前修复 $R(a,x)$ | `safety.DeterministicRepair` | 控制开销结束后用当前实际状态重建 hard mask；只能稳定选择仍安全动作、本地回退或失败 |
| 车辆物理队列 $Q_{v,k}^V$ | `resources.ResourcePool` | CPU、accelerator、encoder 为有限服务器；非抢占 EDF；等待、启动、完成时间分开记录；队列量使用残余 busy-seconds |
| RSU 队列 $Q_{r,k}^R$ | `RSURuntime.ingress`、`RSURuntime.gpu` | 入口 CPU 与 GPU 分离，服务器有限；服务率变化按残余 work 连续推进，不以“长度/均值速率”近似等待 |
| RSU 容量/admission 集合 | `resources.RSUAdmission`、`AdmissionRequest`、profile deployment resource certificate | 请求资源由 profile 对全部 conformal 候选区取保守上界，禁止使用 simulator-only $g^*$ 或 evaluation 实现值；完整 UL 抵达后按实际状态原子 check-then-commit descriptors/VRAM/work/cache/protocol；拒绝零副作用；接纳后 pin 模型哈希 |
| 匿名事务 $\Xi_{n,p}$ 与顺序尝试 $\boldsymbol\zeta_{n,p}$ | `traces.AnonTraceRow`、`AnonAttempt`、`TraceBundle` | attempt、阶段 work/energy、guard、编码、OOM、大小、artifact 及同 artifact FER 整行加载和整体重放 |
| Edge FER 联合实现 | `traces.EdgeFERTraceRow`、`ingress_failed` | 入口结果、入口/GPU/DL work/energy、FER 结果和 artifact/context 保持配对；入口失败不另抽概率，保留已耗成本且绝不进入 GPU |
| 上下行残余比特 $b_n^{\rm UL},b_n^{\rm DL}$ | `state.Transfer`、`physics.py` | 以净 goodput 对真实时间积分；累计交付与完成一致；双侧功率/能量配对；没有第二份平均速率时延 |
| 真实质量区 $g_n^*$ 与候选集 $\mathcal G_n=C_\alpha(u_n)$ | `TaskRecord.true_quality_region`、`conformal_quality_bins`、数值 quality evidence | $g^*$ 仅用于离线证据/评估；在线只暴露冻结估计器输出和 conformal 候选；管线须对候选区交集全部安全 |
| 主体级风险变量 $X_{ih},Y_{ih}$ 与同时 UCB | `profiles.compute_subject_risk_ucb`、数值 evidence 的 `subject_rows/stage_statistics` | 主体是独立单位；identity/verification/link 分开；Bonferroni-Hoeffding 比值界、支持数和 emission LCB 均进入 hard mask |
| 长期虚拟队列 $Z_q$ | `state.VirtualQueueBank` | 车辆/RSU 平均功率、timeout、failure、coverage 按真实事件间隔和实现结果更新并记录轨迹 |
| H=1 漂移—成本比 | `policies.SafeLyapunovPolicy` | 只比较 hard-safe 动作；分母为真实持续时间；包含虚拟队列、物理 residual workload 和冻结归一化 |
| H>1 场景 MPC | `policies.ESLSMPCPolicy`、`ScenarioLibrary`、`ScenarioEnvironment` | 使用 training/validation 联合场景、相对事件和完整 anchor；各动作/场景分支隔离，rollout 后只执行第一步，下一宏事件重规划 |
| 两阶段基线 | `AllLocalPolicy`、两种 `FixedSafe...Policy`、`SafeGreedyPolicy`、`SafeLyapunovPolicy`、`ESLSMPCPolicy` | 共享模拟器、任务、外生 trace、hard mask、repair 和指标；策略只读取同一 observation 边界 |
| 复现与审计 | `MetricLedger`、`manifest.py`、`checkpoint.py` | 任务/网络/RSU/能耗/过滤/修复/队列/利用率输出；manifest 绑定配置、profile、trace、证据和种子；checkpoint 通过前缀哈希确定性重放 |

## 连续时间事件语义

一次循环执行如下闭合顺序：

1. 从堆中弹出下一个时间戳的全部原子事件，并把所有活动 compute/link/battery/hold 物理量精确推进至该时刻；
2. 物化零残余 compute/transfer completion；
3. 处理 fault、link/mobility、thermal、telemetry 和 battery guard；`MODEL_VERSION/MODEL_CACHE` 到达只向有限 RSU GPU 入队 taskless maintenance，版本/cache 在维护 completion 时才原子提交；
4. 处理尚未完成任务的 deadline；
5. 处理 arrival；
6. 提交已到期的确定性 controller-overhead 事件；
7. 派发空闲有限资源、构造新决策、再次派发零控制开销刚产生的作业，并安排下一 completion/energy guard。

因此 completion 明确优先于同刻 deadline；恰好在绝对 deadline 收到有效本地结果或完整 DL 结果可进入 `DONE`。事件队列用有限 ulp 等价而不是十进制完全相等识别同一数学时刻，并推进到同组最晚可表示时间，使 `0.1+0.2` 与 `0.3` 保持完成优先而不合并物理上可区分的近邻。job、transfer 和 pending decision 都带对象身份或版本 token；已取消、重排、故障或终态后的旧 completion 只能成为 stale event。进入终态时 `EventQueue.cancel_task()` 会物理删除堆中全部未来 task-owned 事件；已弹入当前复合批的同刻事件仍由吸收态/stale 检查无副作用。

控制器本机 `perf_counter` 只写诊断字段。进入仿真时钟和能耗的是冻结配置中的 `controller_overhead_s`、`controller_energy_j`；有非零开销时，动作先进入 pending，提交前再次经过 repair，不能用决策时旧状态直接产生物理副作用。

## 状态机与失败闭合

完整主路径为：

```text
RAW_BUF -> PREP_WAIT -> PREP_RUN -> RAW
RAW -> LOCAL_WAIT -> LOCAL_RUN -> DONE
RAW -> ANON_WAIT -> ANON_RUN -> GUARD_WAIT -> GUARD_RUN
    -> ENCODE_WAIT -> ENCODE_RUN -> READY
READY -> UL -> EDGE_WAIT -> EDGE_RUN -> DL -> DONE
```

匿名化 attempt 计数在匿名作业真正入队时递增，表示“已启动总尝试数”，范围为 `1..max_attempts`，不存在 retry/attempt 的 off-by-one 混用。OOM、匿名化失败、guard 拒绝和编码失败按冻结的 retryable reason、attempt 上限及 fallback 进入下一 `ANON_WAIT`、`LOCAL_WAIT` 或 `FAIL`。UL、admission、RSU、DL、永久失联、设备故障、版本变化和 deadline 都有合法本地回退或失败后继。

终态会原子取消/结算未来事件、活动作业、传输和 pending decision，释放车辆 descriptor/memory 和 RSU reservation，并清除原始/对齐句柄。`DONE` 只允许由有效的 `LOCAL_RUN` 或完整 `DL` 在 deadline 内产生；边缘 GPU 完成但结果尚未下行不算完成。

## 资源、无线与 admission

- 每个 `ResourcePool` 是有限个相同逻辑服务器，等待队列使用 `(absolute_deadline, enqueue_seq, job_id)` 的稳定非抢占 EDF。work 的单位为资源忙时秒；热降频只降低 effective service rate，不会凭空缩短残余服务。
- 车辆 descriptor 和 memory 在选定必经路径时预留并按阶段原子 reconcile：PREP 后释放计算 token、保留可信原始缓冲；READY 后释放 CPU/encoder，只保留匿名包与合法本地回退需要的容量。同刻 planned shadow 使连续决策看到此前承诺，防止双重预留。电池在离散操作前检查，持续 compute/radio 则安排 battery guard；失败前成本不回滚。
- `Transfer` 仅接受类型正确的 `AnonFERRequest`（UL）或 `FERResult`（DL）。短暂同 RSU 中断暂停并保留残片；永久失联/handover 删除旧残片，切换 RSU 时只能从车辆保存的完整匿名包重新上传。
- scheduler 看到的 RSU snapshot 是带采样/交付时刻的不可变副本，不是预留。完整匿名包到达后才按实时 admission 状态检查消息证据、协议、模型 cache 及由 profile/pipeline/model/protocol/scenario-trace 绑定的保守 descriptor、VRAM 和 GPU work certificate。拒绝前后 admission snapshot 必须相同。
- 入口成功才进入有限 GPU；配对 `ingress_failed` 会保留上传、admission 和入口成本，释放 reservation，绝不入 GPU，并确定性本地回退或失败。GPU 产出 FER 结果时释放 inference reservation；下行仍累计双侧无线和任务/系统能量，但不继续占用 GPU/VRAM/work reservation。
- 模型/cache 维护使用冻结正 work/energy 的 taskless 非抢占 GPU job，与 inference 竞争相同服务器；同一 `(RSU, model)` 的事务链只允许 head 派发，后继保留在有限 GPU EDF 队列和 workload 中，即使多 GPU 也不能越过 old-version 前置条件。只有 maintenance completion 才修改 cache/version；不同 model 可正常并行竞争。其能量进入 RSU 系统物理能耗，不归因到任务，H>1 镜像同一顺序与无副作用失败语义。

## 原图类型边界和在线冻结

- `packets.RawImageHandle` 与 `AlignedTensorHandle` 拒绝 bytes、pickle、`__getstate__` 和 JSON 序列化，只能留在车辆可信域。
- `EncodedAnon` 只能沿匿名化 evidence → 同 task/attempt/artifact 的 guard certificate → encoding evidence 私有能力链构造；`AnonFERRequest.from_encoded` 是公开上传构造路径，schema 没有 raw/aligned/payload 旁路字段。
- metric、manifest、checkpoint 和策略 observation 分别有 deny-list/allow-list 检查，网络消息不接受原始表示或其句柄。真实 evaluation `artifact_key` 也不进入 policy 可达对象图：simulator 可信域先把当前任务的真实键投影为脱敏 `(RSU, model, pipeline)` capability，会话内 registry 只保存 capability 与不含源键摘要/子串的任务作用域 token。hard mask 对 capability 做 membership，跨任务 token 重放返回 unsupported；完整包到达后 simulator 再以真实任务键私下精确复核 admission 配对。
- `FrozenAdapterRegistry` 将 component/profile/protocol/deployment hash 和执行域绑定；seal 后只给受控推理代理，并在调用前后检查 descriptor、文件/内存指纹和输入输出类型。profile、质量分区、阈值、归一化和策略参数在线不可修改。
- `HardMaskEngine` 在策略评分前删除不安全动作，`DeterministicRepair` 在物理提交前再次运行同一安全边界。unsupported、OOD、版本/协议不兼容、缺联合 trace、缺同 artifact 配对测量一律保守拒绝。

这些措施在本研究程序的 API 和运行时不变量层面阻止原图上传，但不等同于对恶意修改 Python 解释器、源代码、操作系统或同进程攻击者的形式化隔离证明。

## numerical v2 冻结研究模型

`numerical.py` 的当前 evidence/profile 版本分别为 `numerical-study-2.1.0` / `2.2.0`，trace 版本为 `2.3.0`，管线和攻击器使用 `numerical_v2` 标识。它在离线阶段构造互斥的 attack-train、attack-threshold-calibration、quality-calibration、profile-evaluation、scenario-training-validation 和 test 主体 split；在线仿真只加载冻结结果。Paper-v1 额外将到达中心、到达窗口、相对 jitter、预处理失败模式和本地服务缩放冻结进数值研究 spec 与 trace metadata，避免负载条件隐式改变故障或热时间轴。

数值总体包含四类方法（pixelate、blur、generative、diffusion）和三档强度，共 12 条管线。隐私 v2 的残余身份参数是明确标注、写入 evidence 并受 hash 绑定的工程假设：

| 参数 | 冻结值 |
|---|---|
| method identity retention | pixelate 0.63；blur 0.59；generative 0.45；diffusion 0.38 |
| strength identity retention | weak 1.00；medium 0.60；strong 0.08 |
| quality multiplier | `0.72 + 0.28 * quality_score`，只放大 residual identity |
| temporal persistence | link 权重 0.34、其他风险 0.10，并再次由 residual retention 门控 |

`cosine_gallery_numerical_v2`、`projected_margin_numerical_v2`、`temporal_link_numerical_v2` 是三个预注册数值评分器，不是真实攻击神经网络。v2 强制 attacker fitting 与 deployment 使用同一 observable feature：质量和时间持久性只能放大匿名 artifact 中的残余身份，retention 为零时可观察身份信号也为零。score-model 版本、自哈希、工程假设来源、attacker 参数指纹和 threshold split 均写入 evidence；loader 会重新校验这些绑定。

身份识别按冻结 Rank-1 gallery 规则判定；verification/link 使用各自在独立 threshold-calibration 主体上的阈值。三种攻击器 × 三类风险 × 管线 × 质量区组成完整预注册笛卡尔积，主体级三阶段记录（single attempt、guard selected、guard plus retry final）保存原始 `[attack-and-emit, emission]` 行并可重算 UCB。profile loader 还逐项核对 profile 的隐私 bound 与 evidence，不能只靠改 profile JSON 放宽 hard mask。

### 质量区 $g^*$ 支持语义

数值 evidence 不再把每个 profile-evaluation 主体机械复制到所有质量区。冻结过程为：

1. 对 profile-evaluation 每个主体的每一帧，用冻结 partition 和 reference features 确定唯一真实区域 $g^*$；
2. 主体 $i$ 仅在至少一帧属于 $g$ 时进入该 cell，且 $m_{i,g}$ 只计实际落入 $g$ 的帧；
3. 风险行的分母严格为该主体的 $m_{i,g}$，每帧使用其配对的实际 quality score；
4. stable subject/frame 顺序、subject count、frame count、subject hash 和总 support hash 写入 evidence；
5. `evidence.py` 验证所有 profile-evaluation `(subject, frame)` 恰好归属一个 cell、无跨区重复，并验证每条隐私行与对应 cell 的顺序和计数绑定。

如果某个冻结质量区没有任何 profile-evaluation frame，数值生成器直接失败，要求增加支持数据；它不会生成一个貌似可用的空统计 cell。运行时对缺 cell/缺配对仍返回 `unsupported`。这是对方案“支持不足时删除动作”的保守工程表达。

在线的 `conformal_quality_bins` 来自带噪冻结质量估计器和 split conformal；`true_quality_region` 只存在于 simulator-only 评估/admission 配对检查。`FrozenProfileBundle.query_privacy` 对候选集合逐 cell 检查三类风险、主体数和发射覆盖，并取最坏 UCB；任何候选区不安全即删除整条 pipeline。

## H=1、H>1 与公平性边界

H=1 的分数使用冻结成本尺度、真实 action duration、长期虚拟队列以及车辆/RSU residual busy-second 工作量；隐私动作已在评分前被删除。非零 controller overhead 后若原动作失效，执行门在提交时对当前 hard-safe 集调用策略纯评分接口：H=1 重算当前漂移—成本比，H>1 重算隔离场景 rollout；之前 unsafe、现在 newly-safe 的动作也会获得分数，策略诊断/RNG/真实状态在重算后恢复。H>1 从单独 training/validation `ScenarioLibrary` 获取身份无关、相对时间的无线、热、故障、版本、telemetry、背景负载、future task 和 live anchor。vehicle/RSU/task/transfer anchors 保存预测起点的剩余资源与进行中事务，避免把空系统误当 rollout 初态。

每个候选动作共享同一外生 scenario realization，但分支拥有独立 `_PredictionState`、资源、任务、传输、admission、battery 和虚拟队列；分支修改不会回写真实模拟器。场景 scheduler 使用与真实 DES 相同的配对 ingress failure 和 taskless maintenance GPU 竞争/完成提交语义。rollout 完成后仅提交当前动作，下一宏事件重新规划。H>1 是经验增强，代码和文档均不把它声明为自动继承 H=1 的全部理论保证或全局最优解。

所有基线通过同一 `DiscreteEventSimulator`、profile、trace、hard mask、repair、指标和外生 seed 执行。环境随机性由冻结 trace/内容寻址确定，MPC 场景随机性另有 seed；稳定排序和内容寻址使策略遍历顺序不改变外生结果。基线 observation 也经过同一字段隔离，不能读取真实身份、标签、攻击真值或未来 evaluation 状态。

## 输出、复现和恢复

`MetricLedger` 输出任务、状态时间戳、路径、重试、网络、admission、入口失败、RSU 排队/推理/maintenance、失败、任务归因能耗、包含 `system_maintenance_energy_j` 的系统物理能耗、时延分位数、工作量、利用率、过滤、修复、控制器诊断、隐私 bound 和虚拟队列。JSON/CSV 为必需输出；Parquet 在可用依赖存在时输出，否则以结构化状态说明。

manifest 绑定代码树版本、规范化配置及 hash、模型/profile/protocol/evidence 版本、trace checksum、split、参数来源、资源/网络/负载/deadline、controller 参数、环境/控制器独立 seed、软件环境、仿真起止和不变量结果。有限 trace 必须使全部任务到达并进入终态，否则运行失败而不是输出部分“成功”结果。

checkpoint 不序列化 live `TaskRecord` 或原图句柄，只保存运行身份、复合事件计数、时钟和稳定前缀 hash；恢复时从头确定性重放并核对边界。因此它保证逻辑恢复/防篡改，但不节省 checkpoint 之前的重放时间。

## 重要工程假设

1. `synthetic_fixture_only` 用于开发测试；`numerical_simulation` 用于冻结数值论文实验。二者都不是实测数据，正式结论必须写成“在指定数值总体、预注册数值攻击器、置信协议、配置和 seed 下的经验仿真结果”。
2. 无线 `goodput_bps` 是应用层净成功载荷，已经吸收 MAC 调度、协议开销、MCS、HARQ/重传和丢包；不能再叠加独立无线等待。
3. trace stage energy 是联合条件下冻结的动态能量；thermal `service_rate_multiplier` 改变完成时间。`dynamic_power_multiplier` 保留为证据字段，但不会再次乘到已配对 stage energy，避免双计。synthetic/numerical maintenance 的 0.18/0.22 busy-s 与 12.6/18 J 也属于冻结工程假设。
4. idle/hold 是系统物理能量；动态 compute、无线和 per-task hold 同时进入任务归因和物理总量。任务归因和物理总量用途不同，不能相互替代。
5. 失败任务的优化成本时延使用完整 relative deadline，以避免“越早失败成本越低”的选择偏差；真实终止时延仍单独输出。
6. v2 identity-retention、攻击分数、FER 概率、工作量、功率、无线和故障参数均为有界工程假设或压力测试边界，并由 evidence/profile/trace/manifest 标记来源；没有伪造设备或文献出处。
7. Python 私有 issuer、冻结 adapter 和车辆执行域是工程可信边界，不是硬件 TEE、OS sandbox 或密码学证明。

## 只剩严格数学或外部有效性限制

当前仍不能、也不应由有限代码测试宣称：

- 对真实世界总体、未知攻击器或单张图像的绝对匿名/信息论隐私；
- 真实道路与任意负载上 Slater 条件成立、无限时域虚拟队列稳定，或 H=1 定理全部随机过程假设在现实中成立；
- H>1 全局最优，或它自动继承 H=1 的全部理论界；
- 工程假设参数精确代表某个具体车辆、RSU、无线网络、城市或人群；
- 有限 seed、有限样本和有限参数扫描能必然外推到所有未知环境，或证明算法普遍优于全部策略；
- API 级原图禁传等价于抵抗恶意解释器、操作系统、侧信道或被篡改源代码的形式化系统安全。

这些限制不妨碍运行完整数值论文实验，但限定了论文可作出的结论。未来接入实测组件时，必须离线预注册主体 split、攻击器/阈值、质量分区、匿名 artifact 及同 artifact FER，生成内容寻址的冻结 evidence/profile/joint trace；缺任何配对支持都应继续保守返回 `unsupported`，不能用跨管线均值或插值补齐。
