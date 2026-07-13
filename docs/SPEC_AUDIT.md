# 规范文件完整性、修复与实现符合性审计

审计日期：2026-07-13。用户已确认最初写出的 `论文1(3).md` 是输入错误，正确论文文件为 `论文1.md`。

## 当前文件校验

| 文件 | 当前大小 | 严格 UTF-8 | 非空 | 异常 C0/DEL | Unicode replacement | 当前 SHA-256 |
|---|---:|---|---|---:|---:|---|
| `方案1.md` | 118,994 byte | 通过 | 是 | 0 | 0 | `25b10fdd276bfe6035184ad754e5fef80f067cfed43d84e4b2eddee0eb8db7b8` |
| `论文1.md` | 44,718 byte | 通过 | 是 | 0 | 0 | `aba0ae3975bbd2bdfa80f7d8eb1b983db7dd31e3f87801569a4faff5763d881a` |

仓库中的 `tests/test_document_integrity.py` 持续检查：文件存在且非空、严格 UTF-8、无异常控制字符、无孤立 CR、无 replacement character，以及若干典型的 LaTeX escape 损坏模式。该测试防止已修复文本在后续处理时重新被字符串转义破坏。

## `方案1.md` 的最小、可审计修复

首次字节审计时，`方案1.md` 的原始版本为 118,991 byte，SHA-256 为：

```text
a80277666db75a8c75179f606428bf37bca9a15955cb2341bfb7bd85616a3dfa
```

它包含 7 个不应出现在 Markdown/LaTeX 公式中的 C0 控制字符：

- 2 个 U+0008，位于匿名事务的有序 attempt tuple 左右定界处；结合相邻 `\zeta^{(1)},\ldots,\zeta^{(J_{n,p})}`，最小修复为正常 LaTeX 左右括号。当前可见文本位于第 367、369 行的 `\left(` 和 `\right),`。
- 5 个 U+000C，后面紧跟 `rac`，在对应分式上下文中只能形成被 escape 解释损坏的 `\frac`；最小修复为恢复反斜杠和字母 `f`。

没有重写公式、变量、方法、阈值、事件顺序或理论结论。修复后文件为当前 118,994 byte、SHA-256 `25b10f...b8db7b8`，异常控制字符数为 0。保留修复前后 hash 和损坏类型，是为了使这次必要的文本改动可追溯；因此不能再表述为“规范源文件完全未修改”。`论文1.md` 未发现需要修复的控制字符或 LaTeX escape 损坏。

## 研究依据和冲突优先级

```text
方案1.md
> 论文1.md
> 为闭合数值仿真而作出的、明确标记来源的工程假设
```

`论文1.md` 提供研究问题、honest-but-curious 威胁模型、原图不离车的动机、两阶段匿名边缘推理、后选择隐私风险和实验目标。`方案1.md` 进一步冻结了连续事件时间、完整状态机、联合 trace、主体级三风险 UCB、RSU 原子 admission、有限资源、确定性修复和 ESL-SMPC 数学语义。两者冲突时，代码遵循后者。例如论文早期的 RL 主求解器没有覆盖方案第七部分最终指定的事件驱动安全 Lyapunov–场景 MPC；RL 也没有被作为在线训练旁路保留。

## 规范条款与当前实现

| 规范主题 | 当前实现证据 | 审计结论 |
|---|---|---|
| 原图不得离车/策略标识隔离 | `packets.py` 的不可序列化 raw/aligned handle、私有 evidence capability 链；`ObservationBuilder` 的 allow-list 与任务作用域 artifact token | 网络接口没有 raw upload action；真实 evaluation artifact key/完整索引不进入策略可达对象图，registry 只保存脱敏 action capability，跨任务 token 重放不能获得配对支持 |
| 在线不训练 | `profiles.py` 的冻结 dataclass/mapping、`adapters.FrozenAdapterRegistry.seal()`、配置/evidence 的 `online_mutable=False` 校验 | 在线路径只推理、重放、调度、检查、事件推进和修复 |
| 隐私硬约束 | `FrozenProfileBundle.query_privacy`、`HardMaskEngine`、`DeterministicRepair` | identity/verification/link、主体支持、emission LCB、OOD、版本、联合支持均先过滤；策略成本不能抵消 privacy reason |
| 连续真实事件时间 | `events.py`、`simulator.py` | next-event DES；有限 ulp 等价闭合浮点同刻，完成先于 deadline，物理近邻不误合并；终态删除未来 task-owned 事件；无固定步长轮询 |
| 闭合状态机 | `TaskState`、`LEGAL_SUCCESSORS`、`TaskStateMachine` | PREP/local/anon/guard/encode/READY/UL/ingress/GPU/DL/终态均有等待、运行或传输状态及失败后继 |
| 有限资源和真实排队 | `resources.ResourcePool`、车辆 runtime、RSU ingress/GPU、deployment bounds | 有限服务器、非抢占 EDF、residual busy-second、真实 enqueue/start/end；阶段 reservation 原子 reconcile，同刻 planned shadow 防双重承诺；无 queue-length/rate 等待替代 |
| 联合匿名事务和 FER 配对 | `AnonTraceRow`、`EdgeFERTraceRow`、`TraceBundle` loader/support query | 尝试、阶段时间/能量、guard、编码、OOM、大小、artifact、FER 整行重放；`ingress_failed` 与入口/GPU/DL/结果保持配对，不另抽失败概率；缺配对即 unsupported |
| 无线净服务 | `Transfer.advance`、simulator 分段积分 | total/remaining/delivered bit 闭合；临时暂停可恢复；handover 不迁移 partial packet；双侧能量保留 |
| RSU admission | `RSUAdmission.can_admit/admit/snapshot`、profile resource certificate、UL completion handler | READY snapshot 不预留；对全部 conformal 候选区用冻结资源上界构造请求，evaluation $g^*$ 不参与；完整包抵达后按实时状态原子接纳；拒绝零副作用 |
| RSU 模型/cache 维护 | `Operation.RSU_MODEL_MAINTENANCE`、真实/场景 scheduler | 版本/cache 事件必须有正 work/energy；同 `(RSU, model)` 链在多 GPU 上顺序派发并检查 old-version，不同 model 可并行；完成时才提交；H>1 同语义 |
| RAW/READY 两阶段动作 | `safety.Action`、`HardMaskEngine.enumerate`、策略集合 | RAW=local/pipeline/fail，READY=edge/local/fail；每个删除和 repair 有结构化审计 |
| H=1 ESL | `SafeLyapunovPolicy`、`score_current_actions` | 真实持续时间的漂移—成本比；只比较 hard-safe 集；延迟提交失效时对当前安全集重算，包括 newly-safe 动作 |
| H>1 ESL-SMPC | `ESLSMPCPolicy`、scenario environment/anchor/branch scheduler | training/validation 联合场景；分支隔离；模拟完整路径；延迟修复重算当前场景分数且不污染诊断/RNG/真实状态；只执行第一步 |
| 公平基线 | `AllLocalPolicy`、固定安全管线两基线、`SafeGreedyPolicy`、H=1、H>1 | 共用模拟器、trace、hard mask、repair、指标、外生随机性和 observation 边界 |
| 不变量、输出、manifest | `invariants.py`、`metrics.py`、`manifest.py`、`checkpoint.py` | 运行中 fail-fast；终态无 future event/job/transfer/reservation；输出含 `system_maintenance_energy_j`；复现 hash/版本/种子/环境绑定；确定性前缀重放恢复 |

更细的数学对象映射和物理边界见 `docs/ARCHITECTURE_AND_ASSUMPTIONS.md`。本文件不写测试通过数量；最终实际测试命令和结果由 `docs/FINAL_AUDIT.md` 在完整测试结束后统一记录，避免中途数字失真。

## numerical v2 隐私模型修正审计

当前数值 evidence 版本为 `numerical-study-2.1.0`，profile/trace 版本为 `2.2.0`；12 条管线及三个攻击器使用 `numerical_v2` 标识。它不是对方案核心方法的替换，而是“没有真实硬件/攻击模型时，以数值仿真闭合论文实验”的离线数据生成实现。

v2 明确修正了会使数值 safe set 退化的评分语义：

- residual identity 由 method retention × strength retention × quality multiplier 构成，质量不能在 retention 为零时凭空产生身份信息；
- temporal persistence 同样由 residual retention 门控，不能只因主体 latent persistence 存在就产生匿名链接成功；
- projected/temporal/cosine attacker 的 fitting 与 evaluation 使用同一个 `_observable_identity_signal` feature space，避免拟合目标和部署特征错配或重复加入 persistence；
- exact retention、quality、temporal、target intercept/slope/risk offset 均由 `_privacy_score_model_document()` 写入 evidence，标记为 `engineering_assumption` 并计算自哈希；attacker registry 绑定该版本/hash；
- evidence loader 会重算 score-model hash、attacker 绑定、threshold split、阈值 ID、完整预注册维度和 profile bound cross-reference。仅修改一个 JSON 数字不能绕开验证。

三个攻击器仍覆盖 cosine gallery、projected margin 和 temporal linker 三种数值评分骨干；三种风险仍分别为 identity、verification、link。没有为了得到安全动作而删去攻击器、风险类型或缩小 Bonferroni 假设族。所有精确参数都属于工程假设，不是实测人脸攻击结论。

## profile-evaluation 真实质量区 $g^*$ 审计

`方案1.md` 的 $m_{i,g}$ 是主体 $i$ 在真实质量区域 $g$ 中的相关帧数，不能把全部主体/帧复制到每个 cell。当前实现通过 `_profile_evaluation_quality_support()` 冻结这一语义：

- profile-evaluation 每帧按 reference features 和冻结 partition 唯一赋予一个 $g^*$；
- 主体仅在至少一帧位于 $g$ 时进入该 cell；风险分母使用该主体的实际 $m_{i,g}$；
- 每帧攻击分数使用该帧配对的真实 quality score；
- stable subject/frame 顺序、subject/frame count、subject hash 和 support self-hash 被持久化；
- `_privacy_evidence()` 的每条主体风险行绑定 cell support；`evidence.py` 验证 profile-evaluation 的全部 `(subject, frame)` 恰好一次覆盖、无重复/越界/缺失，并验证完整 pipeline × quality-region × attacker × risk 笛卡尔积；
- profile 中每个 UCB 必须与 evidence 重算值一致；在线候选区域通过 `query_privacy()` 取交集和最坏风险。

若生成时某个质量区没有 profile-evaluation frame，生成器保守失败并要求增加支持，而不是制造空 cell 或跨区插值。运行时若真实导入 profile 缺 cell/配对，仍返回 `unsupported`。这种“生成期拒绝无支持研究包”的表达比继续生成一个不可用 profile 更严格，但安全含义与方案一致。

## 数值仿真、真实测量与可作结论

当前仓库已经有一条完整的冻结数值研究路径：互斥主体 split、quality partition/conformal、三个数值攻击器及独立阈值、三风险主体级同时 UCB、FER 配对、联合匿名事务、无线/热/故障/背景负载/telemetry/版本场景、车辆和 RSU 有限资源、两阶段调度、基线、指标和 manifest。缺少真实车辆、RSU、攻击器和人脸模型不阻止运行论文的数值仿真实验。

但 `data_kind=numerical_simulation` 不能重命名为 measurement。正确结论边界是：

> 在指定冻结数值总体、预注册数值攻击器、质量/阈值/置信协议、配置和 seed 下的经验仿真结果。

它不是绝对匿名、真实硬件性能或真实道路长期保证。未来若接入实测数据，应使用同一 schema/hash/adapter/conservative-support 路径，在离线阶段写入真实来源和配对 artifact；不能让在线阶段训练或用平均值补齐缺失组合。

## 仅剩的非代码证明边界

代码、单元/集成测试、确定性受控实验和多 seed 数值实验可以验证实现语义、数值关系和指定总体下的经验表现；当前不应继续列出可由这些手段闭合的“待实现功能”。仍不能由有限仿真严格证明的项目是：

- 对未知真实总体、未知攻击器和单张图像的绝对匿名；
- Slater 条件、随机过程平稳/遍历等 H=1 无限时域定理前提在真实世界必然成立；
- H>1 的全局最优性或自动继承 H=1 的完整理论保证；
- 工程假设参数精确代表特定硬件、城市、网络、人群或攻击生态；
- 有限样本、seed 和扫描结果可无条件外推到所有未知环境；
- Python API/不变量边界构成抵御恶意 OS、解释器、被篡改源代码和侧信道的形式化安全证明。

这些是严格数学或外部有效性限制，而不是可用占位代码掩盖的工程缺口。
