# VLM 推理与提示开发细则

适用版本：`spec_version=coarse_v1.0`。这是待实现的开发规范，不表示推理程序或命令已经存在。总体任务与开发入口见 [agen.md](./agen.md)，边界与标签规则见 [boundary.md](./boundary.md)。本文规定 Qwen 输入、提示、响应和失败处理；同名字段须与总体契约一致。

## 1. 本版目标与限制

- 对每个 episode 只使用主视角，以目标 1 FPS 抽样，按时间顺序拼成网格图，将该 episode 的全部网格页放在同一个逻辑请求内。
- 模型只判断规范适用性、四阶段是否出现，以及三个切换边界和第四阶段完成的采样证据；程序负责原始帧映射、边界与标签生成。
- 本版不训练模型，不读取腕部视角，不做边界局部窗口重看，不使用夹爪信号补充判断，不自动对长 episode 拆成独立子问题。
- 这是粗阶段标注。抽样间隔限制可见证据，1 FPS 不是模型边界误差不超过 1 秒的保证。
- 阶段 ID 固定为 `1, 2, 3, 4`。具体语义必须由 `standard.zip` 图示和校准集确定并冻结；不得凭数据集名称将阶段 4 定成收纳、回位或挂衣。
- 标准的边界归属保持不变：完成当前阶段的帧仍属于当前阶段，例如 422 帧完成阶段 1，则 423 帧才进入阶段 2。本版保存的边界是采样证据推导出的粗近似，不声称恢复了规范中的精确完成帧。

## 2. 模型与推理适配

默认候选权重为 `Qwen/Qwen3-VL-30B-A3B-Instruct`；`Qwen/Qwen3-VL-8B-Instruct` 作为可选基线。二者应使用同一批输入、标准、提示和人工评估集比较，模型变化不能与网格或提示变化混在同一项效果归因中。

权重名称与服务商的 API `model` 别名是两个字段。当前 API endpoint、可用 model ID、图像限制、结构化输出能力和配额均需预检，不得把未验证的地址或别名写成生产配置。已有凭据只通过运行环境或受控秘密文件读取；原始 key 不进入配置样例、缓存、日志、异常栈、网格标题或版本控制。

适配器须提供统一逻辑接口：接收提示文本、按顺序排列的图像和输出 schema，返回原始响应文本、finish reason、用量和请求元信息。API 与本地推理实现应满足同一输入输出契约。

本地加载时检查模型配置中的架构和官方支持情况。30B-A3B 属于 MoE 模型；不要假设密集模型的专用 loader 可直接加载它。优先使用当前运行环境支持的 Auto loader，或根据配置分派到匹配的 Qwen3-VL MoE / dense 实现，同时固定 processor、模型 revision、推理框架及依赖版本。加载成功后须以多图测试确认，不能只验证文本请求。

推理参数约定：

| 项目 | 开发要求 |
| --- | --- |
| 温度 | 服务商支持时设 `temperature=0`；不支持则按能力处理并记录实际值，禁止伪称完全确定性 |
| 输出格式 | 优先 JSON Schema；其次经过验证的 JSON mode；均不支持时用严格 JSON 提示和本地解析，能力差异写入运行清单 |
| 输出长度 | 按真实 schema 和证据字段预算，预留完整三个 proposal 的空间；禁止使用过小上限导致尾部丢失 |
| 流式 | 默认关闭，便于可靠取得完整响应与终止原因；若开启，必须实现完整拼装和中断检测 |
| 解码 | 本地推理只解码新生成 token，不能把输入提示误当响应 |
| 日志 | 记录 model ID / revision、参数、延迟、用量、finish reason 和 request ID；对秘密及请求头脱敏 |

## 3. 采样与网格输入契约

### 3.1 唯一采样表

抽样和原始视频帧映射由数据层先生成唯一采样表。模型不负责计算采样时间或换算帧号。

每行至少包含：`sample_seq`（从 0 连续递增）、`sample_id`（如 `s000010`）、`source_frame_id`、`source_timestamp_s`。ID 在单个 episode 内唯一，跨页不重新编号。主视角名称由元信息明确解析并记录；缺失时失败，不能自动换成另一相机。

目标采样时间为 0、1、2……秒，实际选取哪个解码帧必须由采样实现确定、记录并保持可复现。对非整数帧率、缺帧或时间戳不均匀的视频，以实际采样表为准，不假设 `frame_id = 秒数 × FPS`。不允许向模型输入不存在的帧号。末尾补采等数据层决定应在采样配置中冻结，不能由推理端临时更改。

### 3.2 网格页

- 每页 3 列、4 行，共 12 格，阅读顺序为从左到右、从上到下。
- 相邻页重叠 2 个采样格：第一页覆盖 seq 0–11，第二页覆盖 seq 10–21，第三页覆盖 seq 20–31，依此类推。
- 重叠格复用同一个 `sample_id`，不是新观测，不得把重复格当成时间停顿或增加样本总数。
- 最后一页未用格标为 `EMPTY`，不复制末帧填满，不虚构 sample ID。
- 每页标题含 `episode_id`（如 `episode_000000`）、`page_id`（如 `p000`）、总页数和阅读方向；每格可读地显示 `sample_id`、原始 `frame_id` 和时间戳。
- 单格保留主视角全画面和宽高比；信息标签放在画面外的固定条带中，不遮挡夹爪与衣物。
- 渲染尺寸、字号和压缩参数写入配置并在校准集检验，至少保证夹爪开合和文字标签可辨。不得在不知道服务商二次缩放行为时宣称文字一定可读。
- 禁止跨 episode 拼接。同一请求中的所有 temporal page 必须属于同一个 episode，并按 `page_id` 顺序排列。

应将标准参考图与时序页分开命名，例如 `REFERENCE_STANDARD` 和 `TEMPORAL_PAGE p000`。标准参考图不参加时间排序，也不计入采样表。

### 3.3 一次请求的组成

依次提供：系统提示、已冻结的四阶段标准文本和参考图、episode 控制信息及采样表、按页号排序的全部网格图、输出格式要求。图像前配文本标签，避免页序依赖服务端隐式行为。控制信息至少包括任务上下文、主视角、总帧数、实际采样数和页面数；`robot_type` 可作为元数据，但不属于 `task_type` 类别。

发起请求前估算文本、图像 token、图像数量、请求体大小以及输出预留量。以模型与服务商实测限制为准。输入预算超限时标记 `input_budget_exceeded`，进入待处理队列；禁止静默丢页、截掉视频末尾、降低采样频率或只挑前半段。调整全局渲染策略必须生成新的输入配置 hash 并重新校准。本版不通过局部窗口请求绕过预算限制。

## 4. 中文提示模板

以下模板中的花括号由程序填充。四阶段定义未核对、存在空值或参考图未准备好时，不得发起正式标注；不能用模型常识代替缺失规范。

### 系统提示

```text
你是机器人衣物操作视频的粗阶段标注员。只依据本请求提供的标准图示、
阶段定义和主视角时序网格图判断。网格不是多相机画面，每格是同一个
episode 在不同时间的采样。每页从左到右、从上到下阅读，页面依次衔接。
相邻页有 2 个重复采样格，同 sample_id 表示同一时刻，不是重复动作。

你的职责是给出三组边界的可见采样证据，不输出或猜测连续视频中的精确帧。
只能引用采样表中真实存在的 sample_id，不创造时间、帧号或 sample_id。
遇到遮挡、缺阶段、返工、非适用任务或看不清的情况，必须如实拒判。
不要为了凑齐四阶段而把视频等分，也不要根据页码、时长或文件名猜测阶段。
只返回符合指定 schema 的单个 JSON 对象，不返回 Markdown、解释段落或代码块。
evidence_text 只描述可见事实，不提供思考过程，也不提供自报置信度。
```

### 用户提示

```text
规范版本：coarse_v1.0
episode：{episode_id}
主视角：{main_camera}
原始帧数：{num_frames}
实际唯一采样数：{num_unique_samples}
网格页数：{num_pages}，每页 3 列 4 行，相邻页重叠 2 格。

【标准参考】
REFERENCE_STANDARD 图只用于理解阶段，不属于视频时序。
以下定义由 standard.zip 与校准集核实，禁止自行扩展：
阶段 1：{verified_stage_1_definition}
阶段 2：{verified_stage_2_definition}
阶段 3：{verified_stage_3_definition}
阶段 4：{verified_stage_4_definition}
共同完成判据：当前阶段对应的折叠动作已完成，且参与该动作的夹爪完全松开。
完成帧仍归当前阶段：例如 422 帧完成阶段 1，423 帧才进入阶段 2。
此例只说明边界归属，不是当前 episode 的真实边界。

【任务判断】
task_type 只能取 fold_cloth / fold_pants / hang_cloth / mixed / other / unknown。
applicable 表示本 episode 是否适用上述四阶段规范；按可见操作判断，
不能仅依据文件名或数据集名称判断。无法判断是否适用时 applicable=null，
needs_review=true。阶段 4 的含义严格采用给定定义。

【边界证据】
固定返回 1→2、2→3、3→4 三个槽位，不增加、不合并、不改变顺序。
对于每一组边界：
1. 找到当前阶段的完成判据尚未满足的最后一个确定可见采样，填入
   last_not_complete_sample_id。
2. 找到紧随其后的、首次确定完成判据已满足的采样，填入
   first_complete_sample_id。
3. 两个采样必须是唯一采样表中相邻的 sample_seq。跨页重复格只计一次。
4. 只要两侧之间存在无法判断的采样，不能声称已形成相邻确定证据；
   无法确定的一侧填 null，needs_review=true，并给出原因。
5. 完成动作不等于仅开始下一步；夹爪不可见时不得仅凭衣物位置变化宣称
   “完全松开”。短暂松开后重抓、动作返工或阶段顺序不清时标记复核。
6. 不输出边界秒数或原始帧号，程序会根据采样表完成映射。

all_stages_observed 仅在四阶段均有可见证据时为 true。
另外返回 stage4_completion：observed 表示阶段 4 的完成判据确实可见；
completion_sample_id 引用首次可观察到阶段 4 完成的采样，无法判断则为 null；
evidence_text 简述可见终态依据。该证据不是第四个阶段间边界。
仅看到阶段 4 开始或视频结束，不能推断阶段 4 已完成。
只要规范不适用、任一阶段缺失、任一边界不能确定、第四阶段完成不可见或有明显返工歧义，
needs_review 必须为 true。无法判断 task_type 时使用 unknown。
不能确定的边界槽位仍保留，sample_id 使用 null，并描述可见障碍。

【唯一采样表】
{sample_table_with_sample_seq_sample_id_source_frame_id_source_timestamp_s}

下面按顺序附上 TEMPORAL_PAGE p000 ...，所有页属于同一个 episode。
{ordered_temporal_pages}

【输出约束】
返回 task_type、applicable、all_stages_observed、needs_review、
review_reasons、stage4_completion、boundary_proposals。stage4_completion 仅含
observed、completion_sample_id、evidence_text。boundary_proposals 每项仅含
from_stage、to_stage、last_not_complete_sample_id、first_complete_sample_id、
evidence_text。review_reasons 为字符串数组，不能确定时说明具体遮挡、
缺失或歧义。不要输出 confidence、额外字段或未提供的 sample_id。
```

## 5. 原始模型输出 schema

固定顶层字段如下，所有字段必填；对象层级均拒绝额外字段。`spec_version`、episode 身份、模型身份、调用参数等由 caller 放入结果 envelope，不能由模型自报后直接信任。

| 字段 | 类型与约束 |
| --- | --- |
| `task_type` | 字符串枚举：`fold_cloth`, `fold_pants`, `hang_cloth`, `mixed`, `other`, `unknown` |
| `applicable` | boolean 或 JSON `null`；null 表示无法判断适用性，不接受字符串 `"true"` / `"false"` / `"null"` |
| `all_stages_observed` | boolean |
| `needs_review` | boolean |
| `review_reasons` | string 数组；建议上限 10 条，每条上限 300 字；无原因时空数组 |
| `stage4_completion` | 必填对象：`observed` 为 boolean；`completion_sample_id` 为真实 ID 或 null；`evidence_text` 为非空字符串，最多1000字。observed=true 时ID非空；false 时ID为null |
| `boundary_proposals` | 恰好 3 个对象；顺序固定为 `1→2`, `2→3`, `3→4` |
| `from_stage`, `to_stage` | 严格整数；不能接受 bool、浮点数或数字字符串；按固定槽位验证值 |
| 两个 `*_sample_id` | 字符串或 JSON `null`；非 null 时必须精确匹配本 episode 唯一采样表的 ID |
| `evidence_text` | 非空字符串，上限 1000 字；只记录与该边界有关的可见事实或无法判断原因 |

下面仅为格式示例。`s000010` 等为假设采样 ID，不代表项目真实标注，也不能作为 few-shot 的真实边界值写入提示。

```json
{
  "task_type": "fold_cloth",
  "applicable": true,
  "all_stages_observed": true,
  "needs_review": false,
  "review_reasons": [],
  "stage4_completion": {
    "observed": true,
    "completion_sample_id": "s000040",
    "evidence_text": "假设示例：此采样可见阶段4定义的操作已完成，参与动作的夹爪已松开。"
  },
  "boundary_proposals": [
    {
      "from_stage": 1,
      "to_stage": 2,
      "last_not_complete_sample_id": "s000010",
      "first_complete_sample_id": "s000011",
      "evidence_text": "假设示例：前一采样中参与动作的夹爪仍抓住衣物；后一采样中本次折叠已完成且夹爪已完全松开。"
    },
    {
      "from_stage": 2,
      "to_stage": 3,
      "last_not_complete_sample_id": "s000020",
      "first_complete_sample_id": "s000021",
      "evidence_text": "假设示例：这两个相邻采样分别显示阶段 2 未完成和已满足完成判据。"
    },
    {
      "from_stage": 3,
      "to_stage": 4,
      "last_not_complete_sample_id": "s000030",
      "first_complete_sample_id": "s000031",
      "evidence_text": "假设示例：这两个相邻采样分别显示阶段 3 未完成和已满足完成判据。"
    }
  ]
}
```

未知时不能用字符串 `"null"`、`"unknown"`、空字符串或 `-1` 代替 JSON `null`。例如两侧均看不清时，该槽两个 ID 均为 `null`，`evidence_text` 写明“主视角中夹爪被遮挡，不能确认释放”。

## 6. 程序校验与边界映射

处理顺序为完整响应检查 → JSON 解析 → 严格 schema → 采样引用 → 语义一致性 → 帧映射 → 输出状态。保留原始模型文本和失败原因，不通过修改候选内容把失败“修成成功”。

只有以下全部成立才有资格标记 `accepted`：

1. `applicable=true`、`all_stages_observed=true`、`needs_review=false`、`review_reasons=[]`。
2. 三个固定槽位齐全，所有 L/R ID 非 null，并属于当前 episode。
3. 每组 `R.sample_seq = L.sample_seq + 1`；重复网格格不增加 seq。
4. 每组 `L.source_frame_id < R.source_frame_id`，三组映射后的完成帧严格递增；展开后的各阶段区间均非空、无越界。
5. `task_type` 与已经冻结的标准适用范围相容；`unknown` 不可自动验收。不能仅根据 `fold_cloth` 等名称自动判定适用性。
6. `stage4_completion.observed=true`，终态 sample ID 非空、真实存在且映射帧严格大于第三个粗结束帧；其 evidence_text 与已冻结的阶段 4 完成定义一致。
7. 无输入不完整、响应截断、非法引用、缺阶段、终态不明、重抓/返工歧义或与布尔标记矛盾等失败条件。

任意非 null ID 引用不存在采样、顺序错乱或同组不相邻，均是语义验证失败，不得用“最近的合法 ID”替换。即使模型自报无需复核，程序也必须转入 `needs_review` 并附加机器原因。若模型明确声明 applicable=false，记录 `out_of_scope`；applicable=null 则为 `needs_review`。原始候选照常留存，不能自动填出四阶段。

对于通过验收的每组采样证据，令 L 为 `last_not_complete_sample_id`，R 为 `first_complete_sample_id`，程序计算：

```text
coarse_end_frame = R.source_frame_id
estimated_boundary_interval_frames = [L.source_frame_id + 1, R.source_frame_id]
# 另存 source_timestamp_s 两端，时间区间为 (t_L, t_R]
next_stage_start_frame = coarse_end_frame + 1
```

这个整数不确定区间只有在 L 明确未完成、R 明确完成，且两者为相邻唯一采样时才有含义。它表达“在这些可见证据成立的前提下，完成可能发生的帧范围”；不是统计置信区间，不保证真边界一定落在其中。漏看短暂释放、遮挡或模型误判时都可能不成立。

阶段 segments 持久化统一采用闭区间 `[start_frame, end_frame]`；Python 切片使用 `end_frame + 1`。第一段从 0 开始，相邻段首尾相接，第四段默认延续至 `N-1`。末段延续是粗标签工程约定，不表示规范已认定阶段 4 在 episode 末帧精确完成，也不能据此推断阶段 4 的语义。

仅 `accepted` 记录可生成四阶段训练候选。`needs_review`、`out_of_scope` 以及技术失败记录的训练候选标签全部置 `-1`；仍保存三个原始 proposal、已能解析的映射信息和失败原因，供人工查看。不得将部分确定的边界补齐为四阶段后混入自动验收结果。

## 7. 重试、拒判与调用状态

技术重试与语义拒判分开统计。耗尽总尝试次数时记录 `execution_error=retry_exhausted` 并停止自动派发，只有新 run 或记录原因的显式重置才可再请求。一个逻辑请求包含某 episode 的全部页；因网络失败重发同一完整输入不构成局部重看。语义拒判不通过反复询问同一图像直到得到“成功”来解决。

| 情况 | 处理 |
| --- | --- |
| 网络超时、连接中断、429、可恢复 5xx | 最多 3 次总尝试，即首次 + 2 次重试；优先遵守 `Retry-After`，否则指数退避加 jitter；记录每次请求状态；attempts 持久化、跨进程恢复累计，重启不清零 |
| 推荐退避 | 第 1 次失败后 `2 秒 + U(0,1秒)`，第 2 次后 `4 秒 + U(0,1秒)`；更长的服务商限流值优先，调度层承担等待 |
| 认证失败、权限不足、模型不存在、请求结构不受支持 | 不盲目重试；终止该配置的运行，修复预检项后再启动 |
| 输出长度终止、响应未闭合、流中断 | 记录 `response_truncated`；禁止按完整标注解析；检查输出预算或服务能力后按新配置重跑 |
| JSON / schema 不合法 | 记录 `response_invalid` 并生成最终 `needs_review`；不做字符拼补、正则提取部分 JSON 或自动补字段。实现期定位原因后按新 prompt/schema 版本重跑 |
| 看不清、标准不适用、缺阶段、非相邻证据、动作歧义 | `needs_review` 或 `out_of_scope`；保留证据，不自动重问，不补采，不调用其他视角 |
| 输入预算或页数超限 | `input_budget_exceeded`；禁止默默截断输入或把长视频切成独立标注请求 |

不得将 HTTP 200 等同于标注成功。若服务商不提供 finish reason，应由适配器记录该限制，并用完整解析与 schema 检查作为最低要求；JSON 完整也不能证明服务商没有截掉输入，因此还须完成页数和输入限制预检。

对可恢复错误，可用 idempotency 能力时复用逻辑请求标识，并记录可能重复计费。语义失败默认不进入自动技术重试队列。人审以后若改用新规范、新图或新模型，属于新 run，不能覆盖旧记录。

## 8. 缓存、可复现与观测

缓存键至少覆盖：`spec_version`、标准文本与参考图 hash、`prompt_version`、schema 版本、模型标识及 revision、推理参数、输入配置 hash、dataset/episode 稳定键、源视频 hash、唯一采样表 hash、按页顺序的图像内容 hash。使用规范化序列化构造 hash，图像内容变化、页面顺序变化或模型别名背后已知的版本变化都不能复用旧语义结果。

缓存 envelope 记录 dataset/episode 身份、run ID、请求 ID、开始时间、各次尝试、耗时、token 用量（服务商提供时）、finish reason、原始响应、解析后的输出、程序校验错误和最终状态。provider model alias 无法追溯权重版本时明确记录 `revision_unavailable`，不伪造具体 revision。若 revision 无法确认，禁止跨 run 复用语义缓存；同一 run 已落盘的 response 仍可恢复，报告不能保证服务端模型完全可复现。输入摘要可缓存，原始 key 与 Authorization 头不得保存。

格式合法但语义失败的响应也可缓存，避免重启后反复产生费用；缓存状态不能被当作 `accepted`。模型自报的“确信”“高置信”等词语不形成分数，也不参与自动验收。质量以人工真值上的阶段完整率、边界偏差、整 episode 全边界正确率、复核率和成本衡量。

## 9. 开发完成检查

本文件不约定已经可运行的 CLI。将来的入口统一由 `agen.md` / runbook 描述，避免写出尚不存在的脚本命令。实现验收至少覆盖：

- 一个 episode 多页输入、页序与重复格正确，所有采样 ID 可回查原帧；标准参考图不会混进时序。
- 最后一页有空格、主视角缺失、超长输入、页图损坏或不可读标签时均能显式失败。
- 3 条有效相邻证据能映射为闭区间；未知、跨 episode ID、重复槽位、非相邻采样、乱序和非递增边界均不能自动验收。
- 422/423 归属的程序单元测试不发生 off-by-one；它验证规则实现，不把 1 FPS 输出包装成逐帧真值。
- 429/5xx 至多 3 次总尝试；认证错误不循环；截断和不合法 JSON 不写出可训练标签。
- `needs_review` / `out_of_scope` 的训练候选全部为 `-1`，原始候选和原因仍可人工核查。
- 同一输入和版本可命中缓存；更改规范、prompt、模型或页面后失效；任何日志均不包含原始 key。

## 10. 算法参考与采用边界

下列链接为官方仓库或模型页，查阅日期为 2026-10-22。实现时还需记录采用的 commit / revision，避免跟随 `main` 变化而不留痕。

- [Qwen3-VL 官方仓库](https://github.com/QwenLM/Qwen3-VL)：模型系列、多图输入、processor 与推理方式的接口参考。本项目采用时序网格作为多图输入，不能直接沿用原生视频输入的时间戳行为。
- [Qwen3-VL-30B-A3B-Instruct 官方模型页](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)：默认候选权重与 MoE 架构信息；API endpoint 和服务商 model alias 仍须本地预检。
- [Qwen3-VL-8B-Instruct 官方模型页](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)：可选密集模型基线，是否能替代 30B 由同一校准集结果决定。
- [LeRobot SARM 官方标注实现](https://github.com/huggingface/lerobot/blob/main/src/lerobot/data_processing/sarm_annotations/subtask_annotation.py)：参考“预定义子任务闭词表 + VLM 粗阶段标注 + JSON 输出”的组织方式。本项目使用真实 sample ID，不照搬整数 `MM:SS` 的时间输出；数据适配不得假定当前项目拥有该实现使用的 v3 metadata 布局。

网格尺寸、重叠格、样本 ID 证据对以及拒判策略属于本项目的设计，不能写成上述项目已经验证过的准确率结论。
