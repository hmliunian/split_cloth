# 已实现功能与运行细则

项目：`/gpfs/jiuquyun/home/jackson/xuran_projects/split_cloth`，服务器：`h100-jackson`。本文记录本轮已经实现的能力；阶段语义和网格规则另见 `boundary.md`、`sampling.md`、`vlm.md`。

## 算法和输入

当前配置为 `configs/coarse_v1.yaml`，规范版本 `coarse_v1.1_expert`，模型为百炼 OpenAI 兼容接口上的 `qwen3-vl-plus`。凭据由服务器项目内的 `Qwen_api.txt` 读取；文档、日志和 GUI 不应复制或显示凭据内容。

输入顺序包含：阶段文字与标准参考图、已标注 expert 示范、目标 episode 全部时序网格页。一个目标的全部页面放在一次请求中。仅使用 `base_0_rgb` 主视角，1 FPS 采样，包含末尾补帧；网格为 3 × 4，页间重叠 2 个采样点。当前实现不使用局部窗口、多视角或本地 Qwen 推理。图像准备、API 调用、Web 服务均不要求本地 GPU。

四阶段粗语义如下，仍以标准图及后续人工确认优先：

1. 取衣、整理和展平，直到第一侧纵折完成并松开。
2. 另一侧纵折，形成长条并松开。
3. 横折成小矩形并松开。
4. 拿走、移出折叠工作区；空载返回可作为完成后的可见证据。画外放置质量无法由本视角验证。

Qwen 只选择实际输入中的采样编号，程序统一转换为原视频帧号。对相邻采样 `L / R`，`L` 表示尚未完成，`R` 表示已完成并松开。粗结束帧为 `R.source_frame_id`；真实边界候选区间保留为 `[L.source_frame_id + 1, R.source_frame_id]`。所有区间采用闭区间，完成采样属于前阶段，后一阶段从结束帧加 1 开始。

必须确认适用当前折衣标准、四阶段完整、边界递增、第 4 阶段有完成证据，且无待复核原因，才会自动成为 `accepted`。模型不能解析、边界编号无效、顺序错误、阶段缺失或证据不足均保留 `needs_review`；明确不适用的任务保留 `out_of_scope`。无效标签为 `stage=-1`，不按时长等分补标签。

## 固定试标集与 expert

当前运行：`runs/pilot_100_20261022_v2`。

- 目标 100 条：v3 40、v4 30、mixed 30。
- 按时长四分位分层抽取，随机种子 `20261022`。
- 每个数据集分为 60% calibration、40% heldout，合计 60 / 40。
- 独立 expert 为 v3 的 `episode_000000`，从所有目标池排除。
- Expert 的阶段语义、边界和标准图解释来自 GPT 视觉参考，标记 `validation_level=gpt_visual_reference_not_human_ground_truth`。
- 6 条 GPT 独立网格盲审保留独立文件和审查范围，不能将其解释为 100 条独立人工真值。

`calibration / heldout` 是固定样本划分，不表示已拥有可计算准确率的人工标签。实际分类计数、token 和复核数量以本轮报告文件为准；此文不预填准确率或调用费用。

## 现有 CLI

参数 `--project` 和 `--config` 放在子命令前；`--run` 放在子命令后。默认配置为项目根目录下 `configs/coarse_v1.yaml`。

```bash
cd /gpfs/jiuquyun/home/jackson/xuran_projects/split_cloth
export PYTHONPATH="$PWD/src"
RUN=pilot_100_20261022_v2

.venv/bin/python -m split_cloth.cli --help
.venv/bin/python -m split_cloth.cli --project "$PWD" summary --run "$RUN"
```

| 子命令 | 实际作用 |
|---|---|
| `select --run RUN` | 新建并冻结 100 条清单、60/40 划分、独立 expert 清单；已有清单时拒绝覆盖 |
| `prepare --run RUN` | 为目标轨迹生成采样图、网格图、可播放预览及输入清单 |
| `prepare --run RUN --expert-only` | 仅准备独立 expert 素材 |
| `register-expert --run RUN --review FILE` | 校验并注册 GPT 视觉参考，渲染带阶段标签的 expert 页面、提取标准图 |
| `annotate --run RUN [--limit N]` | 调用 Qwen、严格校验、映射边界、保存结果和粗标签；支持同配置续跑 |
| `evaluate-gpt --run RUN` | 对已保存的 GPT 盲审进行边界/适用性比较，生成抽样验证报告 |
| `compare --baseline BASELINE --run CANDIDATE` | 按同一episode key生成模型配对对照，只写candidate报告，不改原结果 |
| `summary --run RUN` | 汇总处理状态、实际记录的 token 及 GPT 复核数量，更新报告 |
| `serve --host 127.0.0.1 --port PORT` | 启动 GUI 与素材读取、人工复核 API |

没有实现 `inspect`、`evaluate`、`export` 子命令；原开发计划中提及它们不代表目前可以执行。

### 从新运行开始

已完成清单选择的当前运行不要再次执行 `select`。需要独立复现实验时选一个新的 RUN 名称：

```bash
RUN=pilot_100_NEW_RUN_ID
.venv/bin/python -m split_cloth.cli --project "$PWD" --config configs/coarse_v1.yaml select --run "$RUN"
.venv/bin/python -m split_cloth.cli --project "$PWD" prepare --run "$RUN" --expert-only
```

查看 expert 网格图后，准备符合注册校验的视觉参考 JSON。注册输入需要三个采样上的原始结束帧、采样上的第 4 阶段完成帧、四条非空 `stage_definitions`，并保留 `reviewer=gpt_visual`、`status=approved`。这里的 `approved` 仅是 expert 注册流程的状态；来源仍是 GPT，不代表人工验收。

```bash
.venv/bin/python -m split_cloth.cli --project "$PWD" register-expert --run "$RUN" --review /path/to/expert_visual_review.json
.venv/bin/python -m split_cloth.cli --project "$PWD" prepare --run "$RUN"
.venv/bin/python -m split_cloth.cli --project "$PWD" annotate --run "$RUN"
.venv/bin/python -m split_cloth.cli --project "$PWD" summary --run "$RUN"
```

### 续跑和变更

重新运行 `prepare`、`annotate` 时会校验素材/输入指纹。完成记录与输入、模型配置、提示版本和 expert 指纹一致时复用结果；执行异常不会被永久当作完成，可在剩余预算内重试。每条累计最多3次请求，用尽后不会自动重置；坏JSON和截断响应保留归档后才重试。供应商图片审核拒绝保留错误原因并人工处理。原响应存在而推理指纹发生变化时拒绝混用旧结果，应改用新运行名。

`annotate --limit N` 处理冻结清单前 N 条，适合先排查接口。并发和重试由配置控制，当前 `max_inflight=3`、`max_attempts=3`；不在命令行输出密钥。不要把 API 调用 token 数直接换算成未核实的费用。

## 运行产物

```text
runs/<RUN>/
  run.json                    运行配置和版本
  manifest.jsonl              固定100条目标清单
  pilot_split.json            calibration/heldout/expert划分
  expert_manifest.json        独立expert来源
  expert.json                 注册记录和GPT来源声明
  standard_catalog.json       标准图与阶段文字解释
  standards/                  从standard.zip提取的阶段参考图
  reference_expert/            expert素材及gpt_review.json
  episodes/<dataset>/episode_<index>/
    input_manifest.json       输入元数据、来源和指纹
    samples.jsonl             采样编号与原始帧/时间映射
    frames/                   采样图片
    pages.json                网格页清单
    montage/                  按时序拼接的网格图片
    preview.mp4               浏览器可播放的H264预览
    request.json              提示版本及请求元数据
    response.json             API原始响应缓存
    annotation.json           校验后的Qwen结果
    completion.json           可复用结果的指纹/完成记录
    stage.npy                 逐原始帧粗标签；未知为-1
    gpt_review.json            独立GPT视觉复核，若有
    review.json                独立人工复核sidecar，若有
    review_history.jsonl       人工修改历史，若有
    stage_reviewed.npy         人工通过后生成的标签，若有
  reports/
    preparation.json          素材准备记录
    summary.json              可机读处理汇总
    summary.md                可阅读汇总
    gpt_validation.json/.md   GPT抽样比较报告，不代表人工真值
    model_comparison.json/.md 模型同key配对对照，候选运行中生成
```

原始数据集只读使用：不会覆盖源视频、Parquet、原始 `stage` 或数据集元信息。`annotation.json` 与人工 `review.json` 分开，人工修正不覆盖模型原始结果。人工通过或修正通过才保存 `stage_reviewed.npy`；撤回到待复核/不适用时，该有效人工标签会移除。未审核的 `needs_review`、不适用和读取失败不进入有效训练标签，软件没有自动训练流程。

## GUI 和远程访问

当前远程 GUI：`http://127.0.0.1:53291`；当前本地转发：`http://127.0.0.1:18766`。

```bash
# 服务器：安全复用/启动，JSON输出会给出实际端口和SSH命令
python3 scripts/start_gui.py

# 服务器：需要时重启，只停止核验属于本项目的GUI
python3 scripts/start_gui.py --restart

# 本地：当前端口示例；服务若换端口，调整右侧53291
ssh -N -L18766:127.0.0.1:53291 h100-jackson
```

启动脚本以 `.venv/bin/python -m split_cloth.cli --project PROJECT serve --host 127.0.0.1 --port PORT` 启动，设置项目 `src` 的 `PYTHONPATH`，使用独立会话并追加 `logs/gui.log`。默认先核验 PID、进程命令、工作目录和 Linux 进程启动时间，再复用；PID 文件缺失时只查找当前用户下符合这些约束的项目 GUI。已有实例通过 2 秒健康检查后才返回成功。重启时优先复用原端口；原端口不可用时以 `socket.bind(..., 0)` 选择空闲端口，不停止任何占用该端口的其他服务。

直接 `serve` 适合前台调试，但不会自动管理 PID 文件或端口：

```bash
.venv/bin/python -m split_cloth.cli --project "$PWD" serve --host 127.0.0.1 --port 53291
```

GUI 包含轨迹列表与筛选、真实处理统计、视频和原始帧浏览、阶段时间线、网格分页/大图、expert 对照、Qwen 和 GPT 输出、人工三边界/完成帧/结论/备注保存。人工保存包含版本号；并发修改冲突返回 409，要求刷新后重新核对，避免静默覆盖。

读取 API 为 `/api/runs`、`/api/episodes`、`/api/episode`、`/api/frame`、`/api/video`、`/api/assets`、`/api/expert`；人工写入为 `POST /api/review`。GUI 通过 SSH 本地转发使用，无需暴露公网监听端口。

## 验证范围

当前软件测试记录：81 项通过。软件检查覆盖边界约束、输入/采样一致性、缓存与错误处理、Web 资产访问及人工复核等实现行为；这不构成视频阶段标注正确率。

GPT 自身视觉盲审是额外语义检查：先独立观察网格，再与 Qwen 结果比较。盲审文件应保留 `is_human_ground_truth=false` 和 `review_scope=independent_montage_blind_review`，不可伪装成人工确认。100 条全量标注质量、边界偏差和可接受成本仍需依据实际结果与人工抽检作结论。

严格首次松爪的原始帧、主视角遮挡中的夹爪状态、画外放置质量都可能超出当前输入可验证范围。遇到这些情况，保留不确定性和复核原因，不能凭任务顺序强行补全。

## 切换模型时的配置

当前提供 `coarse_v1.yaml`（原始qwen3-vl-plus基线）、`qwen3_8_27b.yaml`（qwen3.8-27b）和 `qwen3_8_max.yaml`（qwen3.8-max-0902固定快照）。通过全局 `--config` 选择，参数放在子命令前；已有run必须按其run.json记录选择原配置，新模型另建run，不能混用旧response。`provider.py` 当前显式使用 `enable_thinking=false` 和 JSON object，三模型输入与生成条件一致。模型官方支持更多thinking配置，但本轮没有开启或评估。

两份新运行分别为 `qwen3_8_27b_paired6_20261022`、`qwen3_8_max_paired6_20261022`，均仅6条配对诊断，不是新100条实验。模型比较用 `compare`，GPT参考比较用 `evaluate-gpt`；完整统计和样本覆盖分母见reports。精确结果汇总见README的模型对照表。
