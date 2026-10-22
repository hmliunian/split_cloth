# 数据读写与产物契约

版本：`coarse_v1.0`。源数据目录只读，标注先以 sidecar 方式保存。

## 1. 输入位置

数据根目录：`/gpfs/jiuquyun/datasets/shenzhen/lerobot_eef`。实际数据集列表、episode 数与顺序见 [agen.md](agen.md)。项目依据文件为 `data_path.txt` 与 `standard.zip`。

源数据为 LeRobot v2.1，常见布局：

```text
<dataset>/meta/info.json
<dataset>/meta/episodes.jsonl
<dataset>/meta/tasks.jsonl
<dataset>/data/chunk-XXX/episode_XXXXXX.parquet
<dataset>/videos/chunk-XXX/base_0_rgb/episode_XXXXXX.mp4
```

路径必须按 `info.json` 的 `data_path/video_path/chunks_size` 计算，不能假设所有数据都在 chunk-000。现有一代臂元数据包含 `stage`；它是否为占位值、如何编码以及是否可信，需读取实际 Parquet 后确认。新 sidecar 不依赖或覆盖旧列。

pushcar 视频目录指向另一个位置的符号链接。保存逻辑路径与 `realpath`，校验目标可读；导出元信息应记录这一依赖，不删除链接目标。

## 2. inventory 与冻结 manifest

每个 episode 生成一个稳定键：`dataset_id + episode_index`；`dataset_id` 默认取数据集目录名，`episode_id` 格式为 `episode_{episode_index:06d}`。不能仅以全局 episode_index 命名缓存。

`manifest.jsonl` 必含：

- `dataset_id`、数据集绝对路径、`episode_index`、`robot_type`；
- 源视频逻辑路径/真实路径、Parquet 路径、camera key；
- `source_fps`、`frame_count`、`duration_s`、meta 声明帧数；
- 实测视频/Parquet 可读状态与时间轴一致性；
- 文件大小、mtime_ns、meta hash、视频 SHA-256；
- `task_text`（仅作来源提示，不作为真实任务分类）；
- `existing_stage_schema` 与采样检查结果；
- `selected_for_run`、排除原因或检查错误。

API/模型调用前以流式方式计算每个已选视频的 SHA-256，计入输入指纹；缓存命中时先检查 stat 信息，再按 run 策略复核 hash。stat 改变或 hash 不匹配必须重新准备输入。大规模扫描/全量 hash 的开销单独计入预处理耗时。

`standard.zip`、内含规范正文和阶段图需保存 hash；规范解释目录 `standard_catalog.json` 也单独保存版本和 hash。

## 3. 运行目录

```text
runs/<run_id>/
  run.json                    # 配置快照、版本、命令、代码commit/hash
  manifest.jsonl              # 冻结的数据清单
  standard_catalog.json       # 阶段语义与规范图片来源
  pilot_split.json            # 若为试标，固定60/40与选择种子
  episodes/<dataset_id>/<episode_id>/
    input_manifest.json
    samples.jsonl
    pages.json
    montage/page_000.png
    request.json              # 无密钥、无鉴权头
    response.json             # 脱敏原始响应及用量
    annotation.json
    stage.npy                 # int64，长度N，未知=-1
    completion.json           # 校验通过后最后原子提交
  logs/events.jsonl
  reports/summary.json
  reports/summary.md
  reports/review_queue.jsonl
  reports/errors.jsonl
```

`run_id` 包含可读时间和配置 hash，避免不同模型、提示或规则混写。最终目录布局可由实现调整，但记录身份/映射/结果/状态的职责不能省略。

## 4. annotation.json

所有持久化 `segments` 使用闭区间 `[start_frame,end_frame]`，长度为 `end_frame-start_frame+1`；源帧号从 0 起。该约定与 boundary.md 一致。

本例仅说明字段，不是真实标签：

```json
{
  "spec_version": "coarse_v1.0",
  "precision": "sampled_1fps",
  "dataset_id": "fold_cloth_v3",
  "episode_index": 0,
  "frame_count": 806,
  "camera_key": "base_0_rgb",
  "status": "needs_review",
  "task_type": "unknown",
  "applicable": null,
  "all_stages_observed": false,
  "needs_review": true,
  "review_reasons": ["illustrative_example_only"],
  "stage4_completion": {"observed": false, "completion_sample_id": null, "evidence_text": "示例，尚无可用终态证据"},
  "boundaries": [],
  "segments": [{"stage": -1, "start_frame": 0, "end_frame": 805}],
  "source_video_sha256": "<hash>",
  "input_hash": "<hash>",
  "model_id": "<resolved-model-id>",
  "model_revision": "<resolved-revision-or-provider-version>",
  "prompt_version": "<version>",
  "prompt_hash": "<hash>",
  "standard_catalog_hash": "<hash>",
  "config_hash": "<hash>"
}
```

模型原始响应格式见 [vlm.md](vlm.md)。原始模型布尔值与候选完整保存在 `response.json`/`model_output`，不能覆盖为最终程序判断。规范化 `annotation.json.needs_review` 定义为 `status == needs_review`；`out_of_scope/unreadable` 时为 false，其原因仍保留。队列和训练导出按最终 `status` 路由，不能依据原始模型的 needs_review 开关。规范化结果的 `boundaries` 除原始 proposal 外，还包括 `coarse_end_frame`、`next_stage_start_frame`、L/R 的源帧/时间，以及估计边界区间端点。`stage4_completion` 另存已校验的终态 sample ID、映射源帧/时间与可见证据，不参与三个阶段间边界的排序数量。所有换算由程序完成。

`applicable` 为 true/false/null；task_type 为 `fold_cloth/fold_pants/hang_cloth/mixed/other/unknown`。机器人型号单独取元数据 `robot_type`。仅经阶段目录确认适用的任务可自动接受；其他任务进入 out_of_scope 或 needs_review。

## 5. 原子写入、状态与恢复

- 状态流程：`pending → prepared → requested → validated → completed`；失败进入 `retryable_error` 或带原因的最终结果状态。
- `completed` 是运行处理完成，和语义 `accepted` 分开。out_of_scope、needs_review、unreadable 都可在产物结构有效后完成处理；`completion.json` 不等于精度或语义验收通过。
- 同一 episode 使用锁/唯一任务租约防止两个 worker 同时提交。
- 产物先写临时文件，校验后原子 rename；`completion.json` 最后写入。
- 恢复时不能仅凭文件存在跳过；需验证完成标记、input/config/prompt/model hash 与产物一致性。
- API 已响应但写盘中断可能造成重复计费；优先恢复已落盘的 response，服务支持幂等键时使用，但不假设所有兼容接口都支持。
- attempts 计数与每次请求结果必须持久化；同一逻辑请求跨进程恢复累计最多 3 次总尝试，重启不能清零。耗尽记 `execution_error=retry_exhausted`、停止自动派发，并计入运行失败清单；只有显式新 run/记录原因的重置才能再次请求。
- 失败重试不得产生两个互相矛盾的最终结果；保留 attempts 记录与最终选择依据。

## 6. 逐帧数组与导出

`stage.npy` 为新 sidecar，dtype=int64，shape=(N,)。accepted 使用 1..4；其他状态默认全部 -1。数组覆盖完整 episode 不表示有完整有效标签。

后续若训练需要写入 Parquet，单独实现 `export`：写到新的数据集版本，明确 `stage` 编码、未知 mask、features 与统计信息，验证视频链接与帧数一致。此步骤不修改原始源目录。默认训练候选只选 accepted；未知值必须通过 mask 排除。

## 7. 凭据与可追踪性

`Qwen_api.txt` 仅作为现有凭据来源线索。读取后只向 provider 适配器传递必要密钥；不打印、复制到文档、写入请求快照或放进 URL query。endpoint 与 model ID 必须显式配置并预检，不从文件名猜测。

日志保存 request_id、状态码、延迟、输入页数、token/计费用量、attempt、错误分类。错误体中可能含鉴权信息，持久化前脱敏。上传研究或故障反馈时不携带原始凭据。
