# 总计划与开发总纲

版本：`coarse_v1.0`；日期：2026-10-22。

## 1. 目标与范围

将已有 LeRobot 数据中的适用叠衣服 episode 标为四个阶段。方法为：主视角 `base_0_rgb` → 1 FPS 时间采样 → 带原始帧号与时间戳的时序网格图 → Qwen 整段一次推理 → 程序校验 → 粗阶段标签。

本版以 VLM 为主要算法，不训练 TCN、不做 DTW/UVD，不从夹爪事件生成候选，不启动局部窗口精修或腕部视角复核。`state/actions` 仅用于既有数据完整性检查和来源保留，不参与本版阶段判定。

### 已确定的需求

1. 一代臂优先，随后处理二代臂小推车。
2. 主视角、1 FPS；用户确认使用按时间顺序排列的网格图。
3. 规范仍以本阶段动作完成、参与动作的夹爪完全松开作为完成语义。
4. 接受粗标注；1 FPS 结果不承诺原视频逐帧边界准确。
5. 先做 100 条试标，再决定生产运行参数与批量规模。
6. 保留源数据，以独立产物保存新标注。

## 2. 已核对的输入

项目目录现有：`standard.zip`、`data_path.txt`、`Qwen_api.txt`。`standard.zip` 内含 `Untitled.md` 和阶段示意图；正文存在重复段落，引用时取一份规范语义，原文件保留用于溯源。

| 数据集 | episode 数 | 元数据 FPS | 处理顺序 |
|---|---:|---:|---|
| fold_cloth_v3 | 7090 | 20 | P1 |
| fold_cloth_v4 | 4289 | 20 | P2 |
| fold_and_hang_cloth_v1 | 5548 | 20 | P3，先识别实际任务 |
| fold_cloth_v3_pushcar_batch_20260923 | 11505 | 10 | P4 |
| fold_cloth_v3_pushcar_batch_20260924_20260928 | 2494 | 10 | P5 |
| fold_cloth_v3_pushcar_batch_20260929_20261009 | 5663 | 10 | P6 |

一代臂合计 16927；二代臂合计 19662；总计 36589。以上来自 2026-10-22 只读盘点，运行前重新核对并冻结清单。

`fold_and_hang_cloth_v1` 的任务字符串同时包含叠衣、裤子和挂衣，不能依靠字符串将全部 episode 认定为适用四阶段。`pushcar` 是机器人类型，不是视频动作类别。

## 3. 算法与模型选择

首个可复现基线采用 `Qwen/Qwen3-VL-30B-A3B-Instruct`，与 LeRobot SARM 的标注实现对应。另以 `Qwen/Qwen3-VL-8B-Instruct` 作为成本/效果对照。第一轮不将 Thinking 或扩大模型视为默认提升，使用同一试标集实测。

可选后续对照为 `MCG-NJU/TimeLens2-8B`。其原生视频时间定位接口与当前网格多图接口不同，若加入对照必须分别记录输入形式，不能将接口变更带来的效果归因于模型本身。本版首个交付不依赖此模型。

复用 SARM 的固定标签列表、JSON 结构、结果保存与可视化思路。其原版 1 FPS 视频输入、整秒时间戳以及 LeRobot v3 读取方式需要替换，不能直接当作本项目精标实现。

模型调用细则见 [vlm.md](vlm.md)，采样和页面细则见 [sampling.md](sampling.md)。

## 4. 建议模块边界

以下为待实现布局，不表示文件已经存在：

```text
src/split_cloth/
  cli.py                 # 统一命令入口
  config.py              # 配置校验、spec/prompt/input hash
  inventory.py           # 数据清单与完整性检查
  lerobot_v21.py         # meta JSONL 与每 episode 文件适配
  standards.py           # 从压缩包读取图示、生成可追踪阶段目录
  sampling.py            # 源时间轴采样与 frame 映射
  montage.py             # 单视角时序网格图、分页
  providers/             # local/API 调用适配，不存储密钥
  annotation.py          # prompt 组装、响应 schema、状态流转
  boundaries.py          # sample -> frame、闭区间、未知标签
  storage.py             # episode 原子落盘、缓存、恢复
  evaluation.py          # 与人工真值比较、错误分类
configs/coarse_v1.yaml
tests/                   # 时间映射、分页、边界、拒判与恢复测试
```

## 5. 初始配置契约

```yaml
spec_version: coarse_v1.0
precision: sampled_1fps
camera_key: base_0_rgb
sample_fps: 1.0
include_last_frame: true
montage:
  columns: 3
  rows: 4
  overlap_samples: 2
  tile_image_size: [224, 224]
  tile_label_height_px: 32
  page_header_height_px: 48
  gutter_px: 8
  reading_order: row_major
model:
  backend: local  # api 也是实现选项；不根据凭据文件猜测服务商
  model_id: Qwen/Qwen3-VL-30B-A3B-Instruct
  temperature: 0
  max_attempts: 3
  max_output_tokens: 2048
pilot:
  total_episodes: 100
  calibration_episodes: 60
  heldout_episodes: 40
batch:
  episodes_per_batch: 500
  max_inflight: 1
output:
  write_source: false
  unknown_stage: -1
  boundary_policy: first_observed_complete_sample
  interval_convention: closed
```

`temperature=0` 在服务支持时设置，不表示模型完全可重复。模型实际输入分辨率、图像数/token 上限、响应结构能力、显存与吞吐量必须在 M0/M2 测量后冻结。API 模式通过环境变量提供 endpoint、model ID、凭据来源；密钥不进入 YAML、命令行、JSON、日志或缓存键。

## 6. 开发里程碑与完成条件

| 阶段 | 工作 | 完成条件 |
|---|---|---|
| M0 规格与输入冻结 | 核对来源、FPS、主视角、规范图、阶段解释、运行环境与模型能力 | 产生 manifest、阶段目录和版本/hash；仍不清晰的规范解释列为待确认 |
| M1 确定性预处理 | v2.1 读取、时间采样、网格分页、映射表 | 时间映射、首尾、重叠页与损坏输入测试通过；人工检查至少 3 条 |
| M2 单条端到端 | 一次请求输入整条 episode 全部网格页，解析固定槽位候选 | 正常、缺阶段、越界/非法 sample ID 等样例均按契约落盘 |
| M3 100 条试标 | 60 条调提示与参数；40 条保留评估 | 产出 [validation.md](validation.md) 指定报告，冻结生产配置 |
| M4 一代臂分批 | 顺序处理三个一代臂数据集，每批默认 500 条 | 有断点恢复、批次统计、失败队列和复核结果；新标签可溯源 |
| M5 二代臂适配 | 重新验证 10 FPS、相机布局、机器人类型与模型域差异 | 小批验证通过后运行三批 pushcar；不直接复用未经验证的接受率 |
| M6 交付 | 导出 sidecar 标注、逐帧数组、可视化、评估与来源 | 可重现，无遗漏状态，无源数据覆盖，所有清单计数闭合 |

## 7. 执行原则

- 本版不把复杂模型训练作为依赖；先测量网格 VLM 基线。
- 不依靠模型自报置信度接受样本；采用证据引用、结构/语义校验与人工真值评估。
- 异常任务、遮挡、缺阶段及不确定边界有明确拒判路径；不强填四阶段。
- 相机/FPS/拼图布局/prompt/阶段语义/边界策略变更后生成新配置 hash 和新 run；不能混入旧缓存。
- 参考 [runbook.md](runbook.md) 落实批处理、失败恢复与交付。

## 8. 原始参考

- Qwen3-VL：<https://github.com/QwenLM/Qwen3-VL>
- LeRobot SARM：<https://github.com/huggingface/lerobot/blob/main/docs/source/sarm.mdx>
- SARM annotator：<https://github.com/huggingface/lerobot/blob/main/src/lerobot/data_processing/sarm_annotations/subtask_annotation.py>
- TimeLens2：<https://github.com/MCG-NJU/TimeLens2>
- VideoChat-R1.5：<https://github.com/OpenGVLab/VideoChat-R1>

借鉴来源的实现思路不构成本项目精度承诺；效果以冻结的本域试标集为准。
