# Split Cloth 标注与复核

按用户指定“固定夹爪节点→每区间1 FPS拼图→AI只选节点”实现与6条实测见 [INTERVAL_EXPERIMENT.md](INTERVAL_EXPERIMENT.md)。[打开本轮复核GUI](http://127.0.0.1:18766/?run_id=interval_nodes_max_paired6_20261022&key=fold_cloth_v4__episode_002768)。

新增夹爪事件实验分支及6条同模型实测见 [EVENT_EXPERIMENT.md](EVENT_EXPERIMENT.md)。当前全局事件候选方案出现阶段语义误选，尚不适合批量标注；原1 FPS流程继续作为独立基线。

用 Qwen 根据文字阶段定义、标准图和已标注示范，为折衣轨迹生成四阶段粗标签；在本地浏览器查看远程服务器上的视频、网格图、模型结果和人工修正。

当前项目位于 `h100-jackson:/gpfs/jiuquyun/home/jackson/xuran_projects/split_cloth`。本轮运行是 `pilot_100_20261022_v2`，调用百炼兼容接口的 `qwen3-vl-plus`。软件在 CPU 上准备图像和运行 GUI，推理由 API 完成，无需本地 GPU。

## 打开当前 GUI

服务器 GUI 当前监听 `127.0.0.1:53291`，本地 `18766` 的 SSH 转发已建立，可直接打开：

[http://127.0.0.1:18766](http://127.0.0.1:18766)

需要重新建立转发时，在本地终端运行并保持该终端打开：

```bash
ssh -N -L18766:127.0.0.1:53291 h100-jackson
```

远程服务重启后，以启动脚本返回的实际 `port` 为准，调整转发命令右侧的端口。

在服务器启动或复用本项目 GUI：

```bash
cd /gpfs/jiuquyun/home/jackson/xuran_projects/split_cloth
python3 scripts/start_gui.py
```

脚本返回 `{pid, port, url, ssh_command}`，默认复用已存在且核验属于本项目的 GUI。需要重启时使用 `python3 scripts/start_gui.py --restart`；它只停止经过进程命令、工作目录和进程启动时间核验的本项目 GUI。服务始终只监听 `127.0.0.1`。

运行信息保存在 `logs/gui.pid`、`logs/gui_port.txt`，服务输出追加至 `logs/gui.log`。脚本会在启动后进行 2 秒健康检查；未通过时返回错误并提示查看日志。

## 本轮 100 条试标

| 数据集 | 目标条数 | calibration | heldout |
|---|---:|---:|---:|
| `fold_cloth_v3` | 40 | 24 | 16 |
| `fold_cloth_v4` | 30 | 18 | 12 |
| `fold_and_hang_cloth_v1` | 30 | 18 | 12 |
| 合计 | 100 | 60 | 40 |

选择按视频时长四分位分层，种子为 `20261022`；清单和划分写入运行目录后冻结。独立 expert 为 `fold_cloth_v3 / episode_000000`，不在这 100 条目标中。

Expert 阶段解释和边界来自 GPT 自身视觉判读，是粗语义参考，尚不是人工金标。本轮另外保留 6 条独立 GPT 网格图盲审，以各 episode 的 `gpt_review.json` 归档。模型间一致性只能提供复核线索，不能代替独立人工真值，也不能据此报告全量准确率。实际处理状态和 token 汇总请查看运行目录的 `reports/summary.json`、`reports/summary.md`。

实际首轮：22 条模型自动接受、77 条模型待复核、1 条百炼图片审核拒绝。6 条 GPT 盲审中，唯一可直接比较的自动接受样本三个边界偏晚 3 / 4 / 4 秒；另有挂衣任务的适用范围分歧。因此本轮结果用于辅助复核，尚不能直接当作准确训练标签。比较报告为 `reports/gpt_validation.json` 与 `.md`。

## 模型切换与实测对照

在相同6条GPT盲审样本、相同prompt/expert/主视角1FPS网格下，仅切换模型；均关闭thinking。

| 模型 | 自动接受 | 模型待复核 | 排除不适用 | 各自可比的接受样本中，三边界均在GPT粗参考±1秒内 |
|---|---:|---:|---:|---:|
| qwen3-vl-plus | 1/6 | 5/6 | 0/6 | 0/1 |
| qwen3.8-27b | 1/6 | 5/6 | 0/6 | 0/1 |
| qwen3.8-max-0902 | 4/6 | 1/6 | 1/6 | 3/4 |

Max的12个可比边界中11个在±1秒内，另一个提前4秒；挂衣样本的适用性判断与GPT一致。三个模型接受的样本集合不同，不能把各自MAE直接当作完整准确率比较。两模型共同接受的同一条episode_003676上，原模型与GPT的MAE为3.667秒，Max为0.667秒。本轮更建议试用Max固定快照，仍需人工验证。原100条结果保留，候选模型各测试6条，尚未重跑100条。

- 27B配置：`configs/qwen3_8_27b.yaml`；运行：`qwen3_8_27b_paired6_20261022`。
- Max配置：`configs/qwen3_8_max.yaml`；运行：`qwen3_8_max_paired6_20261022`。
- 精确模型ID以各配置及request.json为准；旧运行续跑必须使用其原配置，不能直接替换模型并混用缓存。
- 官方能力：[Qwen3.8-27B](https://help.aliyun.com/zh/model-studio/qwen3-8-27b)、[Qwen3.8-Max](https://help.aliyun.com/zh/model-studio/qwen3-8-max)。

```bash
.venv/bin/python -m split_cloth.cli --project "$PWD" --config configs/qwen3_8_max.yaml evaluate-gpt --run qwen3_8_max_paired6_20261022
.venv/bin/python -m split_cloth.cli --project "$PWD" compare --baseline pilot_100_20261022_v2 --run qwen3_8_max_paired6_20261022
```

## 实际方法

1. 仅取 `base_0_rgb` 主视角，按 1 FPS 抽帧，并补入最后一帧。
2. 依时间顺序拼成 3 列 × 4 行网格；每格标明采样编号、时间和原始帧号，分页重叠 2 个采样点。
3. 将阶段文字、标准图、独立 expert 的阶段网格和目标轨迹全部网格页放入同一次 Qwen 请求。
4. Qwen 为三个边界选择相邻的「尚未完成 / 已完成」采样编号，并返回第 4 阶段完成证据。
5. 程序按采样表映射到原始帧、校验顺序和适用性。完成采样属于前一阶段，下一原始帧进入后一阶段；不确定的结果保留未知标签并进入复核。

本版没有局部窗口精修或多视角推理。1 FPS 提供秒级采样的粗标注，不保证首次完全松开的原始帧。原始视频、Parquet 和旧 `stage` 从不覆盖；人工复核存为独立 sidecar。

## 常用命令

以下命令在服务器项目根目录执行。独立 `.venv` 已安装依赖：

```bash
cd /gpfs/jiuquyun/home/jackson/xuran_projects/split_cloth
export PYTHONPATH="$PWD/src"
RUN=pilot_100_20261022_v2

# 查看本轮状态；会更新 reports/summary.json 和 reports/summary.md
.venv/bin/python -m split_cloth.cli --project "$PWD" summary --run "$RUN"

# 根据已保存的 GPT 盲审生成比较报告
.venv/bin/python -m split_cloth.cli --project "$PWD" evaluate-gpt --run "$RUN"

# 继续标注；相同输入及配置会复用已完成结果
.venv/bin/python -m split_cloth.cli --project "$PWD" annotate --run "$RUN"

# 只处理清单开头若干条，用于接口排查
.venv/bin/python -m split_cloth.cli --project "$PWD" annotate --run "$RUN" --limit 3
```

创建新运行、准备素材和注册 expert 的完整命令见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。目前实际 CLI 只有 `select`、`prepare`、`register-expert`、`annotate`、`summary`、`evaluate-gpt`、`compare`、`serve`。

## GUI 复核

左侧可按运行、数据集、状态和 episode 搜索；主区可切换视频、原始帧和时序网格。右侧编辑三个阶段结束帧、第 4 阶段完成帧、结论和备注，再保存人工复核。帧号使用原始视频的 **0 基索引**。

`← / →` 前后移动一帧，`Shift + ← / →` 移动约一秒；视频模式下空格播放或暂停；`Ctrl / ⌘ + S` 保存。边界定位和「取当前帧」可辅助修正。

模型结果、人工作出的复核以及 GPT 盲审分别保留。`needs_review`、不适用和读取失败样本不生成有效训练阶段；软件不自动启动训练。当前测试记录为 **81 项通过**，不等同于标注准确率。

## 文档导航

- [IMPLEMENTATION.md](IMPLEMENTATION.md)：已实现模块、运行命令、产物、恢复和限制。
- [agen.md](agen.md)：原开发总计划与里程碑。
- [boundary.md](boundary.md)：阶段语义、完成条件和粗边界约定。
- [sampling.md](sampling.md)：采样、时序网格与分页规格。
- [vlm.md](vlm.md)：提示、JSON 输出、校验和重试规格。
- [data.md](data.md)：数据清单和产物规格。
- [validation.md](validation.md)：试标及验收计划。
- [runbook.md](runbook.md)：原运行与交付计划。

原规格中的后续设想不表示已实现；实际能力以本 README、IMPLEMENTATION 和 CLI `--help` 为准。
