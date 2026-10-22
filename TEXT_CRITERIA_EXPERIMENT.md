# 文字阶段判据与任一夹爪张开节点

分支 `experiment/gripper-event-refinement`；实测运行 `text_criteria_anyopen_max_paired6_20261022`。旧运行保留。

## 已实现的流程

1. 传统检测器保留任一夹爪的稳定张开事件作为关键帧，不要求双手同时张开。连续处于张开状态不逐帧建点；闭合事件仍在检测器内部用于方向截止和状态计算，但不切本轮区间。`node_policy: any_open` 可配置，旧默认开闭节点流程兼容。
2. GPT 在本次 Codex 会话中查看四张阶段参考图和四页参考时序，离线归纳动作、完成条件、易混淆排除条件及顺序，保存为 `configs/fold_cloth_text_criteria_v1.json`。来源图及文档的 SHA256 留档；没有抄入 expert 边界、帧号、时长比例或待标视频答案。生成一次，六条共用同一文字资产。
3. VLM 只接收该文字判据、固定张开节点表及 TARGET 按时序排列的区间拼图；不发送任何标准图或 expert 图片。区间内部独立 1 FPS 采样并保留原始首尾。一个阶段可以跨多个节点；任意张爪不自动推进阶段。
4. VLM 逐区间记录可见动作及是否完成，再从固定节点选阶段末。程序检查所有区间的顺序、操作所属阶段、实际完成字段、节点递增、证据不越过所选边界、参与手状态等；输出 `temporal_validation`。检查的是模型陈述一致性，不能代替视觉语义验证。

5.2 边界规则保持：衣物已完成但仍抓持属于本阶段；完全松开节点为本阶段结束帧，下一原始帧进入下一阶段。传统节点是反馈平台估计，尚未校准首次物理完全松开的精确帧。

## 六条结果

同六条轨迹、同一 `qwen3.8-max-0902`，6 个响应、6 次请求、无接口执行错误。参考图实际数量为 0（6/6 独立核对），10–15 个张开节点、11–16 个区间、11–17 张 TARGET 图；全部观察完整按时序，原始节点到帧映射 6/6 一致。

| 项目 | 上轮图片参考＋开闭节点 | 本轮文字判据＋张开节点＋时序检查 |
| --- | ---: | ---: |
| 自动 accepted（不是正确条数） | 0 | 3 |
| needs_review | 6 | 3 |
| total_tokens | 120,685 | 70,742 |

本轮已知 token 少 41.38%；不等于费用比例。三项策略同时改变，不能把结果变化归因于单项。

三条自动 accepted 仍不能直接用作正确标签：v4/2768 的 426 帧仍呈长条，与模型“横折后紧凑衣物包”解释矛盾；v4/4150 的前两边界附近仍出现重复提衣/铺开，274 帧形态不支持已成窄长条；mixed/3676 的第一折与整理有歧义。独立的 `gpt_review.json` / `gpt_text_criteria_audit.json` 记录抽样可见疑点，没有修改原模型回答。这些是 GPT 定点复核，不是盲审或人工金标；不报告准确率提升。

六条 `training_ready=false`。三条 accepted 保存候选阶段数组，三条 needs_review 数组为 -1；模型状态、GPT 意见、人工复核独立显示。

## 查看和复现

[新运行 GUI](http://127.0.0.1:18766/?run_id=text_criteria_anyopen_max_paired6_20261022&key=fold_cloth_v4__episode_002768) 的参考区域显示文字动作、完成和排除条件、时序规则与来源，不加载 expert 图片。下方可对照 VLM 原始选择与 GPT 视觉疑点，定位原始帧；人工复核仍独立保存。

```bash
export PYTHONPATH="$PWD/src"
.venv/bin/python -m split_cloth.cli --project "$PWD" --config configs/text_criteria_anyopen_max.yaml interval-probe --run NEW_RUN --baseline qwen3_8_max_paired6_20261022 --prepare-only
# 去掉 --prepare-only 执行；判据/代码/输入变化须使用 NEW_RUN，不能复用旧响应。
python3 scripts/report_text_criteria_probe.py --project "$PWD"
```

完整结果见 [实测报告](experiments/text_criteria_anyopen_max_paired6_20261022/report.md)。259 项测试、60 个子测试通过；GUI 已验证文字判据、阶段时间线和节点跳帧。
