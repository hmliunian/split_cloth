# 固定夹爪节点与区间拼图：六条三版本实测

本报告对照同一批六条既有轨迹的三种工程实现。自动accepted、节点一致性和边界变化均不是准确率。没有独立逐帧人工真值，不计算accuracy，也不能从边界更早或更接近反馈推断结果更正确。反馈稳定平台是固定候选节点估计，尚不是已校准的首次物理完全松爪帧。GPT审计属于模型复核，仍非人工金标。

固定六条匹配检查：`True`。

- 整段1 FPS：`qwen3_8_max_paired6_20261022`
- 事件候选＋连续帧精修：`event_probe_max_paired6_20261022`
- 固定节点＋区间1 FPS：`interval_nodes_max_paired6_20261022`

## 处理状态与已知用量

| 指标 | 整段1 FPS | 事件候选＋连续帧精修 | 固定节点＋区间1 FPS |
| --- | --- | --- | --- |
| 配对manifest条数 | 6 | 6 | 6 |
| 模型 | ["qwen3.8-max-0902"] | ["qwen3.8-max-0902"] | ["qwen3.8-max-0902"] |
| 状态计数 | {"accepted":4,"needs_review":1,"out_of_scope":1} | {"needs_review":2,"accepted":3,"out_of_scope":1} | {"needs_review":6} |
| 自动接受（非准确率） | 4 | 3 | 0 |
| 执行错误 | 0 | 1 | 0 |
| 独立保存响应 | 6 | 20 | 6 |
| 已记录尝试 | 6 | 21 | 6 |
| 已知 prompt_tokens | 52508 | 418305 | 108773 |
| 已知 completion_tokens | 2877 | 7711 | 11912 |
| 已知 total_tokens | 55385 | 426016 | 120685 |

新区间方案 / 整段1 FPS基线：**2.17902 倍**；新区间方案 / 上轮事件精修：**0.283287 倍**。

已知token按保存的响应和重试归档去重累计，缺少响应时才回退annotation聚合值。无用量失败请求成本未知。token倍率不是计费价格倍率。
六条token配对完整：`True`。

## 六条逐项对照

| Episode | 整段1 FPS 状态/边界 | 事件候选＋连续帧精修 状态/边界 | 固定节点＋区间1 FPS 状态/边界 | 节点/eligible/区间 | 新方案已知tokens |
| --- | --- | --- | --- | --- | --- |
| fold_cloth_v3__episode_000543 | accepted / [420, 520, 580] | needs_review / [] | needs_review / [] | 24 / 11 / 25 | 19237 |
| fold_cloth_v3__episode_006453 | accepted / [280, 440, 520] | needs_review / [] | needs_review / [] | 31 / 14 / 32 | 21568 |
| fold_cloth_v4__episode_002768 | accepted / [360, 420, 520] | accepted / [136, 211, 427] | needs_review / [] | 25 / 14 / 26 | 19075 |
| fold_cloth_v4__episode_004150 | needs_review / [] | accepted / [180, 274, 329] | needs_review / [] | 21 / 10 / 22 | 16970 |
| fold_and_hang_cloth_v1__episode_001383 | out_of_scope / [] | out_of_scope / [] | needs_review / [] | 29 / 15 / 30 | 23319 |
| fold_and_hang_cloth_v1__episode_003676 | accepted / [420, 520, 620] | accepted / [167, 384, 604] | needs_review / [] | 26 / 14 / 27 | 20516 |

## 固定节点及帧归属约束

原始节点选择回答的资格/帧归属检查：4/6 条通过。这里的失败包含已被程序拦截的回答矛盾，不等同于程序放行错误。

固定节点帧号映射：6/6 条通过；实验标签放行保护：6/6 条通过。映射通过不代表选对阶段。

5.2采用闭区间：选中节点422则422仍属于结束阶段，423开始下一阶段。第4阶段完成仅取已提供的区间末帧证据，不称为首次完成帧。

| Episode | 阶段 | 选中node_id | 固定原始帧 | 参与手 | 证据区间 | 约束错误 |
| --- | --- | --- | --- | --- | --- | --- |
| fold_cloth_v3__episode_000543 | 1 | a0015 | 400 | ["left","right"] | ["i0015","i0016","i0017"] | [] |
| fold_cloth_v3__episode_000543 | 2 | a0021 | 562 | ["right"] | ["i0018","i0019","i0020"] | [] |
| fold_cloth_v3__episode_000543 | 3 | None | None | [] | [] | [] |
| fold_cloth_v3__episode_006453 | 1 | a0015 | 235 | ["left","right"] | ["i0010","i0011","i0012","i0013","i0014"] | ["participant_not_open"] |
| fold_cloth_v3__episode_006453 | 2 | a0023 | 403 | ["left","right"] | ["i0015","i0016","i0017","i0018","i0019","i0020","i0021","i0022"] | [] |
| fold_cloth_v3__episode_006453 | 3 | a0028 | 488 | ["right"] | ["i0023","i0024","i0025","i0026","i0027"] | [] |
| fold_cloth_v4__episode_002768 | 1 | a0012 | 211 | ["left","right"] | ["i0007","i0008","i0009","i0010","i0011"] | [] |
| fold_cloth_v4__episode_002768 | 2 | a0016 | 338 | ["left","right"] | ["i0012","i0013","i0014","i0015"] | [] |
| fold_cloth_v4__episode_002768 | 3 | None | None | [] | [] | [] |
| fold_cloth_v4__episode_004150 | 1 | a0012 | 274 | ["left","right"] | ["i0008","i0009","i0010","i0011"] | [] |
| fold_cloth_v4__episode_004150 | 2 | a0015 | 329 | ["left","right"] | ["i0012","i0013","i0014"] | [] |
| fold_cloth_v4__episode_004150 | 3 | a0021 | 464 | ["left","right"] | ["i0015","i0016","i0017","i0018","i0019","i0020","i0021"] | [] |
| fold_and_hang_cloth_v1__episode_001383 | 1 | a0016 | 382 | ["left"] | ["i0013","i0014","i0015"] | [] |
| fold_and_hang_cloth_v1__episode_001383 | 2 | a0018 | 439 | ["left"] | ["i0016","i0017","i0018"] | [] |
| fold_and_hang_cloth_v1__episode_001383 | 3 | a0028 | 1088 | ["right"] | ["i0019","i0027"] | [] |
| fold_and_hang_cloth_v1__episode_003676 | 1 | a0014 | 384 | ["left","right"] | ["i0013"] | ["participant_not_open"] |
| fold_and_hang_cloth_v1__episode_003676 | 2 | a0020 | 505 | ["left","right"] | ["i0016","i0017","i0018","i0019"] | [] |
| fold_and_hang_cloth_v1__episode_003676 | 3 | a0023 | 595 | ["right"] | ["i0022"] | [] |

## 拒判、执行错误与节点诊断

| Episode | 新状态 | 复核原因 | alignment | 硬约束错误 |
| --- | --- | --- | --- | --- |
| fold_cloth_v3__episode_000543 | needs_review | ["Stage 4 removal not observed; video ends with cloth on table and no release node.","Stage 1 involves repeated spreading/grasping (intervals i0002-i0013) before the first actual longitudinal fold is completed at a0015.","all_stages_observed: all four stages must be observed","model_requested_review","model_supplied_review_reasons","stage_end_nodes[0]: evidence must belong to this stage before its selected end","stage_end_nodes[0]: evidence must include the interval ending at the selected node","stage_end_nodes[2].node_id: unresolved or unknown fixed node None","stage4_completion.observed: final stage completion must be observed independently","stage4_completion.interval_id: must identify an interval with its ending frame supplied","stage4_completion: completed remove activity evidence is required","stage4_completion: nonempty visual evidence is required"] | True | [] |
| fold_cloth_v3__episode_006453 | needs_review | ["stage_end_nodes[0].participant_arms: all participants must be open at the selected node","stage_end_nodes[0]: terminal interval must confirm completed operation with evidence","stage_end_nodes[2]: terminal interval must confirm completed operation with evidence"] | True | ["stage 1: participant_not_open"] |
| fold_cloth_v4__episode_002768 | needs_review | ["Stage 3 (cross_fold) not observed; video ends during remove phase without a corresponding release node.","Stage 4 completion cannot be confirmed as the final gripper release is missing from the provided nodes.","all_stages_observed: all four stages must be observed","model_requested_review","model_supplied_review_reasons","stage_end_nodes[2].node_id: unresolved or unknown fixed node None","stage4_completion.observed: final stage completion must be observed independently","stage4_completion.interval_id: must identify an interval with its ending frame supplied","stage4_completion: completed remove activity evidence is required","stage4_completion: nonempty visual evidence is required"] | True | [] |
| fold_cloth_v4__episode_004150 | needs_review | ["视频在阶段3横折动作进行中截断，未观察到横折完成及随后的夹爪松开节点","缺少阶段3结束边界及阶段4移除动作的完整证据","all_stages_observed: all four stages must be observed","model_requested_review","model_supplied_review_reasons","stage_end_nodes[2]: evidence must belong to this stage before its selected end","stage_end_nodes[2]: terminal interval must confirm completed operation with evidence"] | True | [] |
| fold_and_hang_cloth_v1__episode_001383 | needs_review | ["Stage 4 removal not observed; video ends with garment on table and arms retracted but no pickup action.","Stage 3 cross-fold evidence is weak; the final fold appears to be a longitudinal adjustment rather than a clear cross-fold into a small rectangle.","all_stages_observed: all four stages must be observed","model_requested_review","model_supplied_review_reasons","stage_end_nodes[1]: evidence must belong to this stage before its selected end","stage_end_nodes[1]: terminal interval must confirm completed operation with evidence","stage4_completion.observed: final stage completion must be observed independently","stage4_completion.interval_id: must identify an interval with its ending frame supplied","stage4_completion: completed remove activity evidence is required","stage4_completion: nonempty visual evidence is required"] | True | [] |
| fold_and_hang_cloth_v1__episode_003676 | needs_review | ["stage_end_nodes[0].participant_arms: all participants must be open at the selected node","stage_end_nodes[1]: terminal interval must confirm completed operation with evidence"] | True | ["stage 1: participant_not_open"] |

## 已保存的后续GPT审计（非人工真值）

审计不覆盖原模型输出，也不据此计算accuracy。完整审计对象保留在JSON内。

### fold_cloth_v3__episode_000543 · 事件候选＋连续帧精修

产物：`gpt_event_audit`；状态：`execution_incomplete`；审计者：`gpt_visual`。

| 阶段 | 类型 | 发现 | 源帧证据 |
| --- | --- | --- | --- |
| 3 | provider_image_inspection_failure | 阶段3精修被供应商图片审核以HTTP400拒绝；本条属于执行不完整，不能计作语义拒判或正确标注。 | [] |

### fold_cloth_v3__episode_006453 · 事件候选＋连续帧精修

产物：`gpt_event_audit`；状态：`reject_for_training`；审计者：`gpt_visual`。

| 阶段 | 类型 | 发现 | 源帧证据 |
| --- | --- | --- | --- |
| 1 | state_direction_contradiction | 精修以f201左右反馈降到近零解释为完全松开，与输入校准中higher表示张开的假设相反。原始读回left=0.00099446/right=0.00291889；不能由该解释确认释放。 | [200,201,202] |
| 2 | state_direction_contradiction | 精修将右反馈在f340降到0.00014833解释为释放，文字同时称fully closed/released baseline，解释自相矛盾。 | [339,340,341] |
| 4 | completion_id_text_conflict | 结构completion_sample_id=s000027映射f540，文字却反复说明应选s000029/f572；程序选择结构值，不能假定文字纠正已生效。 | [540,572] |

### fold_cloth_v4__episode_002768 · 事件候选＋连续帧精修

产物：`gpt_event_audit`；状态：`reject_for_training`；审计者：`gpt_visual`。

| 阶段 | 类型 | 发现 | 源帧证据 |
| --- | --- | --- | --- |
| 1 | preparation_mistaken_for_folding | 新结束帧136附近为取衣/抖开/摊衣，f140/160仍为展开上衣，f180/200再次抖开；不能视为第一次纵折完成。 | [120,140,160,180,200] |
| 2 | stage_sequence_shift | 新结束帧211附近衣服刚铺下，f260/280再次抖开，f320/340才出现第一侧纵折。新阶段2过早，局部松爪细节不能弥补全局阶段语义错位。 | [200,220,260,280,320,340] |

### fold_cloth_v4__episode_004150 · 事件候选＋连续帧精修

产物：`gpt_event_audit`；状态：`reject_for_training`；审计者：`gpt_visual`。

| 阶段 | 类型 | 发现 | 源帧证据 |
| --- | --- | --- | --- |
| 3 | strip_mistaken_for_final_rectangle | 新第三结束帧329位于f320/f340附近，此时画面仍为长条；不支持横折小矩形已经完成。原基线待复核，本轮转accepted暴露误接受。 | [300,320,340,360,380] |

### fold_and_hang_cloth_v1__episode_001383 · 事件候选＋连续帧精修

产物：`gpt_event_audit`；状态：`scope_agreement_only`；审计者：`gpt_visual`。

| 阶段 | 类型 | 发现 | 源帧证据 |
| --- | --- | --- | --- |
| None | scope_rejection_agreement | 新旧模型均将该条判为hang_cloth/out_of_scope；这只是任务适用性一致，不能计作原始帧边界准确率。 | [] |

### fold_and_hang_cloth_v1__episode_003676 · 事件候选＋连续帧精修

产物：`gpt_event_audit`；状态：`reject_for_training`；审计者：`gpt_visual`。

| 阶段 | 类型 | 发现 | 源帧证据 |
| --- | --- | --- | --- |
| 1 | preparation_mistaken_for_folding | 新第一结束帧167附近衣物仍在抖开与铺平；f180/200/220是展开衣物，f240和f320/340又抖开，不能在167结束第一侧纵折。 | [140,160,180,200,220,240,320,340] |
| 2 | stage_sequence_shift | 新第二结束帧384附近更接近第一侧纵折后的状态；精修不能代替正确的阶段语义判断。 | [360,380,400,420] |


无独立逐帧人工真值，本报告不计算准确率、不宣称准确率提升。
