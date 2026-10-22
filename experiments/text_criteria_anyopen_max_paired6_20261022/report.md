# 文字阶段判据＋任一手张开节点：六条工程对照

同六条既有轨迹、同一模型的工程对照。本轮同时改变三项：参考图片改为GPT离线文字判据、开闭节点改为任一手opening-only节点、启用全区间时序一致性校验。无法将结果变化归因于其中某一项。自动状态、节点映射一致性、时序校验和GPT复核均不等于准确率；没有独立逐帧人工真值，不宣称准确率提升。节点仍是实测反馈平台估计，不是首次物理完全松爪帧的人工金标。

上一版：`interval_nodes_max_paired6_20261022`；本轮：`text_criteria_anyopen_max_paired6_20261022`。

固定六条匹配：`True`；六条已完成且training_ready=false：`True`。

## 请求、状态与已知用量

| 指标 | 图片参考＋开闭节点 | 文字判据＋opening-only＋时序gate |
| --- | --- | --- |
| manifest条数 | 6 | 6 |
| 已生成annotation | 6 | 6 |
| 模型身份 | ["qwen3.8-max-0902"] | ["qwen3.8-max-0902"] |
| 状态（不是准确率） | {"needs_review":6} | {"needs_review":3,"accepted":3} |
| 阶段产物状态 | {"annotated":6} | {"annotated":6} |
| 执行错误 | 0 | 0 |
| 独立保存响应 | 6 | 6 |
| 已记录请求尝试 | 6 | 6 |
| prompt_tokens | 108773 | 63130 |
| completion_tokens | 11912 | 7612 |
| total_tokens | 120685 | 70742 |

本轮 / 上版已知total_tokens：`0.586171`；token配对完整：`True`。

实际保存响应及重试归档按request_id或内容去重；未知用量不填0。prepared_for_selection表示尚未请求，annotation_pending表示响应已保存但尚未落标注。token比例不等于计费金额比例。

## 逐条产物与候选

| Episode | 上版状态/候选帧 | 本轮状态/候选帧 | 节点/区间/图 | 参考图 | 仅open changes | 观察全序 | 时序检查 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fold_cloth_v3__episode_000543 | needs_review / [400, 562, None] | needs_review / [400, 485, 562] | 11 / 12 / 12 | 0 | True | True | needs_review |
| fold_cloth_v3__episode_006453 | needs_review / [235, 403, 488] | needs_review / [213, 351, 351] | 14 / 15 / 15 | 0 | True | True | needs_review |
| fold_cloth_v4__episode_002768 | needs_review / [211, 338, None] | accepted / [211, 338, 426] | 14 / 15 / 15 | 0 | True | True | passed |
| fold_cloth_v4__episode_004150 | needs_review / [274, 329, 464] | accepted / [181, 274, 378] | 10 / 11 / 11 | 0 | True | True | passed |
| fold_and_hang_cloth_v1__episode_001383 | needs_review / [382, 439, 1088] | needs_review / [439, 826, 1088] | 15 / 16 / 17 | 0 | True | True | needs_review |
| fold_and_hang_cloth_v1__episode_003676 | needs_review / [384, 505, 595] | accepted / [384, 505, 603] | 14 / 15 / 15 | 0 | True | True | passed |

## 原始节点提议与固定原始帧映射

候选均来自原始模型回答；待复核时仍保留，不用上版结果回填或移动节点。映射通过不证明所选阶段正确。

| Episode | 版本 | 阶段 | 原始node_id | 固定帧 | 参与手 | 证据区间 | 约束问题 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fold_cloth_v3__episode_000543 | previous | 1 | a0015 | 400 | ["left","right"] | ["i0015","i0016","i0017"] | [] |
| fold_cloth_v3__episode_000543 | previous | 2 | a0021 | 562 | ["right"] | ["i0018","i0019","i0020"] | [] |
| fold_cloth_v3__episode_000543 | previous | 3 | None | None | [] | [] | [] |
| fold_cloth_v3__episode_000543 | candidate | 1 | a0008 | 400 | ["left","right"] | ["i0006","i0007","i0008"] | [] |
| fold_cloth_v3__episode_000543 | candidate | 2 | a0009 | 485 | ["left","right"] | ["i0008","i0009"] | [] |
| fold_cloth_v3__episode_000543 | candidate | 3 | a0010 | 562 | ["right"] | ["i0009","i0010"] | [] |
| fold_cloth_v3__episode_006453 | previous | 1 | a0015 | 235 | ["left","right"] | ["i0010","i0011","i0012","i0013","i0014"] | ["participant_not_open"] |
| fold_cloth_v3__episode_006453 | previous | 2 | a0023 | 403 | ["left","right"] | ["i0015","i0016","i0017","i0018","i0019","i0020","i0021","i0022"] | [] |
| fold_cloth_v3__episode_006453 | previous | 3 | a0028 | 488 | ["right"] | ["i0023","i0024","i0025","i0026","i0027"] | [] |
| fold_cloth_v3__episode_006453 | candidate | 1 | a0006 | 213 | ["left"] | ["i0004","i0005","i0006"] | [] |
| fold_cloth_v3__episode_006453 | candidate | 2 | a0010 | 351 | ["right"] | ["i0007","i0008","i0009","i0010"] | [] |
| fold_cloth_v3__episode_006453 | candidate | 3 | a0010 | 351 | ["left","right"] | ["i0010"] | [] |
| fold_cloth_v4__episode_002768 | previous | 1 | a0012 | 211 | ["left","right"] | ["i0007","i0008","i0009","i0010","i0011"] | [] |
| fold_cloth_v4__episode_002768 | previous | 2 | a0016 | 338 | ["left","right"] | ["i0012","i0013","i0014","i0015"] | [] |
| fold_cloth_v4__episode_002768 | previous | 3 | None | None | [] | [] | [] |
| fold_cloth_v4__episode_002768 | candidate | 1 | a0008 | 211 | ["left","right"] | ["i0005","i0006","i0007"] | [] |
| fold_cloth_v4__episode_002768 | candidate | 2 | a0010 | 338 | ["left","right"] | ["i0008","i0009"] | [] |
| fold_cloth_v4__episode_002768 | candidate | 3 | a0012 | 426 | ["left","right"] | ["i0010","i0011"] | [] |
| fold_cloth_v4__episode_004150 | previous | 1 | a0012 | 274 | ["left","right"] | ["i0008","i0009","i0010","i0011"] | [] |
| fold_cloth_v4__episode_004150 | previous | 2 | a0015 | 329 | ["left","right"] | ["i0012","i0013","i0014"] | [] |
| fold_cloth_v4__episode_004150 | previous | 3 | a0021 | 464 | ["left","right"] | ["i0015","i0016","i0017","i0018","i0019","i0020","i0021"] | [] |
| fold_cloth_v4__episode_004150 | candidate | 1 | a0004 | 181 | ["left","right"] | ["i0002","i0003"] | [] |
| fold_cloth_v4__episode_004150 | candidate | 2 | a0006 | 274 | ["left","right"] | ["i0004","i0005"] | [] |
| fold_cloth_v4__episode_004150 | candidate | 3 | a0008 | 378 | ["right"] | ["i0006","i0007"] | [] |
| fold_and_hang_cloth_v1__episode_001383 | previous | 1 | a0016 | 382 | ["left"] | ["i0013","i0014","i0015"] | [] |
| fold_and_hang_cloth_v1__episode_001383 | previous | 2 | a0018 | 439 | ["left"] | ["i0016","i0017","i0018"] | [] |
| fold_and_hang_cloth_v1__episode_001383 | previous | 3 | a0028 | 1088 | ["right"] | ["i0019","i0027"] | [] |
| fold_and_hang_cloth_v1__episode_001383 | candidate | 1 | a0009 | 439 | ["left"] | ["i0007","i0008","i0009"] | [] |
| fold_and_hang_cloth_v1__episode_001383 | candidate | 2 | a0012 | 826 | ["left"] | ["i0009","i0010","i0011","i0012"] | [] |
| fold_and_hang_cloth_v1__episode_001383 | candidate | 3 | a0015 | 1088 | ["right"] | ["i0012","i0013","i0014","i0015"] | [] |
| fold_and_hang_cloth_v1__episode_003676 | previous | 1 | a0014 | 384 | ["left","right"] | ["i0013"] | ["participant_not_open"] |
| fold_and_hang_cloth_v1__episode_003676 | previous | 2 | a0020 | 505 | ["left","right"] | ["i0016","i0017","i0018","i0019"] | [] |
| fold_and_hang_cloth_v1__episode_003676 | previous | 3 | a0023 | 595 | ["right"] | ["i0022"] | [] |
| fold_and_hang_cloth_v1__episode_003676 | candidate | 1 | a0008 | 384 | ["left"] | ["i0007"] | [] |
| fold_and_hang_cloth_v1__episode_003676 | candidate | 2 | a0012 | 505 | ["right"] | ["i0011"] | [] |
| fold_and_hang_cloth_v1__episode_003676 | candidate | 3 | a0014 | 603 | ["right"] | ["i0013"] | [] |

## 独立输入与时序核对

参考图为0独立核对：6/6；只含open changes：6/6；完整且时序一致的区间观察：6/6；固定节点映射：6/6。

参考图核对同时比较保存request图片数、实际TARGET区间页清单和payload图片清单。时序gate只判断模型陈述是否一致，不能验证其看图解释。

| Episode | 时序发现 | 总体复核原因 | 训练就绪 | 保存标签全-1 |
| --- | --- | --- | --- | --- |
| fold_cloth_v3__episode_000543 | ["temporal_order: stage 1 requires completed first_fold evidence within its selected range"] | ["stage_end_nodes[0]: evidence must belong to this stage before its selected end","stage_end_nodes[0]: terminal interval must confirm completed operation with evidence","stage_end_nodes[1]: evidence must belong to this stage before its selected end","stage_end_nodes[2]: evidence must belong to this stage before its selected end","temporal_order: stage 1 requires completed first_fold evidence within its selected range"] | False | True |
| fold_cloth_v3__episode_006453 | ["temporal_order: i0010 cross_fold lies outside selected stage 3 range"] | ["stage_end_nodes[0]: evidence must belong to this stage before its selected end","stage_end_nodes[1]: evidence must belong to this stage before its selected end","stage_end_nodes[2]: selected nodes must be strictly increasing and unique","stage_end_nodes[2]: evidence must belong to this stage before its selected end","stage_end_nodes[2]: evidence must include the interval ending at the selected node","temporal_order: i0010 cross_fold lies outside selected stage 3 range"] | False | True |
| fold_cloth_v4__episode_002768 | [] | [] | False | False |
| fold_cloth_v4__episode_004150 | [] | [] | False | False |
| fold_and_hang_cloth_v1__episode_001383 | ["temporal_order: stage 1 requires completed first_fold evidence within its selected range","temporal_order: stage 3 requires completed cross_fold evidence within its selected range","temporal_order: i0014 remove lies outside selected stage 4 range"] | ["stage_end_nodes[0]: evidence must belong to this stage before its selected end","stage_end_nodes[0]: terminal interval must confirm completed operation with evidence","stage_end_nodes[1]: evidence must belong to this stage before its selected end","stage_end_nodes[2]: evidence must belong to this stage before its selected end","stage_end_nodes[2]: terminal interval must confirm completed operation with evidence","temporal_order: stage 1 requires completed first_fold evidence within its selected range","temporal_order: stage 3 requires completed cross_fold evidence within its selected range","temporal_order: i0014 remove lies outside selected stage 4 range"] | False | True |
| fold_and_hang_cloth_v1__episode_003676 | [] | [] | False | False |

本轮training_ready=false：6/6；已保存标签全为-1：3/6。未完成请求时不预判最终状态。

## 后续视觉审计（非人工真值）

审计独立保存，不覆盖原模型回答，也不把自动accepted或审计结论折算成accuracy。GPT视觉复核仍需人工确认。

### fold_cloth_v4__episode_002768 · candidate

产物：`gpt_text_criteria_audit`；状态：`needs_review`；审计者：`gpt_visual`。

| 阶段 | 类型 | 发现 | 源帧证据 |
| --- | --- | --- | --- |
| 1 | preparation_fold_ambiguity | i0006含整衣抬起、红布遮住近镜头及重新铺回；这些采样不足以支持模型所说第一侧已折入。 | [151,171,191,209,211] |
| 2 | preparation_fold_ambiguity | i0008含再次整衣抬起及放回；334帧仍有较宽衣身和外伸袖部，不能确认已形成两侧折入的窄长条。 | [212,272,292,334,338] |
| 3 | visible_shape_contradiction | i0010及精确节点426帧仍为较长的条状衣物，与模型在该处声称的紧凑方块有明显矛盾。 | [339,425,426] |

### fold_cloth_v4__episode_004150 · candidate

产物：`gpt_text_criteria_audit`；状态：`needs_review`；审计者：`gpt_visual`。

| 阶段 | 类型 | 发现 | 源帧证据 |
| --- | --- | --- | --- |
| 1 | preparation_fold_ambiguity | i0002可见整件衣物被提起展开后放回，未能确认第一侧折叠完成。 | [126,146,166,179,181] |
| 2 | visible_shape_contradiction | i0004重复抬起近镜头再铺开，节点274帧仍为宽衣身及外伸袖片，缺少模型所述窄长条证据。 | [182,242,262,272,274] |
| 3 | partial_visual_support | 329帧可见窄长条，370/378帧可见进一步翻折成包；第三节点有形态支持，前两节点仍需复核。 | [329,370,378] |

### fold_and_hang_cloth_v1__episode_003676 · candidate

产物：`gpt_text_criteria_audit`；状态：`needs_review`；审计者：`gpt_visual`。

| 阶段 | 类型 | 发现 | 源帧证据 |
| --- | --- | --- | --- |
| 1 | uncertain_visible_transition | i0007内先提起展开整件衣物，随后放回并出现上沿叠层；仅凭这些采样不能确定这是纵折还是整理，保留不确定。 | [262,322,342,362,382,384] |
| 2 | partial_visual_support | i0008内衣物从较宽形态变窄成长条，但精确释放与阶段1是否已完成仍未独立确认。 | [385,425,465,485,495] |


未建立独立逐帧人工真值，不能计算或宣称标注准确率提升。
