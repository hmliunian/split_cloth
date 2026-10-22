# 夹爪事件候选与连续帧实测：event_probe_max_paired6_20261022

这是相同样本的工程可行性实测。自动接受数、与夹爪事件的一致性及边界变化均不等于标注准确率。已有GPT记录来自1FPS网格，只作粗语义参考，不是逐帧人工真值，不用于证明首次完全松爪帧的准确率。候选事件来自实测state，仍依赖开合阈值、时间同步、动作语义及视觉可见性。六条既有样本不是独立随机测试集；结果不能外推至全部数据或另一代机器人。

对照：`qwen3_8_max_paired6_20261022` → `event_probe_max_paired6_20261022`；按key匹配 6/6 条。

## 同一批样本的处理与用量

| 指标 | 1FPS基线 | 事件候选＋连续帧 |
| --- | --- | --- |
| 模型身份 | ["qwen3.8-max-0902"] | ["qwen3.8-max-0902"] |
| 轨迹数 | 6 | 6 |
| 自动接受（不是准确率） | 4 | 3 |
| 状态计数 | {"accepted":4,"needs_review":1,"out_of_scope":1} | {"needs_review":2,"accepted":3,"out_of_scope":1} |
| 执行错误 | 0 | 1 |
| 已保存的独立响应 | 6 | 20 |
| 已记录尝试 | 6 | 21 |
| 已知 prompt_tokens | 52508 | 418305 |
| 已知 completion_tokens | 2877 | 7711 |
| 已知 total_tokens | 55385 | 426016 |
| 有已知total_tokens的轨迹 | 6 | 6 |

用量按实际response及重试归档去重累计；annotation聚合用量仅在无响应产物时回退使用。失败请求无用量时成本未知。新流程独立重选阶段并进行精修，其总用量含所有已保存selection/refinement响应。

相同配对样本的已知total_tokens倍率：7.691902。计费金额还受缓存和输入/输出单价影响，token倍率不是价格倍率。

## 本轮工程发现

新增自动接受：fold_cloth_v4__episode_004150；从接受转为非接受：fold_cloth_v3__episode_000543, fold_cloth_v3__episode_006453。这只说明状态变化，不能认定前者修正或后者退化。

双方accepted交集中新−旧边界时间差范围：[-12.65, -0.8] 秒。大幅提前应检查动作语义，不能直接解释为修正了1FPS量化。

| 执行错误Episode | 类型 | 原因 |
| --- | --- | --- |
| fold_cloth_v3__episode_000543 | RuntimeError | ["provider_http_400: data_inspection_failed <400> InternalError.Algo.DataInspectionFailed: Input image data may contain inappropriate content."] |

接口图片审核等执行错误与模型语义拒判分开统计；原始失败记录保留，没有把未完成请求计作正确拒判。

## 全部样本逐条对照

| Episode | 基线状态/边界 | 新状态/边界 | 新−旧（帧） | 候选数 | 新已知total_tokens |
| --- | --- | --- | --- | --- | --- |
| fold_cloth_v3__episode_000543 | accepted / [420, 520, 580] | needs_review / [] | None | 13 | 67716 |
| fold_cloth_v3__episode_006453 | accepted / [280, 440, 520] | needs_review / [] | None | 18 | 81653 |
| fold_cloth_v4__episode_002768 | accepted / [360, 420, 520] | accepted / [136, 211, 427] | [-224,-209,-93] | 15 | 80630 |
| fold_cloth_v4__episode_004150 | needs_review / [] | accepted / [180, 274, 329] | None | 11 | 76224 |
| fold_and_hang_cloth_v1__episode_001383 | out_of_scope / [] | out_of_scope / [] | None | 17 | 39954 |
| fold_and_hang_cloth_v1__episode_003676 | accepted / [420, 520, 620] | accepted / [167, 384, 604] | [-253,-136,-16] | 17 | 79839 |

## 事件时间对齐与校准

| Episode | 候选数 | alignment_verified | 诊断 |
| --- | --- | --- | --- |
| fold_cloth_v3__episode_000543 | 13 | True | [{"code":"multithreshold_feedback_is_not_physical_truth","message":"All raw opening levels are candidate evidence; neither 90% nor any lower threshold proves complete release or stage completion."}] |
| fold_cloth_v3__episode_006453 | 18 | True | [{"code":"multithreshold_feedback_is_not_physical_truth","message":"All raw opening levels are candidate evidence; neither 90% nor any lower threshold proves complete release or stage completion."}] |
| fold_cloth_v4__episode_002768 | 15 | True | [{"code":"multithreshold_feedback_is_not_physical_truth","message":"All raw opening levels are candidate evidence; neither 90% nor any lower threshold proves complete release or stage completion."}] |
| fold_cloth_v4__episode_004150 | 11 | True | [{"code":"multithreshold_feedback_is_not_physical_truth","message":"All raw opening levels are candidate evidence; neither 90% nor any lower threshold proves complete release or stage completion."}] |
| fold_and_hang_cloth_v1__episode_001383 | 17 | True | [{"code":"multithreshold_feedback_is_not_physical_truth","message":"All raw opening levels are candidate evidence; neither 90% nor any lower threshold proves complete release or stage completion."}] |
| fold_and_hang_cloth_v1__episode_003676 | 17 | True | [{"code":"multithreshold_feedback_is_not_physical_truth","message":"All raw opening levels are candidate evidence; neither 90% nor any lower threshold proves complete release or stage completion."}] |

阈值与校准完整记录保存在JSON。alignment_verified为false时，夹爪候选不能作为可靠的原始视频帧锚点。

双方accepted且元数据一致的边界变化覆盖 2 条、6 个边界，其中 6 个改变。变化幅度不代表误差减少。

## 连续帧与候选证据

留存局部精修记录 14 个；模型自报视觉确认 13 个。这些计数不构成人工验证。

| Episode | 阶段 | 旧帧 | 新提议帧（含待复核） | 选中event | 模型自报视觉确认 | 距旧帧最近event（帧差） | 证据/复核原因 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fold_cloth_v3__episode_000543 | 1 | 420 | 275 | ["gripper_left_000273","gripper_right_000272"] | True | gripper_left_000398 (-21) | Stage 1 involves unfolding and the first long-side fold. In frame 274, the grippers are in the process of opening (left raw 0.0709, right raw 0.0647) but the cloth is still being settled. By frame 275, both grippers have reached their maximum open state (left raw 0.0704, right raw 0.0647) and are visually fully retracted away from the cloth, which lies flat in a folded rectangular shape. The state values remain stable in subsequent frames (276-289), confirming the release is complete and not a transient adjustment. Frame 275 is the first frame where the action is fully completed and grippers are confirmed open. |
| fold_cloth_v3__episode_000543 | 2 | 520 | 401 | ["gripper_left_000398","gripper_right_000398"] | True | gripper_right_000484 (-35) | Stage 2 involves folding the other side to form a narrow strip. Visual evidence from base camera shows the cloth formed into a long strip by frame 398-399. The grippers begin opening around frame 398 (state values rise sharply). By frame 401 (t=20.05s), both left and right gripper state values have reached their maximum open plateau (left ~0.070, right ~0.063) and remain stable in subsequent frames (402-414). The wrist cameras at frame 401 show the gripper fingers fully open and retracted slightly from the cloth surface, confirming physical release. Frame 400 shows the transition still completing or just reaching threshold, while 401 is the first frame of stable full release. |
| fold_cloth_v3__episode_006453 | 1 | 280 | 201 | ["gripper_left_000191","gripper_right_000191"] | True | gripper_right_000234 (-45) | Stage 1 involves the first long-side fold. Visual evidence shows the cloth being manipulated and folded between f=177 and f=195. The grippers begin to open around f=191 (state values rise), but the arms remain near the cloth until f=200. At f=201 (t=10.05s), the state values for both arms drop to near-zero (left: 0.00099, right: 0.0029), indicating full physical release and retraction. The cloth lies flat in the folded configuration at f=201 and remains stable in subsequent frames (f=202+). Thus, f=201 is the first frame where the operation is complete and grippers are fully released. |
| fold_cloth_v3__episode_006453 | 2 | 440 | 340 | ["gripper_left_000330","gripper_right_000330"] | False | gripper_right_000471 (+32) | Stage 2 involves folding the garment into a narrow strip and releasing. Visuals show the folding action concluding around f=330-335. The left gripper opens early (f=330-332) but remains stationary near the cloth. The right gripper state decreases steadily from f=332 (0.027) to f=340 (0.0001), indicating a controlled release or retraction. At f=340, the right gripper state reaches its minimum (fully closed/released baseline), and visually, both arms are retracted and open, leaving the folded strip on the table. Frames f=341+ confirm the stable released state. |
| fold_cloth_v3__episode_006453 | 3 | 520 | 489 | ["gripper_right_000487"] | True | gripper_right_000539 (+20) | Stage 3 involves folding the strip into a rectangle. Visuals at f=480-486 show the folded packet on the table with the right arm hovering/adjusting. The right arm state data shows a clear release event starting at f=487 (raw value jumps from 0.016 to 0.050) and reaching full open confirmation at f=489 (raw value 0.067, stable). Visuals at f=489 confirm the right gripper is fully open and retracted slightly, leaving the folded garment on the table. The subsequent frames (f=490+) show the arm remaining open and stationary, confirming the completion of the release. |
| fold_cloth_v4__episode_002768 | 1 | 360 | 136 | ["gripper_left_000134","gripper_right_000135"] | True | gripper_right_000366 (+6) | 在f=134-135期间，左右夹爪状态值迅速上升超过开启阈值（左臂0.071，右臂0.032），表明夹爪正在张开。至f=136 (t=6.8s)，左臂状态稳定在0.0706（接近最大值0.0712），右臂状态为0.0364，且后续帧（137-148）保持稳定或缓慢变化，确认物理释放完成。视觉证据显示，在base相机f=136及之后，衣物已平铺于台面，机械臂处于松开后的回退或悬停状态，不再抓取衣物，符合阶段1（展开并折叠第一侧后松开）的结束特征。 |
| fold_cloth_v4__episode_002768 | 2 | 420 | 211 | ["gripper_left_000210","gripper_right_000209"] | True | gripper_left_000424 (+5) | 在f=210时，左夹爪状态值(0.064)刚超过开启阈值，右夹爪(0.037)虽超过其较低阈值但视觉上仍贴近衣物。至f=211，左夹爪状态值达到峰值(0.071)，右夹爪保持稳定，且base相机显示双臂均已明显回撤离开衣物，衣物呈折叠后的长条状静置，确认操作完成且夹爪完全松开。f=212及之后状态稳定，符合释放后特征。 |
| fold_cloth_v4__episode_002768 | 3 | 520 | 427 | ["gripper_left_000424","gripper_right_000422"] | True | gripper_right_000486 (-33) | 在f=424-426期间，左右夹爪状态值迅速上升超过开启阈值（左>0.063, 右>0.055），表明夹爪正在张开。至f=427 (t=21.35s)，左臂状态值稳定在0.0706，右臂稳定在0.0690，且 wrist 相机画面显示夹爪指尖已完全脱离衣物表面，衣物呈折叠后的矩形静置于台面。f=428及后续帧状态保持稳定，确认释放完成且无重抓动作。 |
| fold_cloth_v4__episode_004150 | 1 | None | 180 | ["gripper_left_000178","gripper_right_000177"] | True | None | Stage 1 involves unfolding and the first long-side fold. Visual evidence from base camera shows the garment being manipulated and folded between frames 164-178. At frame 179, the grippers are in the process of releasing (state values rising: left 0.069, right 0.055). By frame 180 (t=9.0s), both arms have fully released the garment (left state stable at max ~0.069, right state ~0.058 and rising to ~0.060 in subsequent frames), and the garment lies flat on the table, completing the fold operation. The arms remain retracted/open in frames 181-185 before any potential next movement. Frame 179 shows the transition, while frame 180 confirms the stable released state. |
| fold_cloth_v4__episode_004150 | 2 | None | 274 | ["gripper_left_000272","gripper_right_000272"] | True | None | Stage 2 involves folding the garment into a narrow strip. Visual evidence from the base camera shows the arms holding and aligning the fabric through frame 273. At frame 274 (t=13.7s), both grippers have fully opened and retracted slightly, leaving the garment resting flat on the table in a long rectangular shape. The state feedback confirms the opening transition: left arm raw value rises to ~0.050 and right arm to ~0.053 by frame 274, stabilizing in subsequent frames (275-288), indicating a completed release rather than a transient adjustment. |
| fold_cloth_v4__episode_004150 | 3 | None | 329 | ["gripper_left_000328","gripper_right_000328"] | True | None | Stage 3 involves folding the strip into a packet. At f=327, both arms are still holding/pressing the fabric (low raw values ~0.03). At f=328, the grippers begin to open rapidly (left ~0.055, right ~0.062). By f=329, both grippers have reached their maximum opening values (left ~0.058, right ~0.068) and the visual evidence from the wrist cameras confirms the fingers are fully retracted/open, leaving the folded packet on the table. The state remains stable in f=330-339 before the right arm re-grasps at f=340. Thus, f=329 is the first frame where the operation is complete and grippers are fully released. |
| fold_and_hang_cloth_v1__episode_003676 | 1 | 420 | 167 | ["gripper_left_000156","gripper_right_000164"] | True | gripper_right_000382 (-37) | Stage 1 involves the first long-edge fold. Visuals show the left arm releasing around f=158-160 (state confirms left gripper open at 0.9 fraction). The right arm continues to hold/adjust the fabric until f=166. At f=167, the right gripper state reaches ~0.0495 (approx 0.9 open fraction, threshold 0.0497), and the right wrist camera shows the gripper fingers fully open and retracted from the fabric. The fabric lies flat with the fold completed. Frames 168-179 show the arms retracted and the fabric stable, confirming the release is complete and not a regrasp. |
| fold_and_hang_cloth_v1__episode_003676 | 2 | 520 | 384 | ["gripper_left_000380","gripper_right_000382"] | True | gripper_left_000502 (-17) | Stage 2 involves folding the other side to form a narrow strip. Visuals show the strip shape is formed by frame 380-382. The left gripper release event starts at f=380 and confirms at f=382 (raw > 0.054). The right gripper release starts at f=382 and confirms at f=384 (raw > 0.049). At frame 384, both grippers have raw values well above their respective open thresholds (Left: 0.059, Right: 0.056) and are visually retracted/open in the wrist cameras. Frame 383 shows the right gripper still in the process of opening (raw 0.054, just crossing threshold), while frame 384 shows stable open state for both. Thus, frame 384 is the first frame where the operation is fully complete and both grippers are confirmed released. |
| fold_and_hang_cloth_v1__episode_003676 | 3 | 620 | 604 | ["gripper_right_000602"] | True | gripper_right_000602 (-18) | Stage 3 involves folding the strip into a packet. Visuals from f=587 to f=601 show the right arm manipulating and folding the blue cloth. At f=602, the gripper state crosses the opening threshold (raw 0.052 > 0.0497), and the base camera shows the gripper fingers beginning to separate from the cloth. By f=604, the raw state stabilizes at ~0.055 (fully open), and the base camera confirms the gripper is fully open and retracted slightly above the folded packet, which remains stationary on the table. The wrist camera also shows the gripper tips clear of the fabric. This marks the completion of the fold and release. |

最近候选仅是诊断信息，没有用它回填结果；最近松爪可能属于调整、重抓或下一阶段。先判定阶段语义，再查看参与夹爪的连续帧。

## 已有GPT粗参考

下表仅保留既有1FPS视觉记录，不将新原始帧与粗采样帧的距离称为逐帧误差或准确率。

| Episode | 参考状态 | GPT粗候选帧 |
| --- | --- | --- |
| fold_cloth_v3__episode_000543 | reference_only | [420,500,580] |
| fold_cloth_v3__episode_006453 | reference_only | [360,420,500] |
| fold_cloth_v4__episode_002768 | reference_only | [360,440,500] |
| fold_cloth_v4__episode_004150 | needs_review | [300,340,null] |
| fold_and_hang_cloth_v1__episode_001383 | out_of_scope | [null,null,null] |
| fold_and_hang_cloth_v1__episode_003676 | reference_only | [400,500,620] |

## 本轮GPT产物复核（非人工金标）

下列复核独立于原模型自报字段保存，保留原始模型输出。本次复核已知模型结果，不是盲审。GPT复核仍不是人工逐帧真值，不计算准确率。完整审计对象见本报告JSON。

### fold_cloth_v3__episode_000543

复核者：gpt_visual；范围：independent_main_view_montage_and_state_readback；状态：execution_incomplete。

| 阶段 | 问题类型 | 复核发现 | 源帧证据 |
| --- | --- | --- | --- |
| 3 | provider_image_inspection_failure | 阶段3精修被供应商图片审核以HTTP400拒绝；本条属于执行不完整，不能计作语义拒判或正确标注。 | [] |

### fold_cloth_v3__episode_006453

复核者：gpt_visual；范围：independent_main_view_montage_and_state_readback；状态：reject_for_training。

| 阶段 | 问题类型 | 复核发现 | 源帧证据 |
| --- | --- | --- | --- |
| 1 | state_direction_contradiction | 精修以f201左右反馈降到近零解释为完全松开，与输入校准中higher表示张开的假设相反。原始读回left=0.00099446/right=0.00291889；不能由该解释确认释放。 | [200,201,202] |
| 2 | state_direction_contradiction | 精修将右反馈在f340降到0.00014833解释为释放，文字同时称fully closed/released baseline，解释自相矛盾。 | [339,340,341] |
| 4 | completion_id_text_conflict | 结构completion_sample_id=s000027映射f540，文字却反复说明应选s000029/f572；程序选择结构值，不能假定文字纠正已生效。 | [540,572] |

### fold_cloth_v4__episode_002768

复核者：gpt_visual；范围：independent_main_view_montage_and_state_readback；状态：reject_for_training。

| 阶段 | 问题类型 | 复核发现 | 源帧证据 |
| --- | --- | --- | --- |
| 1 | preparation_mistaken_for_folding | 新结束帧136附近为取衣/抖开/摊衣，f140/160仍为展开上衣，f180/200再次抖开；不能视为第一次纵折完成。 | [120,140,160,180,200] |
| 2 | stage_sequence_shift | 新结束帧211附近衣服刚铺下，f260/280再次抖开，f320/340才出现第一侧纵折。新阶段2过早，局部松爪细节不能弥补全局阶段语义错位。 | [200,220,260,280,320,340] |

### fold_cloth_v4__episode_004150

复核者：gpt_visual；范围：independent_main_view_montage_and_state_readback；状态：reject_for_training。

| 阶段 | 问题类型 | 复核发现 | 源帧证据 |
| --- | --- | --- | --- |
| 3 | strip_mistaken_for_final_rectangle | 新第三结束帧329位于f320/f340附近，此时画面仍为长条；不支持横折小矩形已经完成。原基线待复核，本轮转accepted暴露误接受。 | [300,320,340,360,380] |

### fold_and_hang_cloth_v1__episode_001383

复核者：gpt_visual；范围：new_and_baseline_task_type_comparison；状态：scope_agreement_only。

| 阶段 | 问题类型 | 复核发现 | 源帧证据 |
| --- | --- | --- | --- |
| None | scope_rejection_agreement | 新旧模型均将该条判为hang_cloth/out_of_scope；这只是任务适用性一致，不能计作原始帧边界准确率。 | [] |

### fold_and_hang_cloth_v1__episode_003676

复核者：gpt_visual；范围：independent_main_view_montage_and_state_readback；状态：reject_for_training。

| 阶段 | 问题类型 | 复核发现 | 源帧证据 |
| --- | --- | --- | --- |
| 1 | preparation_mistaken_for_folding | 新第一结束帧167附近衣物仍在抖开与铺平；f180/200/220是展开衣物，f240和f320/340又抖开，不能在167结束第一侧纵折。 | [140,160,180,200,220,240,320,340] |
| 2 | stage_sequence_shift | 新第二结束帧384附近更接近第一侧纵折后的状态；精修不能代替正确的阶段语义判断。 | [360,380,400,420] |


本轮复核发现原模型自报确认无法排除的错误，当前实现不适合直接批量生成训练标注。自动accepted不代表成功，需先修正阶段语义选择与证据一致性。

未建立独立逐帧人工真值，无法计算精确边界准确率或确认准确率提升。
