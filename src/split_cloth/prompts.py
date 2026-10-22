from __future__ import annotations

import base64
import json
from pathlib import Path

from .util import read_json, read_jsonl

PROMPT_VERSION = "expert_montage_v1.1"

SYSTEM = """你是机器人衣物操作视频的阶段粗标注员。依据标准参考图、已标注EXPERT示范与TARGET完整时序网格判断。
每页从左到右、从上到下；相邻页有2个重复采样，同sample_id为同一时刻。EMPTY不是画面。
EXPERT和TARGET是不同视频，sample_id各自在自己的视频中有效；输出只能引用TARGET的sample_id。
不能按时长比例套用expert边界，不按页码或图格等分阶段。遇到遮挡、缺阶段、返工或不适用任务如实拒判。
当前阶段操作完成且参与该动作的夹爪完全松开才满足完成判据。衣物折好但仍抓持时仍属于当前阶段。
完成帧仍属于当前阶段，下一原始帧才归下一阶段。输出的是1FPS可见采样证据，不假装恢复未提供的精确帧。
只输出单个JSON，不输出代码块、解释或思考过程。evidence_text用简短可见事实说明。"""

SCHEMA_TEXT = """输出字段严格为：
task_type: fold_cloth|fold_pants|hang_cloth|mixed|other|unknown;
applicable: true/false/null; all_stages_observed: boolean; needs_review: boolean; review_reasons: string[];
stage4_completion: {observed:boolean, completion_sample_id:string|null, evidence_text:string};
boundary_proposals: 恰好三项，依次1→2、2→3、3→4，每项
{from_stage:int,to_stage:int,last_not_complete_sample_id:string|null,first_complete_sample_id:string|null,evidence_text:string}。
每个边界L=最后可确认本阶段尚未完成的采样，R=紧接着首次可确认完成且完全松爪的采样，L/R必须为TARGET相邻唯一采样。
无法给出相邻确定证据时相应ID用null，needs_review=true。第四阶段完成证据必须独立给出，不能只因视频结束推断完成。
时序硬约束：三个R依次严格递增，stage4_completion的sample_seq必须严格大于第三个R，不能用同一个sample同时声称第三阶段结束和第四阶段完成。
请检查最后一页的所有真实采样。若1FPS无法分辨这两次事件，填null并needs_review=true，不能虚构完成位置。
不适用、缺阶段、看不清、明显返工或终态不可见时，needs_review=true并说明。
没有疑问且四阶段与终态均可见时needs_review=false,review_reasons=[]。禁止额外字段、confidence或自由生成秒数。"""


def image_item(path):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data, "detail": "high"}}


def build_messages(run, output):
    run, output = Path(run), Path(output)
    expert = read_json(run / "expert.json")
    catalog = read_json(run / "standard_catalog.json")
    if not expert or not expert.get("validated") or not catalog:
        raise ValueError("A validated expert and stage catalog are required before annotation")
    content = [{"type": "text", "text": "标准阶段定义：\n" + json.dumps(catalog["stages"], ensure_ascii=False)}]
    for stage in catalog["stages"]:
        content.append({"type": "text", "text": f"REFERENCE_STANDARD stage {stage['stage_id']}；不是TARGET时序"})
        content.append(image_item(run / stage["reference_image"]))
    expert_dir = run / expert["directory"]
    content.append({"type": "text", "text": "EXPERT 示范，标签来自GPT视觉审核，非人工金标准。阶段结束原始帧：" + json.dumps(expert["boundary_frames"]) + "。各格有STAGE标签；只参考视觉语义与动作顺序，不能复制时间比例。"})
    for p in read_json(expert_dir / "pages.json", []):
        content.append({"type": "text", "text": "EXPERT " + p["page_id"]})
        content.append(image_item(expert_dir / p["image_path"]))
    manifest = read_json(output / "input_manifest.json")
    samples = read_jsonl(output / "samples.jsonl")
    content.append({"type": "text", "text": "TARGET 待标视频，主视角1FPS；原始帧数=" + str(manifest["frame_count"]) + "\n唯一采样表：\n" + "\n".join(f"{s['sample_id']} seq={s['sample_seq']} t={s['source_timestamp_s']:.3f} f={s['source_frame_id']}" for s in samples)})
    for p in read_json(output / "pages.json", []):
        content.append({"type": "text", "text": "TARGET " + p["page_id"]})
        content.append(image_item(output / p["image_path"]))
    content.append({"type": "text", "text": SCHEMA_TEXT})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
