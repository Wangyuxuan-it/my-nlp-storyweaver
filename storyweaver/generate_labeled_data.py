

from __future__ import annotations

import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"


INTENT_LABELS = ["investigate", "interrogate", "deduce", "act"]

INTENT_TEMPLATES = {
    "investigate": [
        "检查{place}的{detail}，看看是否有可疑痕迹",
        "调取{place}附近的{detail}并核对时间线",
        "查看{object}上的{detail}是否与前后证词一致",
        "勘察{place}，重点关注{detail}",
        "搜查{place}内的{object}，寻找{detail}",
    ],
    "interrogate": [
        "请详细说明你昨晚{time_range}在哪里，和谁接触过",
        "为什么你前后两次说法不一致，请解释{detail}",
        "你和{person}是什么关系，是否隐瞒了{detail}",
        "请回忆{time_point}前后发生了什么，尤其是{detail}",
        "你是否知道{object}的去向，请直接回答",
    ],
    "deduce": [
        "我怀疑{person}在伪造不在场证明，动机可能是{motive}",
        "根据{evidence}和时间线推断，真正的作案者更像是{person}",
        "结合{detail}来看，这起案件应该不是意外，而是{motive}",
        "从{evidence}判断，嫌疑人显然提前布置了{detail}",
        "我认为{person}在刻意转移视线，背后与{motive}有关",
    ],
    "act": [
        "我现在前往{place}调取监控，并通知协警封锁{place2}",
        "先跟踪{person}，再在{place}布控",
        "立即联系{organization}核对{object}的流向",
        "我去{place}取证，同时记录{detail}的变化",
        "马上进入{place}，把{object}带回去做进一步分析",
    ],
}

PLACE = [
    "案发现场",
    "地下停车场",
    "旧仓库",
    "公寓楼道",
    "商场监控室",
    "医院走廊",
    "河边码头",
    "居民楼顶层",
]

DETAIL = [
    "门锁边缘的刮痕",
    "走廊里的脚印",
    "手机最后一次通话记录",
    "监控画面中的反光衣角",
    "地面残留的泥点",
    "桌面上未喝完的咖啡杯",
    "窗台上的纤维碎屑",
    "嫌疑人衣袖上的灰尘",
]

OBJECT = [
    "手机",
    "钥匙串",
    "车牌记录",
    "通话清单",
    "行车记录仪",
    "纸质笔录",
    "药盒",
    "手提袋",
]

PERSON = [
    "报案人",
    "失踪者的同事",
    "楼下保安",
    "便利店店员",
    "嫌疑人",
    "目击者",
    "受害者家属",
    "出租车司机",
]

MOTIVE = [
    "财务纠纷",
    "家庭矛盾",
    "债务压力",
    "保险赔偿",
    "职场竞争",
    "旧日恩怨",
    "隐瞒关系",
    "非法交易",
]

EVIDENCE = [
    "门锁无撬痕但门把有摩擦",
    "监控在关键时间段突然中断",
    "手机恢复出陌生号码的通话",
    "鞋印大小与嫌疑人一致",
    "转账记录与口供时间冲突",
    "走廊地面留下新鲜泥渍",
    "窗台纤维与雨衣材质相同",
    "车库刹车痕方向异常",
]

TIME_RANGE = [
    "23:00到01:00",
    "昨晚20:00到23:30",
    "凌晨0点到2点",
    "傍晚到深夜之间",
    "案发前后半小时",
]

TIME_POINT = [
    "案发前",
    "报案后",
    "进入大楼时",
    "离开现场前",
    "接到电话之后",
]

ORGANIZATION = [
    "物业公司",
    "通信运营商",
    "交警支队",
    "医院保安部",
    "商场管理处",
    "车管所",
]


def cycle_item(items: list[str], index: int) -> str:
    return items[index % len(items)]


def build_intent_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    per_label = 125
    for label_index, label in enumerate(INTENT_LABELS):
        templates = INTENT_TEMPLATES[label]
        for i in range(per_label):
            global_index = label_index * per_label + i + 1
            template = templates[i % len(templates)]
            template_id = f"{label}_t{i % len(templates)}"
            row = {
                "sample_id": f"INT{global_index:03d}",
                "template_id": template_id,
                "user_input": template.format(
                    place=cycle_item(PLACE, i + label_index),
                    place2=cycle_item(PLACE, i + label_index + 1),
                    detail=cycle_item(DETAIL, i + 2 * label_index),
                    object=cycle_item(OBJECT, i + 3 * label_index),
                    person=cycle_item(PERSON, i + 4 * label_index),
                    time_range=cycle_item(TIME_RANGE, i + label_index),
                    time_point=cycle_item(TIME_POINT, i + label_index),
                    motive=cycle_item(MOTIVE, i + label_index),
                    evidence=cycle_item(EVIDENCE, i + label_index),
                    organization=cycle_item(ORGANIZATION, i + label_index),
                ),
                "intent_label": label,
                "source": "synthetic",
                "split": "",
            }
            rows.append(row)
    return rows


def build_coherence_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for i in range(150):
        place = cycle_item(PLACE, i)
        detail = cycle_item(DETAIL, i + 1)
        person = cycle_item(PERSON, i + 2)
        evidence = cycle_item(EVIDENCE, i + 3)
        positive_context = f"第1回合：{place}有初步异常；第2回合：{detail}与前文线索相互印证"
        positive_reply = f"你继续围绕{place}展开调查，并发现{evidence}，这一推进与先前信息保持一致。"
        negative_context = f"第1回合：{place}发生异常；第2回合：{person}的说法仍有疑点"
        negative_reply = f"剧情突然跳到与案件无关的校园活动，完全没有回应{place}和{person}的线索。"

        rows.append(
            {
                "pair_id": f"COH{2 * i + 1:03d}",
                "template_id": f"coh_pos_t{i % len(PLACE)}",
                "history_context": positive_context,
                "candidate_reply": positive_reply,
                "coherence_label": "1",
                "source": "synthetic",
                "split": "",
            }
        )
        rows.append(
            {
                "pair_id": f"COH{2 * i + 2:03d}",
                "template_id": f"coh_neg_t{i % len(PLACE)}",
                "history_context": negative_context,
                "candidate_reply": negative_reply,
                "coherence_label": "0",
                "source": "synthetic",
                "split": "",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    intent_rows = build_intent_rows()
    coherence_rows = build_coherence_rows()

    write_csv(
        RAW_DIR / "intent_annotations.csv",
        intent_rows,
        ["sample_id", "template_id", "user_input", "intent_label", "source", "split"],
    )
    write_csv(
        RAW_DIR / "coherence_annotations.csv",
        coherence_rows,
        ["pair_id", "template_id", "history_context", "candidate_reply", "coherence_label", "source", "split"],
    )

    print(f"Generated intent rows: {len(intent_rows)}")
    print(f"Generated coherence rows: {len(coherence_rows)}")
    print(f"Intent output: {RAW_DIR / 'intent_annotations.csv'}")
    print(f"Coherence output: {RAW_DIR / 'coherence_annotations.csv'}")


if __name__ == "__main__":
    main()