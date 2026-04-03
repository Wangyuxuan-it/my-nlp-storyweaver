"""
StoryWeaver 主程序（Gradio 前端）

运行方式：
python app.py

功能：
1. 提供聊天式文字冒险交互界面。
2. 管理游戏对话历史与重开逻辑。
3. 调用 story_generator 生成剧情推进。
"""

from __future__ import annotations

import os
import re
import html
import time
import uuid
import socket
from typing import Dict, Generator, List, Tuple
import gradio as gr


from case_state import init_case_state, summarize_case_state, update_case_state
from experiment_logger import ExperimentLogger
from dialogue_manager import DialogueManager
from human_evaluation import append_human_evaluation
from ml_models import CoherenceClassifier, IntentClassifier
from intent_recognizer import INTENT_LABELS, detect_intent
from metrics import evaluate_turn_metrics
from story_generator import StoryGenerator
from config import ENABLE_CASE_MEMORY, ENABLE_INTENT_PROMPT, EXPERIMENT_TAG


generator = StoryGenerator()
experiment_logger = ExperimentLogger()
intent_model = IntentClassifier.load_or_none()
coherence_model = CoherenceClassifier.load_or_none()
dialogue_manager = DialogueManager()
ChatMessage = Dict[str, object]
MAX_NARRATIVE_TURNS = 20


def _pick_launch_port(preferred_port: int = 7860, max_tries: int = 20) -> int:
    env_port = os.environ.get("GRADIO_SERVER_PORT")
    if env_port:
        try:
            preferred_port = int(env_port)
        except ValueError:
            pass

    for port in range(preferred_port, preferred_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
            except OSError:
                continue
            return port

    return preferred_port


def _remove_blank_lines(text: str) -> str:
    """移除文本中的空白行，避免聊天区出现多余空行。"""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


SECTION_ORDER = ["场景", "可疑线索", "可执行行动"]
SECTION_ALIASES: Dict[str, List[str]] = {
    "场景": ["场景", "当前场景", "现场", "情景", "行动结果", "结果反馈"],
    "可疑线索": ["可疑线索", "线索", "关键线索", "证据", "新线索", "技术鉴定失败"],
    "可执行行动": ["可执行行动", "可执行运动", "行动建议", "行动选项", "建议行动", "下一步行动"],
}


def _strip_prefix_index(line: str) -> str:
    """移除模型可能输出的前缀编号，避免重复编号。"""
    return re.sub(r"^[a-zA-Z0-9①-⑩\-\.\)\））、:：\s]+", "", line).strip()


def _detect_section(line: str) -> Tuple[str | None, str]:
    """识别当前行是否为小节标题，并返回规范小节名与同行尾部内容。"""
    stripped = line.strip()
    for canonical, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            matched = re.match(
                rf"^([>#\-\s]*)\*{{0,2}}[\[【(（]?\s*{re.escape(alias)}\s*[\]】)）]?\*{{0,2}}([：:]?)(.*)$",
                stripped,
            )
            if matched:
                tail = (matched.group(3) or "").strip()
                return canonical, tail
    return None, stripped


def _normalize_message_blocks(text: str) -> str:
    """将回复归一为三段结构：【场景】【可疑线索】【可执行行动】并统一大写编号。"""
    buckets: Dict[str, List[str]] = {section: [] for section in SECTION_ORDER}
    current_section = "场景"

    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue

        section, tail = _detect_section(raw_line)
        if section:
            current_section = section
            if tail:
                buckets[current_section].append(tail)
            continue

        buckets[current_section].append(raw_line.strip())

    if not buckets["场景"]:
        buckets["场景"] = ["现场暂无新增描述。"]
    if not buckets["可疑线索"]:
        buckets["可疑线索"] = ["暂未发现新增可疑线索。"]
    if not buckets["可执行行动"]:
        buckets["可执行行动"] = ["继续调查现场细节", "追问关键相关人", "核验已得线索"]

    clues = [_strip_prefix_index(item) for item in buckets["可疑线索"] if _strip_prefix_index(item)]
    if len(clues) > 1:
        buckets["可疑线索"] = [f"{chr(ord('A') + i)}. {item}" for i, item in enumerate(clues)]
    elif len(clues) == 1:
        buckets["可疑线索"] = clues
    else:
        buckets["可疑线索"] = ["暂未发现新增可疑线索。"]

    actions = [_strip_prefix_index(item) for item in buckets["可执行行动"] if _strip_prefix_index(item)]
    if actions:
        buckets["可执行行动"] = [f"{chr(ord('A') + i)}. {item}" for i, item in enumerate(actions)]
    else:
        buckets["可执行行动"] = ["A. 继续调查现场细节", "B. 追问关键相关人", "C. 核验已得线索"]

    output_lines: List[str] = []
    for section in SECTION_ORDER:
        output_lines.append(f"**[ {section} ]**")
        output_lines.append("")
        output_lines.extend(buckets[section])
        output_lines.append("")

    return "\n".join(output_lines).strip()


def render_chat_html(messages: List[ChatMessage]) -> str:
    """将消息状态渲染为纯 HTML，避免 Chatbot 默认样式导致的额外空白。"""
    shell_style = (
        "min-height:320px;max-height:calc(100vh - 320px);border:1px solid #e5e7eb;border-top:none;"
        "border-radius:0 0 10px 10px;background:#fcfdff;overflow:hidden;"
    )
    scroll_style = (
        "min-height:320px;max-height:calc(100vh - 320px);overflow-y:auto;overflow-x:hidden;padding:10px 12px;"
        "box-sizing:border-box;"
    )
    list_style = "margin:0;padding:0;display:flex;flex-direction:column;gap:8px;"
    auto_scroll_script = (
        "<img src='x' alt='' style='display:none' onerror=\"(function(){"
        "const el=document.getElementById('story-scroll');"
        "if(!el){return;}"
        "const jump=function(){el.scrollTop=el.scrollHeight;};"
        "jump();"
        "requestAnimationFrame(jump);"
        "setTimeout(jump,60);"
        "setTimeout(jump,180);"
        "})();this.onerror=null;this.remove();\">"
    )
    typing_style = (
        "<style>"
        "@keyframes storyTypingBlink{0%,80%,100%{opacity:.25;}40%{opacity:1;}}"
        ".story-typing{display:inline-flex;align-items:center;gap:4px;height:16px;}"
        ".story-typing-dot{width:6px;height:6px;border-radius:50%;background:#64748b;display:inline-block;animation:storyTypingBlink 1.2s infinite;}"
        ".story-typing-dot:nth-child(2){animation-delay:.2s;}"
        ".story-typing-dot:nth-child(3){animation-delay:.4s;}"
        "</style>"
    )

    if not messages:
        return (
            f'<div id="story-shell" style="{shell_style}"><div id="story-scroll" style="{scroll_style}"><div id="story-list" style="{list_style}">'
            '<div class="story-empty" style="color:#6b7280;font-size:14px;line-height:1.5;padding:8px 10px;">剧情已加载，等待案件开场...</div>'
            f'</div></div></div>{auto_scroll_script}'
        )

    parts: List[str] = []
    for msg in messages:
        role = msg.get("role", "assistant")
        row_justify = "flex-end" if role == "user" else "flex-start"
        bubble_bg = "#f1f5f9" if role == "user" else "#ffffff"
        bubble_border = "#cbd5e1" if role == "user" else "#dbeafe"
        is_pending = bool(msg.get("pending", False))
        content = msg.get("content", "") or ""
        if is_pending:
            safe = (
                '<span class="story-typing">'
                '<span class="story-typing-dot"></span>'
                '<span class="story-typing-dot"></span>'
                '<span class="story-typing-dot"></span>'
                "</span>"
            )
        else:
            safe = html.escape(str(content))
            # 仅启用最小 Markdown：**加粗**，其余仍按纯文本处理。
            safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
            safe = safe.replace("\n", "<br>")
        parts.append(
            f'<div class="story-row" style="display:flex;justify-content:{row_justify};margin:0;">'
            f'<div class="story-bubble" style="max-width:90%;margin:0;padding:10px 12px;border-radius:12px;border:1px solid {bubble_border};background:{bubble_bg};box-sizing:border-box;">'
            f'<div class="story-content" style="margin:0;padding:0;word-break:break-word;line-height:1.5;font-size:14px;color:#0f172a;">{safe}</div>'
            "</div>"
            "</div>"
        )

    return (
        typing_style
        + f'<div id="story-shell" style="{shell_style}"><div id="story-scroll" style="{scroll_style}"><div id="story-list" style="{list_style}">'
        + "".join(parts)
        + f"</div></div></div>{auto_scroll_script}"
    )


def start_new_game(mode: str) -> Tuple[List[ChatMessage], List[Tuple[str, str]], str, int, bool, Dict[str, object]]:
    """
    初始化新游戏。

    返回:
    - chat_display: Gradio Chatbot 显示内容
    - game_history: 供模型调用的历史轮次
    - status_text: 状态提示
    - turn_count: 当前回合数
    - is_game_over: 是否已结案
    """
    opening = generator.opening_scene().strip()
    chat_display: List[ChatMessage] = [
        {"role": "assistant", "content": opening}
    ]
    game_history: List[Tuple[str, str]] = []
    case_state = init_case_state()
    if mode == "叙事模式":
        status = f"叙事模式已开始：0/{MAX_NARRATIVE_TURNS} 回合。"
    else:
        status = "自由模式已开始：回合不设上限。"
    return chat_display, game_history, status, 0, False, case_state


def submit_action(
    user_input: str,
    chat_display: List[ChatMessage],
    game_history: List[Tuple[str, str]],
    mode: str,
    turn_count: int,
    is_game_over: bool,
) -> Tuple[List[ChatMessage], List[Tuple[str, str]], str, str, int, bool]:
    """
    处理玩家输入并返回 UI 所需更新。

    返回:
    - 更新后的 chat_display
    - 更新后的 game_history
    - 状态文本
    - 清空后的输入框内容
    - 更新后的 turn_count
    - 更新后的 is_game_over
    """
    user_text = (user_input or "").strip()
    if not user_text:
        return chat_display, game_history, "请输入你的行动或推理。", "", turn_count, is_game_over
    if len(user_text) > 100:
        return (
            chat_display,
            game_history,
            "输入超出 100 字，请精简后再提交。",
            user_text[:100],
            turn_count,
            is_game_over,
        )

    if mode == "叙事模式" and is_game_over:
        return (
            chat_display,
            game_history,
            "本案已结案，请点击“重新开始”开启新案件。",
            "",
            turn_count,
            is_game_over,
        )

    next_turn = turn_count + 1

    try:
        ai_reply = generator.generate_next_with_control(
            player_input=user_text,
            history=game_history,
            mode=mode,
            turn=next_turn,
            max_turns=MAX_NARRATIVE_TURNS,
        )
        ai_reply = _remove_blank_lines(ai_reply)
        ai_reply = _normalize_message_blocks(ai_reply)
    except Exception as exc:  # noqa: BLE001
        ai_reply = (
            "系统提示：当前无法连接剧情引擎，请检查 API Key、网络或配额后重试。\n"
            f"错误信息：{exc}"
        )

    # 展示层历史：Gradio 6 推荐使用 messages 格式。
    chat_display = chat_display + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": ai_reply},
    ]

    # 模型层历史：只记录用户与助手文本对，供后续上下文续写。
    game_history = game_history + [(user_text, ai_reply)]

    if mode == "叙事模式":
        finished = next_turn >= MAX_NARRATIVE_TURNS
        if finished:
            status = f"叙事模式：{next_turn}/{MAX_NARRATIVE_TURNS} 回合，真相已揭示，案件已结案。"
        else:
            status = f"叙事模式：{next_turn}/{MAX_NARRATIVE_TURNS} 回合，剧情推进中。"
    else:
        finished = False
        status = f"自由模式：第 {next_turn} 回合，剧情已更新。"

    return chat_display, game_history, status, "", next_turn, finished


def start_new_game_ui(mode: str) -> Tuple[str, List[ChatMessage], List[Tuple[str, str]], str, int, bool, str, Dict[str, object]]:
    """开局并同步返回剧情区 HTML。"""
    chat_display, game_history, status, turn_count, is_game_over, case_state = start_new_game(mode)
    story_html = render_chat_html(chat_display)
    session_id = str(uuid.uuid4())
    return story_html, chat_display, game_history, status, turn_count, is_game_over, session_id, case_state


def submit_action_ui(
    user_input: str,
    chat_display: List[ChatMessage],
    game_history: List[Tuple[str, str]],
    mode: str,
    turn_count: int,
    is_game_over: bool,
) -> Tuple[str, List[ChatMessage], List[Tuple[str, str]], str, str, int, bool]:
    """提交行动并同步返回剧情区 HTML。"""
    next_chat, next_history, status, cleared_input, next_turn, finished = submit_action(
        user_input=user_input,
        chat_display=chat_display,
        game_history=game_history,
        mode=mode,
        turn_count=turn_count,
        is_game_over=is_game_over,
    )
    story_html = render_chat_html(next_chat)
    return story_html, next_chat, next_history, status, cleared_input, next_turn, finished


def submit_action_stream(
    user_input: str,
    immersion_score: int,
    chat_display: List[ChatMessage],
    game_history: List[Tuple[str, str]],
    mode: str,
    turn_count: int,
    is_game_over: bool,
    session_id: str,
    case_state: Dict[str, object],
) -> Generator[Tuple[str, List[ChatMessage], List[Tuple[str, str]], str, str, int, bool, str, Dict[str, object]], None, None]:
    """先回显用户输入与AI占位气泡，再在生成完成后替换为真实回复。"""
    user_text = (user_input or "").strip()
    if not user_text:
        yield render_chat_html(chat_display), chat_display, game_history, "请输入你的行动或推理。", "", turn_count, is_game_over, session_id, case_state
        return

    if len(user_text) > 100:
        yield (
            render_chat_html(chat_display),
            chat_display,
            game_history,
            "输入超出 100 字，请精简后再提交。",
            user_text[:100],
            turn_count,
            is_game_over,
            session_id,
            case_state,
        )
        return

    if mode == "叙事模式" and is_game_over:
        yield (
            render_chat_html(chat_display),
            chat_display,
            game_history,
            "本案已结案，请点击“重新开始”开启新案件。",
            "",
            turn_count,
            is_game_over,
            session_id,
            case_state,
        )
        return

    next_turn = turn_count + 1
    pending_chat = chat_display + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": "", "pending": True},
    ]

    # 第一阶段：立即回显用户输入，并显示 AI 加载占位气泡。
    yield (
        render_chat_html(pending_chat),
        pending_chat,
        game_history,
        "AI 正在分析线索...",
        "",
        turn_count,
        is_game_over,
        session_id,
        case_state,
    )

    started_at = time.perf_counter()
    if intent_model is not None:
        try:
            intent_prediction = intent_model.predict(user_text)
            intent_label = intent_prediction.label
            intent_confidence = intent_prediction.confidence
            intent_source = "model"
        except Exception:
            intent_label = detect_intent(user_text)
            intent_confidence = 0.0
            intent_source = "rule"
    else:
        intent_label = detect_intent(user_text)
        intent_confidence = 0.0
        intent_source = "rule"
    intent_text = INTENT_LABELS.get(intent_label, intent_label)
    intent_prompt_text = intent_text if ENABLE_INTENT_PROMPT else None
    case_summary = summarize_case_state(case_state) if ENABLE_CASE_MEMORY else None
    dialogue_intent_label = intent_text if ENABLE_INTENT_PROMPT else "未显式注入意图"
    dialogue_intent_confidence = intent_confidence if ENABLE_INTENT_PROMPT else 0.0
    dialogue_case_memory = case_summary if ENABLE_CASE_MEMORY else ""
    dialogue_plan = dialogue_manager.build_control_prompt(
        mode=mode,
        turn=next_turn,
        max_turns=MAX_NARRATIVE_TURNS,
        intent_label=dialogue_intent_label,
        intent_confidence=dialogue_intent_confidence,
        case_memory=dialogue_case_memory,
    )
    candidate_count = 1
    selected_candidate_index = 0
    rerank_method = "single"
    try:
        raw_candidates = generator.generate_candidates_with_control(
            player_input=user_text,
            history=game_history,
            mode=mode,
            turn=next_turn,
            max_turns=MAX_NARRATIVE_TURNS,
            intent_label=intent_prompt_text,
            case_memory=case_summary,
            control_prompt_override=dialogue_plan.control_prompt,
            num_candidates=3,
        )
        candidate_count = len(raw_candidates)

        normalized_candidates = [
            _normalize_message_blocks(_remove_blank_lines(text)) for text in raw_candidates
        ]

        if coherence_model is not None and normalized_candidates:
            history_context = "；".join(f"用户:{u} | 回复:{a}" for u, a in game_history[-3:]) or case_summary or ""
            scored_candidates: List[Tuple[float, int, str]] = []
            for idx, candidate_text in enumerate(normalized_candidates):
                try:
                    pred = coherence_model.predict(
                        history_context=history_context,
                        candidate_reply=candidate_text,
                    )
                    score = float(pred.probability)
                except Exception:
                    score = -1.0
                scored_candidates.append((score, idx, candidate_text))

            scored_candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)
            selected_candidate_index = scored_candidates[0][1]
            ai_reply = scored_candidates[0][2]
            rerank_method = "coherence_model"
        else:
            ai_reply = normalized_candidates[0]
    except Exception as exc:  # noqa: BLE001
        ai_reply = (
            "系统提示：当前无法连接剧情引擎，请检查 API Key、网络或配额后重试。\n"
            f"错误信息：{exc}"
        )
    latency_ms = int((time.perf_counter() - started_at) * 1000)

    final_chat = chat_display + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": ai_reply},
    ]
    final_history = game_history + [(user_text, ai_reply)]

    if mode == "叙事模式":
        finished = next_turn >= MAX_NARRATIVE_TURNS
        if finished:
            status = f"叙事模式：{next_turn}/{MAX_NARRATIVE_TURNS} 回合，真相已揭示，案件已结案。"
        else:
            status = f"叙事模式：{next_turn}/{MAX_NARRATIVE_TURNS} 回合，剧情推进中。"
    else:
        finished = False
        status = f"自由模式：第 {next_turn} 回合，剧情已更新。"

    metric_scores = evaluate_turn_metrics(
        user_input=user_text,
        ai_reply=ai_reply,
        history=game_history,
    )
    if coherence_model is not None:
        try:
            coherence_prediction = coherence_model.predict(
                history_context="；".join(f"用户:{u} | 回复:{a}" for u, a in game_history[-3:]) or case_summary or "",
                candidate_reply=ai_reply,
            )
            model_coherence_probability = coherence_prediction.probability
            model_coherence_label = coherence_prediction.label
        except Exception:
            model_coherence_probability = None
            model_coherence_label = None
    else:
        model_coherence_probability = None
        model_coherence_label = None
    next_case_state = update_case_state(
        state=case_state,
        turn=next_turn,
        user_input=user_text,
        ai_reply=ai_reply,
        intent_label=intent_label,
    )
    status = (
        f"{status} | 响应 {latency_ms}ms | 连贯性 {metric_scores['plot_coherence_score']}/5"
        f" | 匹配度 {metric_scores['choice_match_score']}/5 | 意图 {intent_text}"
        f" | 意图模型 {intent_confidence:.2f} | 实验 {EXPERIMENT_TAG}"
    )

    experiment_logger.log_turn(
        {
            "session_id": session_id,
            "mode": mode,
            "turn": next_turn,
            "is_game_over": finished,
            "user_input": user_text,
            "ai_reply": ai_reply,
            "latency_ms": latency_ms,
            "immersion_score": int(immersion_score),
            "intent_label": intent_label,
            "intent_model_source": intent_source,
            "intent_model_confidence": intent_confidence,
            "used_intent_prompt": ENABLE_INTENT_PROMPT,
            "used_case_memory": ENABLE_CASE_MEMORY,
            "experiment_tag": EXPERIMENT_TAG,
            "dialogue_phase": dialogue_plan.phase,
            "dialogue_branch_focus": dialogue_plan.branch_focus,
            "dialogue_control_prompt": dialogue_plan.control_prompt,
            "coherence_model_probability": model_coherence_probability,
            "coherence_model_label": model_coherence_label,
            "candidate_count": candidate_count,
            "selected_candidate_index": selected_candidate_index,
            "rerank_method": rerank_method,
            "timeline_size": len(next_case_state.get("timeline", [])),
            "evidence_size": len(next_case_state.get("evidence", [])),
            **metric_scores,
        }
    )

    # 第二阶段：替换占位气泡为真实回复，并滚动到最新消息。
    yield render_chat_html(final_chat), final_chat, final_history, status, "", next_turn, finished, session_id, next_case_state


def submit_human_evaluation(
    narrative_quality: int,
    response_speed: int,
    choice_match: int,
    coherence: int,
    immersion: int,
    overall: int,
    comments: str,
    session_id: str,
    mode: str,
    turn_count: int,
) -> str:
    """记录局末人工评测结果。"""
    append_human_evaluation(
        {
            "session_id": session_id,
            "mode": mode,
            "turn_count": turn_count,
            "experiment_tag": EXPERIMENT_TAG,
            "narrative_quality": narrative_quality,
            "response_speed": response_speed,
            "choice_match": choice_match,
            "coherence": coherence,
            "immersion": immersion,
            "overall": overall,
            "comments": comments,
        }
    )
    return "人工评测已记录。"


def build_interface() -> gr.Blocks:
    """构建 Gradio UI。"""
    with gr.Blocks(
        title="StoryWeaver - AI 文字冒险",
        css="""
        #header-row { align-items: flex-start; }
        #mode-panel { padding-top: 6px; }
        #mode-panel .label-wrap,
        #status-box .label-wrap,
        #action-input .label-wrap {
            min-height: 28px;
        }
        #mode-panel .label-wrap .label-text,
        #status-box .label-wrap .label-text,
        #action-input .label-wrap .label-text {
            font-size: 14px;
            line-height: 1.4;
            font-weight: 600;
        }
        #layout-root {
            height: calc(100vh - 240px);
            min-height: 0;
        }
        #left-panel {
            height: calc(100vh - 240px);
            min-height: 0;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            gap: 0;
        }
        #story-label {
            height: 32px;
            line-height: 32px;
            font-size: 16px;
            font-weight: 700;
            margin: 0;
            padding: 0 8px;
            flex-shrink: 0;
            border-bottom: 1px solid #e5e5e5;
        }
        #story-panel {
            flex: 1 1 auto;
            min-height: 320px;
            display: block;
        }
        #story-panel .html-container {
            min-height: 320px;
            padding: 0;
            margin: 0;
        }
        #story-shell {
            height: 100%;
            min-height: 0;
            border: 1px solid #e5e7eb;
            border-top: none;
            border-radius: 0 0 10px 10px;
            background: #fcfdff;
            overflow: hidden;
        }
        #story-scroll {
            height: 100%;
            min-height: 0;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 10px 12px;
            box-sizing: border-box;
        }
        #story-list {
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .story-empty {
            color: #6b7280;
            font-size: 14px;
            line-height: 1.5;
            padding: 8px 10px;
        }
        .story-row {
            display: flex;
            margin: 0;
        }
        .story-assistant {
            justify-content: flex-start;
        }
        .story-user {
            justify-content: flex-end;
        }
        .story-bubble {
            max-width: 90%;
            margin: 0;
            padding: 10px 12px;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            box-sizing: border-box;
        }
        .story-assistant .story-bubble {
            background: #ffffff;
            border-color: #dbeafe;
        }
        .story-user .story-bubble {
            background: #ecfeff;
            border-color: #a5f3fc;
        }
        .story-role {
            margin: 0 0 4px 0;
            font-size: 12px;
            color: #475569;
            font-weight: 700;
        }
        .story-content {
            margin: 0;
            padding: 0;
            white-space: normal;
            word-break: break-word;
            line-height: 1.5;
            font-size: 14px;
            color: #0f172a;
        }
        .story-content strong {
            font-weight: 700;
            color: #111827;
        }
        #right-panel {
            display: flex;
            flex-direction: column;
            height: calc(100vh - 240px);
            min-height: 0;
            gap: 12px;
        }
        #right-panel > * {
            width: 100%;
        }
        #status-box {
            flex: 0 0 120px;
            min-height: 120px !important;
        }
        #status-box .wrap,
        #status-box textarea {
            min-height: 76px !important;
            max-height: 76px !important;
            height: 76px !important;
            font-size: 14px;
            line-height: 1.4;
        }
        #action-input {
            flex: 1 1 auto;
            min-height: 180px !important;
        }
        #action-input .wrap,
        #action-input textarea {
            min-height: 150px !important;
            max-height: 320px !important;
            height: 100% !important;
        }
        #action-buttons {
            margin-top: 0;
            display: grid;
            grid-template-columns: 1fr 1fr;
            flex: 0 0 56px;
            min-height: 56px !important;
            gap: 10px;
            align-items: stretch;
        }
        #send-btn button,
        #restart-btn button {
            min-height: 48px !important;
            max-height: 48px !important;
            height: 48px !important;
            font-size: 15px !important;
        }
        """,
    ) as demo:
        with gr.Row(elem_id="header-row"):
            with gr.Column(scale=7):
                gr.Markdown(
                    """
                    # StoryWeaver - AI 文字冒险
                    你将扮演私家侦探 **林深**，调查一宗离奇失踪案。  
                    输入你的行动、提问或推理，推动剧情发展。
                    """
                )
            with gr.Column(scale=5, elem_id="mode-panel"):
                mode_selector = gr.Radio(
                    choices=["自由模式", "叙事模式"],
                    value="自由模式",
                    label="游戏模式",
                    info="自由模式：无回合上限；叙事模式：最多 20 回合并在结尾揭示真相。",
                )

        with gr.Row(equal_height=True, elem_id="layout-root"):
            with gr.Column(scale=7, elem_id="left-panel"):
                gr.HTML('<div id="story-label">📖 剧情内容</div>')
                story_panel = gr.HTML(
                    value=render_chat_html([]),
                    elem_id="story-panel",
                    container=False,
                )

            with gr.Column(scale=5, elem_id="right-panel"):
                status = gr.Textbox(
                    label="系统状态",
                    interactive=False,
                    lines=4,
                    max_lines=4,
                    elem_id="status-box",
                )
                user_input = gr.Textbox(
                    label="你的行动",
                    placeholder="例如：我先检查摔裂手机的最近通话记录，并询问报案人昨晚时间线。",
                    lines=9,
                    max_lines=9,
                    max_length=100,
                    elem_id="action-input",
                )
                immersion_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="沉浸感满意度（1-5）",
                    info="提交本回合行动时一并记录，用于实验评估。",
                )
                with gr.Row(elem_id="action-buttons"):
                    send_btn = gr.Button("提交行动", variant="primary", elem_id="send-btn")
                    restart_btn = gr.Button("重新开始", elem_id="restart-btn")

                with gr.Accordion("人工评测（建议通关后填写）", open=False):
                    eval_narrative = gr.Slider(1, 5, value=4, step=1, label="叙事质量")
                    eval_speed = gr.Slider(1, 5, value=4, step=1, label="响应速度")
                    eval_choice = gr.Slider(1, 5, value=4, step=1, label="选择匹配")
                    eval_coherence = gr.Slider(1, 5, value=4, step=1, label="情节连贯")
                    eval_immersion = gr.Slider(1, 5, value=4, step=1, label="沉浸体验")
                    eval_overall = gr.Slider(1, 5, value=4, step=1, label="总体满意度")
                    eval_comments = gr.Textbox(label="补充意见", placeholder="例如：某回合逻辑跳跃、某段对话很自然", lines=3)
                    eval_submit = gr.Button("提交人工评测")
                    eval_status = gr.Textbox(label="评测状态", interactive=False, lines=2)

        # 两个状态变量：一个用于 UI 展示历史，一个用于模型上下文历史。
        chat_state = gr.State([])
        game_state = gr.State([])
        turn_state = gr.State(0)
        game_over_state = gr.State(False)
        session_state = gr.State("")
        case_state = gr.State(init_case_state())

        # 页面加载时自动开局。
        demo.load(
            fn=start_new_game_ui,
            inputs=[mode_selector],
            outputs=[story_panel, chat_state, game_state, status, turn_state, game_over_state, session_state, case_state],
        )

        mode_selector.change(
            fn=start_new_game_ui,
            inputs=[mode_selector],
            outputs=[story_panel, chat_state, game_state, status, turn_state, game_over_state, session_state, case_state],
        )

        send_btn.click(
            fn=submit_action_stream,
            inputs=[user_input, immersion_slider, chat_state, game_state, mode_selector, turn_state, game_over_state, session_state, case_state],
            outputs=[story_panel, chat_state, game_state, status, user_input, turn_state, game_over_state, session_state, case_state],
            show_progress="hidden",
        )

        user_input.submit(
            fn=submit_action_stream,
            inputs=[user_input, immersion_slider, chat_state, game_state, mode_selector, turn_state, game_over_state, session_state, case_state],
            outputs=[story_panel, chat_state, game_state, status, user_input, turn_state, game_over_state, session_state, case_state],
            show_progress="hidden",
        )

        restart_btn.click(
            fn=start_new_game_ui,
            inputs=[mode_selector],
            outputs=[story_panel, chat_state, game_state, status, turn_state, game_over_state, session_state, case_state],
        )

        eval_submit.click(
            fn=submit_human_evaluation,
            inputs=[
                eval_narrative,
                eval_speed,
                eval_choice,
                eval_coherence,
                eval_immersion,
                eval_overall,
                eval_comments,
                session_state,
                mode_selector,
                turn_state,
            ],
            outputs=[eval_status],
            show_progress="hidden",
        )

    return demo


if __name__ == "__main__":
    app = build_interface()
    launch_port = _pick_launch_port()
    print(f"Launching StoryWeaver on port {launch_port}")
    app.launch(server_name="0.0.0.0", server_port=launch_port, theme=gr.themes.Soft())
