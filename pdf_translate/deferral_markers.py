"""串联翻译「段尾原文顺延」：模型输出中的标识符与合并后纯规则处理（不调 LLM）。"""

from __future__ import annotations

import re

# 分段：合并后替换为段落间隔
MARKER_FENDUAN = "《&fenduan&》"
# 分句：合并后删除标识符，与上文无缝拼接
MARKER_FENJU = "《&fenju&》"

_MARKERS = (MARKER_FENDUAN, MARKER_FENJU)


def parse_model_output_with_deferral(
    raw: str,
    *,
    use_deferral: bool,
) -> tuple[str, str]:
    """
    从模型完整输出中分离「写入 md 的正文（可含标识符行）」与「顺延至下一块的英文原文」。

    use_deferral 为 False 时：整段视为译文，顺延为空。
    期望格式（最后一行英文为顺延，之前为中文译文 + 单独一行的标识符）：
        …中文译文…
        《&fenduan&》
        The deferred English tail copied from source...
    """
    t = (raw or "").strip()
    if not use_deferral or not t:
        return t, ""

    # 从后往前找第一个合法标识符，避免正文中误出现相同字面量
    best: tuple[int, str] | None = None  # (index, marker)
    for m in _MARKERS:
        i = t.rfind(m)
        if i >= 0 and (best is None or i > best[0]):
            best = (i, m)
    if best is None:
        return t, ""

    idx, marker = best
    zh_and_marker = t[: idx + len(marker)].strip()
    deferred = t[idx + len(marker) :].strip()
    # 允许标识符单独成行；正文区保留标识符供合并后替换
    return zh_and_marker, deferred


def finalize_merged_translation_markdown(md: str) -> str:
    """
    合并全文后调用：将标识符替换为版式结果，不产生新 LLM 调用。
    - 《&fenduan&》→ 双换行（段落分界）
    - 《&fenju&》→ 空串（句间衔接）
    """
    s = md
    s = s.replace(MARKER_FENDUAN, "\n\n")
    s = s.replace(MARKER_FENJU, "")
    # 压缩因替换产生的过多空行（保留最多双换行）
    s = re.sub(r"\n{4,}", "\n\n\n", s)
    return s.strip() + ("\n" if s.strip() else "")


DEFERRAL_PROTOCOL_USER_BLOCK = """【段尾顺延协议 — 当前为文档分段中的非最后一段，必须遵守】
1. 在【待译正文】（及若有的「紧接上段未译英文」）中，将末尾约 8%～20% 的原文保留为英文不译，只输出其之前部分的简体中文译文。
2. 切分点优先选在段落边界；若只能落在句中，下一块需与当前译文同段无缝衔接。
3. 输出结构固定为三节（不要加额外标题或说明）：
   （一）已译中文正文；
   （二）单独一行，且整行只能是以下二者之一：《&fenduan&》 或 《&fenju&》
       · 《&fenduan&》 表示切在段落边界，合并全文时会在两处译文之间留出段落空行；
       · 《&fenju&》 表示切在句中，合并时会去掉该标记使前后中文直接相连；
   （三）从下一行起，输出未译部分的英文，必须与原文对应片段完全一致（可复制粘贴），直至该段原文结束。
4. 标识符必须字面完全一致，勿增删空格或全角符号。"""


def strip_markers_from_plain_text(s: str) -> str:
    """摘要/段尾记忆等场景，去掉顺延标记字面量。"""
    t = s
    for m in _MARKERS:
        t = t.replace(m, "")
    return t.strip()


def strip_yaml_front_matter(chunk_file_text: str) -> str:
    """单个块文件可能含 --- meta ---，合并完整译文时只取正文区。"""
    lines = chunk_file_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return chunk_file_text.strip()
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[i + 1 :]).lstrip("\n")
    return chunk_file_text.strip()
