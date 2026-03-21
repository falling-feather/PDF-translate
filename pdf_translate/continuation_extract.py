"""从已完成的译文中截取少量「段尾」，供下一块串联翻译时做语气/指代衔接（控制 token）。"""

from __future__ import annotations


def translation_tail_for_next_chunk(zh: str, *, max_chars: int = 480) -> str:
    """
    取最后一个非空段落的尾部，上限 max_chars。
    不调用模型；用于写入 memory，供下一块 prompt 使用。
    """
    z = (zh or "").strip()
    if not z:
        return ""
    paras = [p.strip() for p in z.split("\n\n") if p.strip()]
    tail = paras[-1] if paras else z
    if len(tail) > max_chars:
        tail = tail[-max_chars:].lstrip()
    return tail
