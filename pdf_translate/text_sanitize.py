"""弱化 PDF 抽取中的目录点线等噪声，减轻模型照抄整行点号的问题。"""

from __future__ import annotations


def collapse_toc_dot_leaders(text: str, *, min_run: int = 24) -> str:
    """
    删除「几乎只有点/间隔符」的行（常见于目录引导线）。
    不处理含字母数字的正常句子。
    """
    out_lines: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if len(s) >= min_run:
            # 仅由点、中点、省略号、空白组成
            if all(c in ".·… \t\u3000" for c in s):
                continue
        out_lines.append(line)
    return "\n".join(out_lines)
