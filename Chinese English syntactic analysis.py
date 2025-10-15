# !pip install ltp spacy
# !python -m spacy download en_core_web_sm

import sys
from typing import List
from wcwidth import wcswidth

chinese_file = r'resourse\\Chinese news.txt'
english_file = r'resourse\\English news.txt'
chinese_output = r'syntax_results\\chinese_syntax.txt'
english_output = r'syntax_results\\english_syntax.txt'

# 辅助函数：表格与树状输出 ----------
def _display_width(s: str) -> int:
    """
    返回字符串在等宽字体/终端下的大致显示宽度。
    wcswidth 在遇到不可打印字符时返回 -1，fallback 回 len(s)。
    """
    try:
        w = wcswidth(s)
        return w if w >= 0 else len(s)
    except Exception:
        return len(s)

def _pad_to_display(s: str, width: int) -> str:
    """
    根据目标显示宽度 width 补空格，返回被填充后的字符串。
    这里我们始终在右侧填充空格（左对齐）。
    """
    cur = _display_width(s)
    if cur >= width:
        return s
    # 直接补 ascii 空格，等宽字体下效果最好
    return s + " " * (width - cur)

def table_text(words, pos_tags, heads, rels):
    """
    返回按显示宽度对齐的表格文本（支持中英混合更整齐）。
    header: idx | token | pos | head_idx | head_token | rel
    """
    n = len(words)
    rows = []
    header = ["idx", "token", "pos", "head_idx", "head_token", "rel"]
    rows.append(header)
    for i in range(1, n+1):
        head_idx = heads[i-1]
        head_token = words[head_idx-1] if (1 <= head_idx <= n) else "ROOT"
        rows.append([str(i), words[i-1], pos_tags[i-1], str(head_idx), head_token, rels[i-1]])

    # 计算每列的最大显示宽度（用 wcswidth）
    num_cols = len(header)
    col_display_widths = [0] * num_cols
    for r in rows:
        for j in range(num_cols):
            col_display_widths[j] = max(col_display_widths[j], _display_width(r[j]))

    # 生成行字符串（用 display-padding）
    lines = []
    # header
    hline = " | ".join(_pad_to_display(rows[0][j], col_display_widths[j]) for j in range(num_cols))
    sep = "-+-".join("-" * col_display_widths[j] for j in range(num_cols))
    lines.append(hline)
    lines.append(sep)
    for row in rows[1:]:
        lines.append(" | ".join(_pad_to_display(row[j], col_display_widths[j]) for j in range(num_cols)))
    return "\n".join(lines)


def build_tree_text(words: List[str], pos_tags: List[str], heads: List[int], rels: List[str], max_depth: int = 2000) -> str:
    """
    更稳健的树状文本渲染：
    - 把非法 head 当作 ROOT 处理
    - 检测循环（visited set），若发现 cycle 标注为 <cycle>
    - 超出 max_depth 标注为 <maxdepth>
    """
    n = len(words)
    from collections import defaultdict
    children = defaultdict(list)
    roots = []
    for i, h in enumerate(heads, start=1):
        if h == 0 or h == i:
            roots.append(i)
        elif 1 <= h <= n:
            children[h].append(i)
        else:
            # 非法 head，视为 root
            roots.append(i)

    sys.setrecursionlimit(max(1000, n + 50))

    def render(node, visited, depth):
        if depth > max_depth:
            return f"({words[node-1]}_{pos_tags[node-1] if node-1 < len(pos_tags) else '_'} <maxdepth>)"
        if node in visited:
            return f"({words[node-1]}_{pos_tags[node-1] if node-1 < len(pos_tags) else '_'} <cycle>)"
        visited.add(node)
        inner = []
        for c in children.get(node, []):
            inner.append(render(c, visited, depth + 1))
        visited.remove(node)
        token_repr = f"{words[node-1]}_{pos_tags[node-1] if node-1 < len(pos_tags) else '_'}"
        if inner:
            return f"({token_repr} {' '.join(inner)})"
        else:
            return f"({token_repr})"

    if not roots:
        # 兜底：若无 root，按序输出每个 token（避免无限递归）
        return " ".join(render(i, set(), 0) for i in range(1, n + 1))
    return "\n".join(render(r, set(), 0) for r in roots)

# 中文分析
def analyze_chinese(in_path: str, out_path: str, debug_limit: int = 300):
    """
    更健壮地处理 ltp.pipeline 的多种返回格式。
    debug_limit: 若需要输出调试信息，最多写入多少字符（避免写入巨量对象）
    """
    print(f"[LTP] 开始中文分析：{in_path}")
    try:
        from ltp import LTP
    except Exception as e:
        raise RuntimeError("请先安装 ltp：pip install ltp") from e

    ltp = LTP()

    with open(in_path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    with open(out_path, 'w', encoding='utf-8') as fout:
        fout.write("=== 中文句法分析 (LTP) ===\n\n")
        for idx, sent in enumerate(lines, start=1):
            fout.write(f"--- Sentence {idx} ---\n")
            fout.write(sent + "\n\n")

            words = []
            pos = []
            heads = []
            rels = []

            try:
                # 使用 pipeline（新版 API）
                output = ltp.pipeline([sent], tasks=["cws", "pos", "dep"])
            except Exception as e:
                fout.write(f"[ERROR] 调用 ltp.pipeline 时出现异常：{e}\n\n")
                continue

            # 通用获取器：安全提取 output 中给定 key 的第一个结果（若存在）
            def safe_extract(out_obj, key):
                """
                支持 Namespace-like (hasattr), dict-like, 或直接返回 list 的情况。
                返回：None 或者第0项（通常为该句的结果，如 tokens 列表 或 dep 列表）
                """
                val = None
                # hasattr 形式
                if hasattr(out_obj, key):
                    try:
                        val = getattr(out_obj, key)
                    except Exception:
                        val = None
                # dict 形式
                if val is None and isinstance(out_obj, dict) and key in out_obj:
                    val = out_obj[key]
                # 有时 output 本身就是 tuple/list，尝试取 key as index 名称（不常见）
                # 最后，若是列表或元组，返还第0项（通常 pipeline([sent]) 的返回里每个 key 对应一个 list-of-sentences）
                if val is None:
                    # fallback: if out_obj itself is indexable and contains key-like entries, skip
                    pass

                # 规范化：如果 val 是 list/tuple，返回 val[0]（句子级别）
                if isinstance(val, (list, tuple)) and len(val) >= 1:
                    return val[0]
                # 其它可直接返回的标量/对象
                return val

            # 取 tokens（cws）与 pos
            try:
                cws = safe_extract(output, "cws")
                pos_out = safe_extract(output, "pos")
                dep_out = safe_extract(output, "dep")
            except Exception as e:
                fout.write(f"[ERROR] 提取 output 字段时异常：{e}\n\n")
                # 尝试写出部分 debug 信息（截断）
                try:
                    dbg = repr(output)
                    fout.write(f"[DEBUG] output repr (truncated): {dbg[:debug_limit]!r}\n\n")
                except Exception:
                    pass
                continue

            # 规范化 tokens 与 pos 为字符串列表
            try:
                if cws is None:
                    words = []
                elif isinstance(cws, (list, tuple)):
                    words = [str(x) for x in cws]
                else:
                    # 兜底：单个字符串或其他类型，尝试用空格切分
                    words = str(cws).split()
            except Exception:
                words = []

            try:
                if pos_out is None:
                    pos = ["_"] * len(words)
                elif isinstance(pos_out, (list, tuple)):
                    pos = [str(x) for x in pos_out]
                else:
                    pos = [str(pos_out)] * len(words)
            except Exception:
                pos = ["_"] * len(words)

            # 处理 dep 输出，可能的常见形式：
            #  1) [(head, rel), ...]
            #  2) [{'head': h, 'rel': r}, ...]
            #  3) 其他（则退回默认）
            try:
                if dep_out is None:
                    heads = [0] * len(words)
                    rels = ["_"] * len(words)
                elif isinstance(dep_out, (list, tuple)) and len(dep_out) > 0:
                    first = dep_out[0]
                    if isinstance(first, (list, tuple)) and len(first) >= 2:
                        # 形如 [(h, rel), ...]
                        heads = [int(item[0]) if item[0] is not None else 0 for item in dep_out]
                        rels  = [str(item[1]) for item in dep_out]
                    elif isinstance(first, dict):
                        # 形如 [{'head':h, 'rel':r}, ...]，不同版本 key 可能不同
                        heads = []
                        rels = []
                        for d in dep_out:
                            if not isinstance(d, dict):
                                heads.append(0)
                                rels.append("_")
                                continue
                            # 尝试常见键名
                            h = d.get("head", d.get("H", d.get("HEAD", 0)))
                            r = d.get("rel", d.get("deprel", d.get("dep", d.get("r", "_"))))
                            try:
                                heads.append(int(h))
                            except Exception:
                                heads.append(0)
                            rels.append(str(r))
                    else:
                        # 元素是标量或其它情况，退回默认（避免抛异常）
                        heads = [0] * len(words)
                        rels = ["_"] * len(words)
                else:
                    heads = [0] * len(words)
                    rels = ["_"] * len(words)
            except Exception as e:
                # 若解析 dep 出错，记录并退回默认
                fout.write(f"[WARN] 解析 dep 输出时发生异常：{e}\n")
                try:
                    dbg = repr(dep_out)
                    fout.write(f"[DEBUG] dep_out repr (truncated): {dbg[:debug_limit]!r}\n\n")
                except Exception:
                    pass
                heads = [0] * len(words)
                rels = ["_"] * len(words)

            # 最终长度对齐（防止不一致导致后续崩溃）
            n = len(words)
            if len(pos) != n:
                pos = (pos + ["_"] * n)[:n]
            if len(heads) != n:
                heads = (heads + [0] * n)[:n]
            if len(rels) != n:
                rels = (rels + ["_"] * n)[:n]

            # 输出表格与树状表示
            fout.write("Tokens & POS & Dependency (table):\n")
            fout.write(table_text(words, pos, heads, rels) + "\n\n")
            fout.write("Tree-like representation:\n")
            try:
                fout.write(build_tree_text(words, pos, heads, rels) + "\n\n\n")
            except Exception as e:
                fout.write(f"[ERROR] 生成树状表示时异常：{e}\n\n")

    print(f"[LTP] 中文分析完成，结果写入：{out_path}")


# 英文分析（spaCy）
def analyze_english(in_path: str, out_path: str):
    print(f"[spaCy] 开始英文分析：{in_path}")
    try:
        import spacy
    except Exception as e:
        raise RuntimeError("请先安装 spaCy：pip install spacy") from e

    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

    with open(in_path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    with open(out_path, 'w', encoding='utf-8') as fout:
        fout.write("=== English Syntax Analysis (spaCy) ===\n\n")
        for idx, sent in enumerate(lines, start=1):
            fout.write(f"--- Sentence {idx} ---\n")
            fout.write(sent + "\n\n")
            doc = nlp(sent)
            tokens = [t for t in doc]
            words = [t.text for t in tokens]
            pos = [t.pos_ for t in tokens]
            rels = [t.dep_ for t in tokens]

            # 注意：spaCy 中 root 的 t.head == t，本处把 root 明确映射为 0（与 LTP 保持一致）
            token_i_map = {t.i: i + 1 for i, t in enumerate(tokens)}  # 全局 doc index -> 本句 1-based idx
            heads = []
            for t in tokens:
                if t.dep_.upper() == "ROOT" or t.head.i == t.i:
                    heads.append(0)
                else:
                    heads.append(token_i_map.get(t.head.i, 0))

            fout.write("Tokens & POS & Dependency (table):\n")
            fout.write(table_text(words, pos, heads, rels) + "\n\n")
            # 使用新 build_tree_text（增加 max_depth 防护）
            fout.write("Tree-like representation:\n")
            try:
                fout.write(build_tree_text(words, pos, heads, rels) + "\n\n")
            except RecursionError:
                fout.write("[ERROR] 生成树时递归超限，已中断。可能存在循环依存关系。\n\n")
            fout.write("\n")
    print(f"[spaCy] 英文分析完成，结果写入：{out_path}")
    

def main():
    analyze_chinese(chinese_file, chinese_output)
    analyze_english(english_file, english_output)
    print("全部完成。")

if __name__ == "__main__":
    main()
