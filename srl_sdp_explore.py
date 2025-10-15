
# pip install ltp spacy wcwidth
# pip install allennlp allennlp-models   # 可选（英文 SRL），较大
# python -m spacy download en_core_web_sm

from typing import List
import os
from wcwidth import wcswidth

english_file = r'resourse\\English news.txt'

corpus_paths = [
    r"resourse\\Chinese news.txt",   # 新闻语料
    r"resourse\\Chinese novel.txt",  # 小说语料
    r"resourse\\Chinese poem.txt"    # 诗歌语料
]
output_dir = r"srl_sdp_output"
os.makedirs(output_dir, exist_ok=True)


# 显示宽度/表格工具（支持中英文混排对齐）
def _display_width(s: str) -> int:
    try:
        w = wcswidth(s)
        return w if w >= 0 else len(s)
    except Exception:
        return len(s)

def _pad_to_display(s: str, width: int) -> str:
    cur = _display_width(s)
    if cur >= width:
        return s
    return s + " " * (width - cur)

def table_text(words: List[str], pos_tags: List[str], heads: List[int], rels: List[str]) -> str:
    header = ["idx", "token", "pos", "head_idx", "head_token", "rel"]
    rows = [header]
    n = len(words)
    for i in range(1, n+1):
        h = heads[i-1] if i-1 < len(heads) else 0
        head_token = words[h-1] if 1 <= h <= n else "ROOT"
        pos = pos_tags[i-1] if i-1 < len(pos_tags) else "_"
        rel = rels[i-1] if i-1 < len(rels) else "_"
        rows.append([str(i), words[i-1], pos, str(h), head_token, rel])
    num_cols = len(header)
    col_display_widths = [0] * num_cols
    for r in rows:
        for j in range(num_cols):
            col_display_widths[j] = max(col_display_widths[j], _display_width(r[j]))
    lines = []
    lines.append(" | ".join(_pad_to_display(rows[0][j], col_display_widths[j]) for j in range(num_cols)))
    lines.append("-+-".join("-" * col_display_widths[j] for j in range(num_cols)))
    for r in rows[1:]:
        lines.append(" | ".join(_pad_to_display(r[j], col_display_widths[j]) for j in range(num_cols)))
    return "\n".join(lines)

# 帮助：把 SRL predicate-args 美观化
def format_srl(srl) -> str:
    """
    更鲁棒地格式化 SRL 输出，兼容多种常见返回格式：
    - None / empty -> "(no predicates)"
    - list of dicts (常见：{'verb':..., 'description':...} / {'predicate':..., 'arguments':[...]} )
    - list of tuples (如 (predicate, [(role, text), ...]) 或 (predicate, args_span) )
    - single tuple
    - 其它 -> 直接 str()
    返回多行可读字符串。
    """
    if not srl:
        return "(no predicates)"

    lines = []

    # 统一处理 list/tuple 的情形
    if isinstance(srl, (list, tuple)):
        # 若整个对象只是一个 tuple 且内层是基本类型，则把它当作单个 predicate 处理
        seq = list(srl) if not isinstance(srl, list) else srl

        for item in seq:
            # dict 风格（最常见）
            if isinstance(item, dict):
                verb = item.get("verb") or item.get("predicate") or item.get("pred") or item.get("action") or ""
                # 优先 human-friendly description
                if "description" in item and item["description"]:
                    lines.append(f"Predicate `{verb}` -> {item['description']}")
                    continue
                # 如果有 arguments 字段（list）
                if "arguments" in item and isinstance(item["arguments"], (list, tuple)):
                    arg_texts = []
                    for a in item["arguments"]:
                        if isinstance(a, dict):
                            lab = a.get("label") or a.get("arg") or a.get("role") or ""
                            txt = a.get("text") or a.get("span") or a.get("span_text") or str(a)
                            arg_texts.append(f"{lab}:{txt}")
                        else:
                            arg_texts.append(str(a))
                    lines.append(f"Predicate `{verb}` -> " + ", ".join(arg_texts))
                    continue
                # AllenNLP style might store role masks or tags; try to pretty print keys we know
                if "tags" in item and isinstance(item["tags"], (list, tuple)):
                    # tags like ["B-ARG0", "O", "B-V", ...] -> reconstruct spans (best-effort)
                    tags = item["tags"]
                    arg_spans = _reconstruct_spans_from_bio(tags, item.get("tokens"))
                    lines.append(f"Predicate `{verb}` -> " + ", ".join(arg_spans) )
                    continue
                # 兜底显示 dict 内容（简洁）
                lines.append(f"Predicate `{verb}` -> { {k:v for k,v in item.items() if k in ('arguments','description','tags') or len(str(v))<80} }")
                continue

            # tuple / list 风格
            if isinstance(item, (list, tuple)):
                # 常见形式： (predicate, args_list) 或 (predicate, [(role, text), ...])
                if len(item) >= 2:
                    pred = item[0]
                    args = item[1]
                    # args 为序列
                    if isinstance(args, (list, tuple)):
                        arg_texts = []
                        for a in args:
                            if isinstance(a, (list, tuple)):
                                # a 可能是 (role, span_text) 或 (role, start, end, text)
                                if len(a) >= 2:
                                    # 把第二项当作文本展示
                                    arg_texts.append(f"{a[0]}:{a[1]}")
                                else:
                                    arg_texts.append(str(a))
                            elif isinstance(a, dict):
                                lab = a.get("label") or a.get("role") or a.get("arg") or ""
                                txt = a.get("text") or a.get("span") or str(a)
                                arg_texts.append(f"{lab}:{txt}")
                            else:
                                arg_texts.append(str(a))
                        lines.append(f"Predicate `{pred}` -> " + ", ".join(arg_texts))
                        continue
                    else:
                        # args 不是序列，直接打印 repr
                        lines.append(f"Predicate `{pred}` -> {repr(args)}")
                        continue
                else:
                    lines.append(str(item))
                    continue

            # 其它类型（字符串、数字等）
            lines.append(str(item))

        return "\n".join(lines)

    # 其它不可识别类型，直接返回 str
    return str(srl)


# 辅助：一个简单的从 BIO tag 还原 spans 的小函数（best-effort）
def _reconstruct_spans_from_bio(tags, tokens=None):
    """
    tags: list like ["B-ARG0","I-ARG0","O","B-V",...]
    tokens: optional list of token strings
    返回 ['ARG0:...','ARG1:...'] 形式的列表（若不能重建，将返回原 tags）
    """
    if not isinstance(tags, (list, tuple)):
        return [str(tags)]
    spans = []
    cur_label = None
    cur_tokens = []
    for i, tg in enumerate(tags):
        if not tg or tg == "O":
            if cur_label:
                text = " ".join(cur_tokens) if tokens else " ".join(cur_tokens)
                spans.append(f"{cur_label}:{text}")
                cur_label = None
                cur_tokens = []
            continue
        if tg.startswith("B-"):
            if cur_label:
                text = " ".join(cur_tokens) if tokens else " ".join(cur_tokens)
                spans.append(f"{cur_label}:{text}")
            cur_label = tg[2:]
            cur_tokens = [tokens[i]] if tokens else [tg]
        elif tg.startswith("I-") and cur_label:
            cur_tokens.append(tokens[i] if tokens else tg)
        else:
            # unexpected tag pattern
            if cur_label:
                text = " ".join(cur_tokens) if tokens else " ".join(cur_tokens)
                spans.append(f"{cur_label}:{text}")
            cur_label = None
            cur_tokens = []
    if cur_label:
        text = " ".join(cur_tokens) if tokens else " ".join(cur_tokens)
        spans.append(f"{cur_label}:{text}")
    return spans

# ---------------- 中文：LTP (seg,pos,dep,srl,sdp) --------------
def analyze_chinese_file(path_in: str, out_prefix: str, csv_out: bool = True):
    """
    使用 LTP 的 pipeline（若可用）做 cws/pos/dep/srl/sdp
    输出文本和 csv token-level（可打开 Excel）
    """
    try:
        from ltp import LTP
    except Exception as e:
        print("[WARN] ltp 未安装或导入失败：", e)
        return

    ltp = LTP()
    lines = []
    with open(path_in, 'r', encoding='utf-8') as f:
        for ln in f:
            s = ln.strip()
            if s:
                lines.append(s)

    txt_out = out_prefix + ".txt"
    csvpath = out_prefix + ".csv" if csv_out else None
    if csvpath:
        import csv
        cf = open(csvpath, 'w', encoding='utf-8', newline='')
        writer = csv.writer(cf)
        writer.writerow(['sentence_id','token_idx','token','pos','head_idx','head_token','rel'])

    with open(txt_out, 'w', encoding='utf-8') as fout:
        fout.write(f"=== Chinese SRL/SDP analysis by LTP - {os.path.basename(path_in)} ===\n\n")
        for sid, sent in enumerate(lines, start=1):
            fout.write(f"--- Sent {sid} ---\n{sent}\n\n")
            # 调用 pipeline（若 ltp 版本支持）
            words=[]; pos=[]; heads=[]; rels=[]; srl_out=None; sdp_out=None
            try:
                if hasattr(ltp, "pipeline"):
                    out = ltp.pipeline([sent], tasks=["cws","pos","dep","srl","sdp"])
                    # 取各种 key（多版本兼容写法）
                    def get_first(o, k):
                        if hasattr(o, k):
                            val = getattr(o, k)
                        elif isinstance(o, dict) and k in o:
                            val = o[k]
                        else:
                            val = None
                        if isinstance(val, (list,tuple)) and len(val)>0:
                            return val[0]
                        return val
                    words = list(get_first(out, "cws") or [])
                    pos = list(get_first(out, "pos") or [])
                    dep = get_first(out, "dep") or []
                    # dep 常见 [(head, rel), ...]
                    if isinstance(dep, (list,tuple)) and len(dep)>0 and isinstance(dep[0], (list,tuple)):
                        heads = [int(x[0]) for x in dep]
                        rels  = [x[1] for x in dep]
                    else:
                        heads = [0]*len(words); rels=["_"]*len(words)
                    srl_out = get_first(out, "srl")
                    sdp_out = get_first(out, "sdp")
                else:
                    # 旧接口回退：逐步运行 seg/pos/dep，然后尝试 srl/sdp 如有
                    seg = ltp.seg([sent])
                    if isinstance(seg, tuple) and len(seg)>0:
                        words = list(seg[0])
                        # hidden state 支持 pos/dep
                        hidden = seg[1] if len(seg)>1 else None
                        if hidden is not None:
                            pos = list(ltp.pos(hidden)[0])
                            dep = list(ltp.dep(hidden)[0])
                            heads = [int(x[0]) for x in dep]; rels=[x[1] for x in dep]
                    # 试 srl/sdp if available
                    try:
                        srl_out = ltp.srl([sent])
                    except Exception:
                        srl_out = None
                    try:
                        sdp_out = ltp.sdp([sent])
                    except Exception:
                        sdp_out = None
            except Exception as e:
                fout.write(f"[ERROR] LTP pipeline 处理时异常：{e}\n\n")
                continue

            # 对齐长度安全处理
            n = len(words)
            if len(pos) != n: pos = (pos + ["_"]*n)[:n]
            if len(heads) != n: heads = (heads + [0]*n)[:n]
            if len(rels) != n: rels = (rels + ["_"]*n)[:n]

            # 输出 token 表格
            fout.write("Tokens & POS & Dependency (table):\n")
            fout.write(table_text(words, pos, heads, rels) + "\n\n")

            # 输出 SRL（格式尽量通用）
            fout.write("SRL (predicate -> args):\n")
            try:
                fout.write(format_srl(srl_out) + "\n\n")
            except Exception as e:
                fout.write(f"[WARN] 无法格式化 SRL 输出：{e}\n\n")

            # 输出 SDP（若有）
            fout.write("SDP (semantic dependency edges):\n")
            if sdp_out:
                try:
                    # sdp_out 可能是一列表，直接 dump 较稳妥
                    fout.write(repr(sdp_out) + "\n\n")
                except Exception:
                    fout.write("(sdp exists but could not pretty-print)\n\n")
            else:
                fout.write("(no sdp output)\n\n")

            # 写 CSV 行（token-level）
            if csvpath:
                for i in range(n):
                    head_token = words[heads[i]-1] if 1 <= heads[i] <= n else 'ROOT'
                    writer.writerow([sid, i+1, words[i], pos[i], heads[i], head_token, rels[i]])
        if csvpath:
            cf.close()
    print("[LTP] 中文语料处理完毕，输出：", txt_out, csvpath or "(no csv)")



def main():
    # 处理中文语料列表（3 个）
    for path in corpus_paths:
        bn = os.path.basename(path).rsplit(".",1)[0]
        prefix = os.path.join(output_dir, bn + "_ltp_srl")
        analyze_chinese_file(path, prefix, csv_out=True)

if __name__ == "__main__":
    main()

