# 安装依赖：pip install jieba thulac snownlp pandas

import os
import re
import string
import jieba.posseg as pseg
import thulac
from snownlp import SnowNLP
import pandas as pd

# 语料路径
corpus_paths = [
    r"resourse\\Chinese news.txt",   # 新闻语料
    r"resourse\\Chinese novel.txt",  # 小说语料
    r"resourse\\Chinese poem.txt"    # 诗歌语料
]

# 输出目录
output_dir = "./seg_results"
os.makedirs(output_dir, exist_ok=True)

# 输出目录
output_dir = "./seg_results"
os.makedirs(output_dir, exist_ok=True)

# 初始化 THULAC
thu = thulac.thulac(seg_only=False)  # seg_only=False 表示同时进行词性标注

# 去除标点符号
CHINESE_PUNCTUATION = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～、。《》〈〉「」『』【】〔〕（）——…￥·"
ALL_PUNCT = set(string.punctuation + CHINESE_PUNCTUATION)

def clean_text(text):
    cleaned = []
    for ch in text:
        if ch not in ALL_PUNCT and not ch.isspace():
            cleaned.append(ch)
    return "".join(cleaned)

# 三种分词函数
def segment_jieba(text):
    return [(w.word, w.flag) for w in pseg.cut(text)]

def segment_thulac(text):
    return thu.cut(text)

def segment_snownlp(text):
    s = SnowNLP(text)
    words, tags = s.words, s.tags
    return list(zip(words, tags))


tools = {
    "jieba": segment_jieba,
    "thulac": segment_thulac,
    "snownlp": segment_snownlp
}

for corpus_path in corpus_paths:
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    corpus_name = os.path.splitext(os.path.basename(corpus_path))[0]

    # 清理文本中的标点符号
    cleaned_text = clean_text(text)

    for tool_name, segment_func in tools.items():
        print(f"正在处理 {corpus_name} 使用 {tool_name} 分词...")

        try:
            result = segment_func(cleaned_text)
        except Exception as e:
            print(f"{tool_name} 在 {corpus_name} 上出错：{e}")
            continue

        # 过滤空字符串和标点
        filtered = [(w, t) for w, t in result if w.strip() and not re.match(r"^\W+$", w)]

        # 转换为“词/词性”的格式
        output_text = " ".join([f"{w}/{t}" for w, t in filtered])

        out_path = os.path.join(output_dir, f"{corpus_name}_{tool_name}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output_text)

        print(f"→ 结果已保存到 {out_path}")

print("\n所有语料处理完成！请查看 ./seg_results 文件夹。")
