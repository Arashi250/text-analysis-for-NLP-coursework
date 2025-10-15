import os
import pandas as pd
from collections import Counter

input_dir = "./seg_results"
output_path = "./seg_summary.csv"

# 统计函数
def evaluate_segmentation(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # 分割成 "词/词性" 对
    pairs = text.split()
    words, poses = [], []

    for pair in pairs:
        if "/" in pair:
            w, p = pair.rsplit("/", 1)
            if w:
                words.append(w)
                poses.append(p)

    if not words:
        return None

    word_counts = Counter(words)
    pos_counts = Counter(poses)

    total_words = len(words)
    unique_words = len(word_counts)
    avg_word_len = sum(len(w) for w in words) / total_words
    unique_pos = len(pos_counts)

    top_words = ", ".join([f"{w}({c})" for w, c in word_counts.most_common(10)])

    return {
        "文件名": os.path.basename(file_path),
        "总词数": total_words,
        "唯一词数": unique_words,
        "平均词长": round(avg_word_len, 2),
        "词性种类数": unique_pos,
        "高频前10词": top_words
    }


results = []

for fname in os.listdir(input_dir):
    if fname.endswith(".txt"):
        fpath = os.path.join(input_dir, fname)
        print(f"正在分析：{fname}")
        res = evaluate_segmentation(fpath)
        if res:
            results.append(res)

df = pd.DataFrame(results)
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("\n定量评价完成！结果已保存为：", output_path)
