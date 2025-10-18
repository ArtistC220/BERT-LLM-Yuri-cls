import os
import regex  # 支持嵌套括号
import re
import pandas as pd
from tqdm import tqdm
from utils import load_config
import chardet   #  自动检测编码

config = load_config()

# -------------------- 配置 --------------------
input_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), config["txt_drop_dialogue_dir"])# 原始 txt 文件夹
output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), config["csv_cut_verb_dir"])    # 输出 CSV 文件夹
verb_dict_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), config["verbword_file"])      # 动词库文件

os.makedirs(output_folder, exist_ok=True)
block_size = 200

# -------------------- 自动检测文件编码并读取 --------------------
def read_file_auto(file_path: str) -> str:
    """自动检测编码并读取文本"""
    with open(file_path, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result["encoding"] or "utf-8"
    with open(file_path, "r", encoding=encoding, errors="ignore") as f:
        return f.read()

# -------------------- 加载动词库 --------------------
with open(verb_dict_file, "r", encoding="utf-8") as f:
    verbs = set(line.strip() for line in f if line.strip())
print(f"已加载 {len(verbs)} 个动词")

# -------------------- 停用词 --------------------
stopwords = {
    "卷", "章", "序", "目次", "目录", "后记", "前言", "引言", "附录", "附表", "附注",
    "作序", "编者按", "小结", "总结", "简介", "作者", "致谢", "完结",
    "全书", "全卷", "正文", "正文完", "终章", "跋", "后序", "附", "表", "注","作品"
}

# -------------------- 分句规则 --------------------
sentence_split_pattern = re.compile(r"(?<=[。！？；\.\?\!;（）()])")

# -------------------- 过滤函数 --------------------
def is_clean_sentence(sentence: str) -> bool:
    allowed_pattern = re.compile(r"^[\u4e00-\u9fffA-Za-z0-9\s，。、！？；：,.!?;“”‘’（）()\-\—…★◆◎※]+$")
    return bool(allowed_pattern.match(sentence))

def remove_nested_brackets(sentence: str) -> str:
    """删除嵌套括号内容"""
    pattern = regex.compile(r"\((?:[^()]+|(?R))*\)|（(?:[^（）]+|(?R))*）")
    return pattern.sub("", sentence)

def remove_remaining_brackets(sentence: str) -> str:
    """安全删除剩余括号（循环）"""
    pattern = re.compile(r"[（）()]")
    while pattern.search(sentence):
        sentence = pattern.sub("", sentence)
    return sentence

# -------------------- 处理文件 --------------------
file_list = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

for filename in tqdm(file_list, desc="Processing files"):
    file_path = os.path.join(input_folder, filename)
    text = read_file_auto(file_path)   # 自动识别编码

    sentences = [s.strip() for s in sentence_split_pattern.split(text) if s.strip()]
    rows = []

    for i, sentence in enumerate(sentences, start=1):
        # 删除括号（嵌套版）
        sentence = remove_nested_brackets(sentence)
        # 再安全删除残留括号
        sentence = remove_remaining_brackets(sentence)
        # 清理掉 5 个以上连续空格
        sentence = re.sub(r"\s{5,}", "", sentence)
        # 停用词过滤
        if any(sw in sentence for sw in stopwords):
            continue
        # 长度过滤
        if len(sentence) < 15 or len(sentence) > 45:
            continue
        # 特殊字符过滤
        if not is_clean_sentence(sentence):
            continue
        # 判断动词库
        if any(verb in sentence for verb in verbs):
            block_id = (i - 1) // block_size + 1
            rows.append({
                "text_id": filename.replace(".txt", ""),
                "block_id": block_id,
                "line_id": i,
                "sentence": sentence
            })

    output_file = os.path.join(output_folder, filename.replace(".txt", ".csv"))
    df = pd.DataFrame(rows, columns=["text_id", "block_id", "line_id", "sentence"])
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("处理完成，每个文件生成一个 CSV")
