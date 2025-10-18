import os
import re
import pandas as pd
from tqdm import tqdm   # 进度条
import chardet   # 自动检测编码
from utils import load_config

# 读取配置
config = load_config()

# -------------------- 配置 --------------------
input_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), config["txt_test_dir"])
output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), config["txt_drop_dialogue_dir"])
os.makedirs(output_folder, exist_ok=True)

# -------------------- 自动检测文件编码并读取 --------------------
def read_file_auto(file_path: str) -> str:
    """自动检测编码并读取文本"""
    with open(file_path, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result["encoding"] or "utf-8"
    with open(file_path, "r", encoding=encoding, errors="ignore") as f:
        return f.read()

# -------------------- 匹配中英文引号里的内容（包含引号本身） --------------------
quote_pattern = re.compile(r'[“「『"].+?[”」』"]')

summary_list = []

# 获取所有 txt 文件
file_list = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

# 遍历文件夹，带进度条
for filename in tqdm(file_list, desc="Processing texts"):
    file_path = os.path.join(input_folder, filename)
    text = read_file_auto(file_path)   # 自动识别编码

    # 删除台词（包括引号）
    cleaned_text = quote_pattern.sub("", text)

    # 保存为新的 txt 文件
    output_file = os.path.join(output_folder, filename)  # 保持原文件名
    with open(output_file, "w", encoding="utf-8") as f:  # 输出统一用 UTF-8
        f.write(cleaned_text)

    # 汇总信息
    summary_list.append({
        "text_id": filename.replace(".txt", ""),
        "original_length": len(text),
        "cleaned_length": len(cleaned_text)
    })
"""
# 保存 summary.csv（汇总信息）
summary_df = pd.DataFrame(summary_list)
summary_df.to_csv(os.path.join(output_folder, "summary.csv"),
                  index=False, encoding="utf-8-sig")
"""
print("台词删除完成，已输出为 txt！")
