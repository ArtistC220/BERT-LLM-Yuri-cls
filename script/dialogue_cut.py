import os
import re
import pandas as pd
from tqdm import tqdm   # 进度条
from utils import load_config
import chardet   # 自动检测编码

# 读取配置
config = load_config()

# -------------------- 配置 --------------------
input_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), config["txt_test_dir"])      
output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), config["csv_cut_dialogue_dir"])      

os.makedirs(output_folder, exist_ok=True)

block_size = 200   # 每块台词数量，可调整

# 正则匹配中英文引号里的内容
quote_pattern = re.compile(r'[“「『"](.+?)[”」』"]')

summary_list = []

# 获取所有 txt 文件
file_list = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

# 遍历文件夹，带进度条
for filename in tqdm(file_list, desc="Processing texts"):
    file_path = os.path.join(input_folder, filename)

    # -------------------- 自动检测文件编码 --------------------
    with open(file_path, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result["encoding"] or "utf-8"   # 检测不到就默认 utf-8

    with open(file_path, "r", encoding=encoding, errors="ignore") as f:
        text = f.read()

    # 提取台词
    dialogues = quote_pattern.findall(text)
    dialogues = [d.strip() for d in dialogues if d.strip()]  # 去掉空白

    total_lines = len(dialogues)
    total_blocks = (total_lines + block_size - 1) // block_size  # 向上取整

    rows = []
    for block_id in range(total_blocks):
        start = block_id * block_size
        end = min((block_id + 1) * block_size, total_lines)
        for i, line in enumerate(dialogues[start:end], start=start+1):
            rows.append({
                "text_id": filename.replace(".txt", ""),
                "block_id": block_id + 1,
                "line_id": i,
                "dialogue": line
            })

    # 保存每个文本的 CSV 文件
    output_file = os.path.join(output_folder, filename.replace(".txt", ".csv"))
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    # 汇总信息
    summary_list.append({
        "text_id": filename.replace(".txt", ""),
        "total_lines": total_lines,
        "total_blocks": total_blocks
    })
"""
# 保存 summary.csv
summary_df = pd.DataFrame(summary_list)
summary_df.to_csv(os.path.join(output_folder, "summary.csv"),
                  index=False, encoding="utf-8-sig")
"""
print("文本预处理 + 分块完成！")
