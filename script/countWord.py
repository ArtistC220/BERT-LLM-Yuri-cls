
# -*- coding: utf-8 -*-
TXT_FILE = r"C:\Users\19716\OneDrive\Desktop\杂\py\train\txt\3525_146329.txt"
TARGET_WORD = "雪菜"

import re
from pathlib import Path

file_path = Path(TXT_FILE)
if not file_path.is_file():
    print(f"文件不存在：{file_path}")
    input("\n按任意键退出…")
    exit(1)

text = file_path.read_text(encoding="utf-8")
# 关键改动：用汉字边界代替 \b
pattern = rf"(?<![\u4e00-\u9fff]){re.escape(TARGET_WORD)}(?!=[\u4e00-\u9fff])"
count = len(re.findall(pattern, text))

print(f"词语“{TARGET_WORD}”在文件中一共出现 {count} 次")
input("\n统计完成，按任意键退出…")