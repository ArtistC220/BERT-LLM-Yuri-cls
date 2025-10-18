import os
import csv

# 设置路径
label_csv_path = 'train\label\label.csv'
folder_path = r'train\assets\txt_val_cleaned2'  # 替换为你的文件夹路径
output_csv_path = r'train\label\txt_label_count\txt_val_label2.csv'

# 读取 label.csv，建立 book_id -> label 的映射
label_map = {}
with open(label_csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        book_id = row['book_id'].strip()
        label_value = row['label'].strip()
        if label_value in {'0', '1'}:
            label_map[book_id] = int(label_value)

# 扫描文件夹中的 txt 文件
results = []
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        name_part = filename[:-4]  # 去掉 .txt
        parts = name_part.split('_')
        if len(parts) >= 1:
            book_id = parts[0]
            if book_id in label_map:
                results.append((filename, label_map[book_id]))

# 写入结果到 output.csv
with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'label'])
    writer.writerows(results)

# 统计 label 数量
count_1 = sum(1 for _, label in results if label == 1)
count_0 = sum(1 for _, label in results if label == 0)

# 输出统计结果
print(f"label=1 的数量: {count_1}")
print(f"label=0 的数量: {count_0}")