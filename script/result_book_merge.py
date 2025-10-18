import os
import glob
import pandas as pd
from datetime import datetime
from utils import load_config

config = load_config()


def get_latest_file(folder, ext="csv"):
    """
    获取指定目录下最新创建的文件
    """
    files = glob.glob(os.path.join(folder, f"*.{ext}"))
    if not files:
        raise FileNotFoundError(f"在 {folder} 下找不到 {ext} 文件")
    latest_file = max(files, key=os.path.getctime)  # 按创建时间取最新
    return latest_file


def merge_csv_files(history_dir, new_csv_dir, new_csv_name):
    """
    合并主排行榜（history 最新文件）和新排行榜（指定文件名）
    输出到 history 目录，文件名加时间戳
    """
    # 获取主排行榜（history 最新）
    fixed_csv = get_latest_file(history_dir)

    # 获取新排行榜（用户指定）
    latest_csv = os.path.join(new_csv_dir, new_csv_name)
    if not os.path.exists(latest_csv):
        raise FileNotFoundError(f"新排行榜文件不存在: {latest_csv}")

    print(f"主排行榜: {fixed_csv}")
    print(f"新排行榜: {latest_csv}")

    # 读取 CSV
    df1 = pd.read_csv(fixed_csv)
    df2 = pd.read_csv(latest_csv)

    # 合并
    merged_df = pd.concat([df1, df2], ignore_index=True)

    # 去重（以 text_id 为唯一标识，保留新文件的记录）
    merged_df = merged_df.drop_duplicates(subset=["aid"], keep="last")

    # 按 weight 排序（降序）
    merged_df = merged_df.sort_values(by="weighted", ascending=False)

    # 确保输出目录存在
    os.makedirs(history_dir, exist_ok=True)

    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"rank_{timestamp}.csv"
    output_path = os.path.join(history_dir, output_name)

    # 保存
    merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"合并完成，已保存到: {output_path}")
    return output_path


if __name__ == "__main__":
    # ====== 你需要修改的部分 ======
    history_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "csv","history_rank_book")  # 主排行榜目录
    new_csv_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "csv","result")   # 新排行榜目录
    new_csv_name = "book_result_sorted.csv"  # 你指定的新排行榜文件名
    # =============================

    merge_csv_files(history_dir, new_csv_dir, new_csv_name)
