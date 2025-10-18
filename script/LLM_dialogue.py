import os
import re
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore
from utils import load_config

# -------------------- 配置 --------------------
config = load_config()
input_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), config["csv_cut_dialogue_dir"])
output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), config["csv_prediction_dir"])
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "LLM_dialogue_prediction.csv")  # 具体输出文件名


client = OpenAI(
    api_key=config["api_key"],  # 替换为有效 Moonshot API Key
    base_url="https://api.moonshot.cn/v1",
)

# -------------------- 限速参数 --------------------
MAX_RPM = 200          # 每分钟请求数上限
MAX_THREADS = config["LLM_threads_dialogue"]    # 最大并发数
SLEEP_TIME = 60 / MAX_RPM  # 每次请求间隔
MAX_RETRIES = 5        # 每个请求最大重试次数

lock = Lock()          # 写 CSV 时加锁
rate_semaphore = Semaphore(MAX_THREADS)  # 控制并发线程数

empty_files = []

# -------------------- 获取文件列表 --------------------
file_list = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

# -------------------- 保持数字排序逻辑 --------------------
def get_leading_number(fname):
    # 提取文件名开头连续数字
    match = re.match(r'(\d+)', fname)
    return int(match.group(1)) if match else float('inf')

file_list.sort(key=get_leading_number)

# -------------------- 删除已存在的输出文件 --------------------
if os.path.exists(output_file):
    os.remove(output_file)

# -------------------- 处理单个 block 的函数 --------------------
def process_block(filename, block_id, block_df):
    dialogues = block_df["dialogue"].tolist()

    prompt_text = f"""以下是 {len(dialogues)} 条台词，请统计其中体现“百合氛围”（暧昧、亲密、暗恋、浪漫等）的数量，只输出数字，不要额外文字：\n"""
    for i, line in enumerate(dialogues, start=1):
        prompt_text += f"{i}. {line}\n"

    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            start_time = time.time()
            with rate_semaphore:  # 控制并发
                completion = client.chat.completions.create(
                    model="kimi-k2-0905-preview",
                    messages=[
                        {"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，擅长中文和英文对话。"},
                        {"role": "user", "content": prompt_text}
                    ],
                    temperature=0.0,
                )
            text = completion.choices[0].message.content
            match = re.search(r'\d+', text)
            count = int(match.group()) if match else 0
            elapsed = time.time() - start_time
            break
        except Exception as e:
            if 'rate_limit' in str(e).lower() or '429' in str(e):
                wait = 1 + random.random()
                tqdm.write(f"Block {block_id} 限速，等待 {wait:.1f}s 重试...")
                time.sleep(wait)
                retry_count += 1
            else:
                tqdm.write(f"Block {block_id} 处理失败: {e}")
                count = 0
                elapsed = 0
                break

    # 控制全局限速
    time.sleep(SLEEP_TIME)
    return block_id, count, elapsed

# -------------------- 遍历文本 --------------------
with tqdm(total=len(file_list), desc="Processing all texts") as pbar_file:
    for filename in file_list:
        file_path = os.path.join(input_folder, filename)

        # 跳过空文件
        if os.path.getsize(file_path) == 0:
            empty_files.append(filename)
            tqdm.write(f"跳过空文件: {filename}")
            pbar_file.update(1)
            continue

        try:
            df = pd.read_csv(file_path)
            if df.empty:
                empty_files.append(filename)
                tqdm.write(f"跳过空内容文件: {filename}")
                pbar_file.update(1)
                continue
        except pd.errors.EmptyDataError:
            empty_files.append(filename)
            tqdm.write(f" 跳过无法读取的空文件: {filename}")
            pbar_file.update(1)
            continue

        yuri_count = 0
        total_lines = len(df)
        futures = []

        blocks = list(df.groupby("block_id"))

        # -------------------- 多线程处理 blocks --------------------
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            with tqdm(total=len(blocks), desc=f"Processing blocks in {filename}", leave=False) as pbar_block:
                for block_id, block_df in blocks:
                    futures.append(executor.submit(process_block, filename, block_id, block_df))

                for future in as_completed(futures):
                    block_id, count, elapsed = future.result()
                    yuri_count += count
                    tqdm.write(f"Text {filename}, Block {block_id}: {count} 百合台词, 耗时 {elapsed:.2f}s")
                    pbar_block.update(1)

        yuri_concentration = yuri_count / total_lines if total_lines > 0 else 0

        # -------------------- 写入 CSV --------------------
        row = pd.DataFrame([{
            "text_id": filename.replace(".csv", ""),
            "total_lines": total_lines,
            "yuri_lines": yuri_count,
            "yuri_concentration": round(yuri_concentration, 3)
        }])

        with lock:
            if not os.path.exists(output_file):
                row.to_csv(output_file, index=False, encoding="utf-8-sig")
            else:
                row.to_csv(output_file, index=False, encoding="utf-8-sig", mode='a', header=False)

        pbar_file.update(1)

print(f"百合浓度计算完成！共跳过 {len(empty_files)} 个空文件")
