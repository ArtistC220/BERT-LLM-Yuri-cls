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

    # 构建带详细判定标准的 prompt
    prompt_text = f"""【判定任务】
对以下 {len(dialogues)} 条台词进行重百合（Hard Yuri）二元判定，统计符合标准的数量。

【判定标准 - 满足任一即计1】

1. 直球爱意（显）
   明确"爱/喜欢/恋/想在一起"等词，且隐含或明示排他性（不需要"疯狂"修饰，平静的独占也算）

2. 病态占有/焦虑（隐或显）
   - 显性：疯狂/嫉妒/想去死/好痛苦/只属于我/不要看别人 等词
   - 隐性：过度关注细节（"你今天换了发绳""你和谁说话了"）、反复确认关系、害怕被遗忘
   
3. 时间维度的永恒绑定（强制排他）
   含"一辈子/永远/一直/到死/世界末日"等时间词，体现"非阶段性而是永恒"的承诺感
   *注：平静的"想永远在一起"比激动的"现在好喜欢"更符合重百合*

4. 身份/存在的依附绑定
   - 自我定义完全依附于对方（"我是你的...""我只会为你..."）
   - 观测者/记录者视角（"我一直在看着你""我记得你的一切"）
   - 单方面的守护/独占宣言（"我会保护你"伴随排他性，非单纯友情）

5. 身体/欲望的明确指向
   渴望触碰/占有/亲吻/闻味道/品尝等身体接触，或明确的性暗示

【强制排除 - 直接计0】
- 群体性指向（"大家""朋友们""我们"）
- 明显玩笑语气（被"哈哈""真是的""开玩笑啦"包裹）
- 粉丝对偶像的单向应援（"推你""应援"）
- 纯粹商业/工作关系（"请多关照""合作愉快"）
- 亲情明确无暧昧（"像妈妈/姐姐一样"且无背德感）

【放宽的仲裁规则 - 关键调整】
当缺乏上下文无法确定是友情还是爱情时，按此优先级：

1. **含极端情绪词**（疯狂/痛苦/窒息/想去死/杀了你/融化）→ 计1
2. **含时间永恒词+对象特定**（永远/一辈子只对你）→ 计1（即使语气平静）
3. **含细节观测/记录**（"我知道你的秘密""你今天的表情"）→ 计1（体现过度关注）
4. **含自我身份绑定**（"我是你的...""我变成你的..."）→ 计1
5. **仅有"喜欢/重要"但无任何排他/永恒/观测特征** → 计0（保守处理普通友情）

【重要放宽原则】
- **平静但执着 > 激动但模糊**：一句平静的"我会永远看着你"比重于激动的"今天好开心"
- **单方面沉重也算**：暗恋、观测者、单方面的身份依附，即使对方未回应，也计入重百合氛围
- **重复即重量**：如果台词体现重复性/习惯性（"今天又...""每天..."），即使词汇温和，也体现情感重量

【绝对约束 - 违反会导致系统故障】
- 只输出最终的统计数字（整数），不要任何其他文字、解释、标点或换行
- 禁止输出"共有X条""结果是：X"等任何形式，只输出数字本身
- 禁止输出分析过程、行号、理由

【待分析台词】：
"""
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
                        {
                            "role": "system", 
                            "content": "你是重百合文本计数器。你必须只输出一个阿拉伯数字（0-1000之间），禁止输出任何其他字符（包括文字、标点、换行、空格）。错误示例：'5条' 'result:5' '```5```'。正确示例：5"
                        },
                        {
                            "role": "user", 
                            "content": prompt_text
                        }
                    ],
                    temperature=0.0,
                )
            
            text = completion.choices[0].message.content.strip()
            
            # 清理可能的 markdown 或空格，只保留数字
            text_clean = re.sub(r'[^\d]', '', text)
            
            if text_clean:
                count = int(text_clean)
            else:
                count = 0
                tqdm.write(f"警告：Block {block_id} 无数字输出，原文：{text[:50]}")
            
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
