import os
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import time
import re
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore
from utils import load_config

# -------------------- 配置 --------------------
config = load_config()
input_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), config["csv_cut_verb_dir"])
output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), config["csv_prediction_dir"])
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "LLM_verb_prediction.csv")  # 具体输出文件名

client = OpenAI(
    api_key=config["api_key"],
    base_url="https://api.moonshot.cn/v1",
)

# -------------------- 限速参数 --------------------
MAX_RPM = 200
MAX_THREADS = config["LLM_threads_verb"]
SLEEP_TIME = 60 / MAX_RPM
MAX_RETRIES = 5

lock = Lock()
rate_semaphore = Semaphore(MAX_THREADS)
empty_files = []

# -------------------- 获取文件列表并数字排序 --------------------
def extract_numbers(fname):
    nums = re.findall(r'\d+', fname)
    return [int(n) for n in nums] if nums else [0]

file_list = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
file_list.sort(key=lambda x: extract_numbers(x))  # 安全的数字排序

# 如果 CSV 文件存在，先删除
if os.path.exists(output_file):
    os.remove(output_file)

# -------------------- 处理单个 block 的函数 --------------------
def process_block(filename, block_id, block_df):
    dialogues = block_df["sentence"].tolist()
    
    # 构建带详细判定标准的 prompt
    prompt_text = f"""【判定任务】
对以下 {len(dialogues)} 条动作或心理描写进行重百合（Hard Yuri）二元判定，统计符合标准的数量。

【判定标准 - 满足任一即计1】

**动作描写类：**
1. 直接亲密动作
   - 亲吻、拥抱、抚摸、牵手、咬/舔等明确身体接触
   - 整理对方衣物/头发（带有执着感而非随意帮忙）

2. 独占/介入动作
   - 物理隔离：将对方拉离人群、挡在他人面前、插入两人之间
   - 视线控制：遮挡他人视线、强制对方只看自己、掰过对方下巴/脸
   - 物品占有：紧握对方物品、阻止他人触碰对方

3. 沉重系身体反应
   - 因嫉妒/渴望产生的生理反应：颤抖、窒息感、手指僵硬、无意识地抓握
   - 自我压抑动作：掐自己、咬嘴唇、蜷缩、指甲陷入掌心

4. 观测者式动作（放松新增）
   - 过度关注细节：记住对方今天发绳颜色变化、注意到对方微小伤痕
   - 长时间凝视：盯着对方侧脸、嘴唇、手指发呆
   - 偷拍、偷闻气味、保存对方用过的物品（即使描写平静也计1）

**心理描写类：**
1. 直球情感宣言（内心独白）
   - 明确认知到"这是爱情/欲望/独占欲"（不需要说出来）
   - 内心使用"我的""只属于我"等排他性词汇

2. 独占欲心理
   - 嫉妒：看到对方和他人互动时内心刺痛、想把对方藏起来
   - 恐慌：害怕被遗忘、害怕对方离开、担心自己在对方心中不重要

3. 沉重系思虑（放松标准）
   - 过度解读：对方一个眼神就内心翻涌、反复回想对方一句话
   - 自我厌恶式依赖："我这样很恶心吧，但是..."
   - 存在绑定：自我身份完全依附对方（"我是为了她而存在"）

4. 欲望与幻想（放松新增）
   - 渴望更多接触："想触碰""想占有""想永远在一起"
   - 性张力幻想：对对方身体的细节想象（气味、温度、触感）
   - 平静的执着："想一直这样下去""想把她关起来（即使开玩笑语气）"

【强制排除 - 直接计0】
- 纯粹生理疼痛/恐惧（无情感指向的"头痛""害怕"）
- 客观环境动作（单纯走路、拿东西、坐下等无特殊情感）
- 群体性活动（"大家一起做某事"中的动作）
- 亲情/友情明确界限（"像姐姐一样照顾她"且无暧昧）
- 工作/任务性接触（医生检查、战斗配合等）

【无上下文强制仲裁 - 关键放松点】
无法确定是深刻友情还是爱情时：

1. **含身体执念细节**（盯着看、记住细节、闻气味、保存物品）→ 计1
2. **含"永远/一直/每天"等时间词+内心波动** → 计1（即使动作平静）
3. **含自我压抑/自我厌恶但继续执着** → 计1
4. **仅是"开心/紧张"但无排他/持久特征** → 计0（普通情绪）

【重要放宽原则】
- **温柔但异常**：平静地记住对方三天前穿的衣服颜色 > 激动地挥手告别
- **单方面重量**：即使没有互动，单方面的观测、渴望、执着也算
- **细节即重量**：动作/心理描写中对细节的过度关注（反复整理同一缕头发）体现执着

【绝对约束】
- 只输出最终的统计数字（整数），不要任何其他文字、解释、标点或换行
- 禁止输出分析过程、行号、理由

【待分析文本】：
"""
    for i, line in enumerate(dialogues, start=1):
        prompt_text += f"{i}. {line}\n"

    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            start_time = time.time()
            with rate_semaphore:
                completion = client.chat.completions.create(
                    model="kimi-k2-0905-preview",
                    messages=[
                        {
                            "role": "system", 
                            "content": "你是重百合动作心理计数器。你必须只输出一个阿拉伯数字（0-1000之间），禁止输出任何其他字符（包括文字、标点、换行、空格）。错误示例：'5条' 'result:5' '```5```'。正确示例：5"
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
                tqdm.write(f"Block {block_id} 限速，等待 {wait:.1f}s重试...")
                time.sleep(wait)
                retry_count += 1
            else:
                tqdm.write(f"Block {block_id} 处理失败: {e}")
                count = 0
                elapsed = 0
                break

    time.sleep(SLEEP_TIME)
    return block_id, count, elapsed

# -------------------- 遍历文本 --------------------
with tqdm(total=len(file_list), desc="Processing all texts") as pbar_file:
    for filename in file_list:
        file_path = os.path.join(input_folder, filename)

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
            tqdm.write(f"跳过无法读取的空文件: {filename}")
            pbar_file.update(1)
            continue

        yuri_count = 0
        total_lines = len(df)
        futures = []
        blocks = list(df.groupby("block_id"))

        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            with tqdm(total=len(blocks), desc=f"Processing blocks in {filename}", leave=False) as pbar_block:
                for block_id, block_df in blocks:
                    futures.append(executor.submit(process_block, filename, block_id, block_df))
                for future in as_completed(futures):
                    block_id, count, elapsed = future.result()
                    yuri_count += count
                    tqdm.write(f"Text {filename}, Block {block_id}: {count} 符合百合心理和动作的描写, 耗时 {elapsed:.2f}s")
                    pbar_block.update(1)

        yuri_concentration = yuri_count / total_lines if total_lines > 0 else 0

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
