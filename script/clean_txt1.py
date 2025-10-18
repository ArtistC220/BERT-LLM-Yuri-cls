# clean_txt1.py  
import glob, os, re, tqdm

#这是训练时使用的手动清理脚本

SRC_DIR  = r'train\txt'               # 原始目录
OUT_DIR  = r'train\assets\txt_cleaned' # 清洗后目录
os.makedirs(OUT_DIR, exist_ok=True)

# ① 横幅
banner = r'★☆★☆★☆轻小说文库\(Www\.WenKu8\.com\)\☆★☆★☆★'

# ② 制作组关键词
keys = [
    r'台版\s*转自', r'图源', r'录入', r'校对', r'修图', r'美工', r'澄空学园',
    r'轻之国度', r'扫图',r'台版',r'转自'
]

# ③ 组合成正则：横幅 + 0~N 组声明（连续行）
pat = re.compile(
    rf'^\s*{banner}[\s\r\n]*'
    rf'(?:\s*(?:{"|".join(keys)})[\s\S]*?)*'   # 非贪婪匹配直到正文
    rf'(?:\s*\n)+',                             # 至少一个换行才算结束
    flags=re.I | re.M
)

def clean_text(text: str) -> str:
    # 只删最开头的声明块
    return pat.sub('', text).lstrip()

for fp in tqdm.tqdm(glob.glob(os.path.join(SRC_DIR, '*.txt'))):
    txt = open(fp, encoding='utf-8').read()
    cleaned = clean_text(txt)
    out_path = os.path.join(OUT_DIR, os.path.basename(fp))
    open(out_path, 'w', encoding='utf-8').write(cleaned)



print('批量清洗完成 →', OUT_DIR) #仅清洗开头