# =============== clean_txt.py ===============
import glob, os, re, tqdm, codecs
from utils import load_config   # 统一配置入口

#1. 读 config
cfg = load_config()

#  2. 解析路径
BASE_DIR = os.path.dirname(os.path.dirname(__file__))          # 项目根目录
SRC_DIR  = os.path.join(BASE_DIR, cfg["txt_test_dir"])        # 原始测试 txt
OUT_DIR  = os.path.join(BASE_DIR, cfg["txt_test_cleaned_dir"]) # 清洗后输出
os.makedirs(OUT_DIR, exist_ok=True)

#  3. 清洗规则 
banner = cfg.get("clean_banner",
                 r'★☆★☆★☆轻小说文库\(Www\.WenKu8\.Com\)\☆★☆★☆★')
keys   = cfg.get("clean_keys", [
    r'台版\s*转自', r'图源', r'录入', r'校对', r'修图', r'美工', r'澄空学园',
    r'轻之国度', r'扫图', r'台版', r'转自'
])
pat = re.compile(
    rf'^\s*{banner}[\s\r\n]*'
    rf'(?:\s*(?:{"|".join(keys)})[\s\S]*?)*'
    rf'(?:\s*\n)+',
    flags=re.I | re.M
)

def clean_text(text: str) -> str:
    return pat.sub('', text).lstrip()

def read_text(path):
    for enc in ('utf-8', 'gbk'):
        try:
            with open(path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f'无法解码文件：{path}')

#4. 批量清洗
for fp in tqdm.tqdm(glob.glob(os.path.join(SRC_DIR, '*.txt'))):
    txt = read_text(fp)
    cleaned = clean_text(txt)
    out_path = os.path.join(OUT_DIR, os.path.basename(fp))
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)

print('批量清洗完成 →', OUT_DIR)