# baihe_aid_title_cover_local.py
import os, csv, time, requests, re
from bs4 import BeautifulSoup

# ========= 配置 =========
PHPSESSID = '90f1865b27164faac84026327560e396'
UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36')
LIST_URL = 'https://www.wenku8.net/modules/article/tags.php?t=%B0%D9%BA%CF&page={}'
CSV_FILE = 'baihe_aid_title_cover_local.csv'
IMG_DIR  = 'csv/cover'                # 本地封面保存目录
os.makedirs(IMG_DIR, exist_ok=True)

# ========= 会话 =========
s = requests.Session()
s.headers.update({'User-Agent': UA})
s.cookies.set('PHPSESSID', PHPSESSID)

# ========= 抓列表 + 下载封面 =========
all_rows = []
for page in range(1, 12):
    print(f'>>> 抓取第 {page} 页...')
    resp = s.get(LIST_URL.format(page))
    resp.encoding = 'gbk'
    soup = BeautifulSoup(resp.text, 'html.parser')

    for div in soup.select('div[style*="width:373px"]'):
        a   = div.select_one('a[href*="book"]')
        img = div.select_one('img[src*="image/"]')
        if not (a and img):
            continue
        aid   = re.search(r'book/(\d+)\.htm', a['href']).group(1)
        title = a['title'].strip()
        cover_url = img['src'].strip()

        # 下载封面 → 本地路径
        ext = os.path.splitext(cover_url)[1] or '.jpg'
        local_path = os.path.join(IMG_DIR, f'{aid}{ext}')

        if not os.path.exists(local_path):        # 增量下载
            try:
                r = s.get(cover_url, timeout=10)
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    f.write(r.content)
                print(f'  封面已下载：{local_path}')
            except Exception as e:
                print(f'  下载失败：{cover_url}  {e}')
                local_path = ''                     # 留空方便后续手动补
        else:
            print(f'  封面已存在：{local_path}')

        all_rows.append((aid, title, local_path))
        time.sleep(0.3)     # 礼貌延迟

# ========= 保存 CSV（本地路径） =========
with open(os.path.join('csv', CSV_FILE), 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['aid', 'title', 'local_cover'])
    writer.writerows(all_rows)

print(f'全部完成！共 {len(all_rows)} 条，封面保存在 {IMG_DIR}，CSV 路径：csv/{CSV_FILE}')