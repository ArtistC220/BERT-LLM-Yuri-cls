import chardet
import requests, re, os, time
from bs4 import BeautifulSoup


# ========= 配置 =========
PHPSESSID = '51354d8631d3235711905030b6e42f42'  
UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36')
LIST_URL = 'https://www.wenku8.net/modules/article/tags.php?t=%B0%D9%BA%CF&page={}'
os.makedirs('txt', exist_ok=True)
os.makedirs('txt2', exist_ok=True) 

# ========= 会话 =========
s = requests.Session()
s.headers.update({'User-Agent': UA})
s.cookies.set('PHPSESSID', PHPSESSID)

# ========= 工具函数 =========
def get_packtxt_links(aid):
    packshow_url = f'https://www.wenku8.net/modules/article/packshow.php?id={aid}&type=txt'
    resp = s.get(packshow_url)
    resp.encoding = 'gbk'
    soup = BeautifulSoup(resp.text, 'html.parser')

    links = []
    for a in soup.select('a[href*="packtxt.php"]'):
        href = a['href']
        if href.startswith('/'):
            href = 'https://dl.wenku8.com' + href
        vid = re.search(r'vid=(\d+)', href).group(1)
        links.append((vid, href))

    # 按 vid 去重，保留第一次出现的顺序
    seen = dict()
    for vid, href in links:
        seen.setdefault(vid, href)     # 只保留第一次
    return [(v, u) for v, u in seen.items()], packshow_url



def download_txt(aid, vid, url, referer):
    save_name = f'{aid}_{vid}.txt'
    save_path = f'txt2/{save_name}'

    if os.path.exists(save_path):
        print(f'    {save_name} 已存在，跳过')
        return True

    r = s.get(url, headers={'Referer': referer}, stream=True)
    raw = r.content
    if not raw:
        print(f'    {aid}_{vid} 下载为空')
        return False

    # 先探测
    guess = chardet.detect(raw) or {}
    enc = guess.get('encoding') or 'gbk'

    # 按优先级试
    for try_enc in (enc, 'gbk', 'utf-16', 'utf-8-sig'):
        try:
            text = raw.decode(try_enc)
            # 真正中文编码成功，跳出
            break
        except Exception:
            continue
    else:
        # 全部失败，放弃
        print(f'    {save_name} 无法正确解码，已跳过')
        return False

    # 落盘
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'    已保存 {save_name}  (编码={try_enc})')
    return True

# ========= 主流程 =========

for page in range(2, 11):
    resp = s.get(LIST_URL.format(page))
    resp.encoding = 'gbk'
    soup = BeautifulSoup(resp.text, 'html.parser')

    links = soup.select('a[href*="book"]:not([href*="m."])') or \
            soup.select('a[href*="modules/article/book.php"]')

    # 按 aid 去重
    seen_aid, dedup_links = set(), []
    for a in links:
        m = re.search(r'(?:book/|id=)(\d+)', a['href'])
        if not m:
            continue
        aid = m.group(1)
        if aid not in seen_aid:          # 只保留第一次出现的 aid
            seen_aid.add(aid)
            dedup_links.append(a)

    print(f'第 {page} 页去重后 {len(dedup_links)} 本书')

    for a in dedup_links:                # 用去重后的列表
        aid = re.search(r'(?:book/|id=)(\d+)', a['href']).group(1)
        print(f'  开始处理 aid={aid}')
        txt_urls, packshow_url = get_packtxt_links(aid)
        if not txt_urls:
            print(f'    aid={aid} 无 TXT 分卷')
            continue
        for vid, url in txt_urls:
            download_txt(aid, vid, url, referer=packshow_url)
            time.sleep(1)
        time.sleep(2)