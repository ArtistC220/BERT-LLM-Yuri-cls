#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import requests, csv, re, time, random
from tqdm import tqdm

PHPSESSID = 'f52f6a0486fca52af4d1482362bb802d'
LOCAL_CSV = r'train\csv\result\volume_weighted.csv'
OUT_CSV   = r'train\csv\result\filename_realvolume.csv'

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/122.0 Safari/537.36',
    'Cookie': f'PHPSESSID={PHPSESSID}'
})

need_set = set(pd.read_csv(LOCAL_CSV, usecols=['filename'])['filename'])

# ① 只抓“第X卷”原文，不做任何数字转换
vol_re = re.compile(r'第([一二三四五六七八九十\d\.]+)卷.*?vid=(\d+)', re.S)

result = []
bookids = {fn.split('_')[0] for fn in need_set}

for bookid in tqdm(bookids, desc='抓取书籍'):
    url = f'https://www.wenku8.net/modules/article/packshow.php?id={bookid}&type=txt'
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding
    except Exception as e:
        print(f'[WARN] 获取 {bookid} 失败：{e}')
        continue

    # ② 直接保存中文卷名
    for vol_name, vid in vol_re.findall(resp.text):
        filename = f'{bookid}_{vid}'
        if filename in need_set:
            result.append([filename, f'第{vol_name}卷'])

    time.sleep(random.uniform(1, 2))

# 写文件
with open(OUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
    csv.writer(f).writerows([['filename', 'realvolume']] + result)

print(f'完成！共 {len(result)} 条映射 -> {OUT_CSV}')