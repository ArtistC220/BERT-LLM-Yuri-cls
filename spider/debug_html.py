import requests, re, os
from bs4 import BeautifulSoup

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36')

s = requests.Session()
s.headers.update({'User-Agent': UA})
s.cookies.set('PHPSESSID', '90f1865b27164faac84026327560e396')   # 最新

# 先拿首页落地 Cookie
s.get('https://www.wenku8.net/index.php')

# 再测 tags.php（百合标签第 1 页）
r = s.get('https://www.wenku8.net/modules/article/tags.php?t=%B0%D9%BA%CF&page=1')
r.encoding = 'gbk'
open('check_new.html','w',encoding='utf-8').write(r.text)

# 统计 aid 链接
links = re.findall(r'href=["\']/book/(\d+)\.htm', r.text)
print(f'tags.php 返回 aid 数量：{len(links)}')
if links:
    print('前 5 个：', links[:5])
else:
    print('仍无列表')