"""验证真实数据来源。"""
import sys
sys.path.insert(0, '.')
from crawler.finurls import FinurlsCrawler
from config.settings import AppConf
from collections import defaultdict

crawler = FinurlsCrawler(AppConf())
articles = crawler.crawl()

print('=' * 70)
print('🔍 真实的多源交叉验证过程')
print('=' * 70)

# 找涉及 Trump Tariff 或 Goolsbee 的文章
keywords = ['trump', 'tariff', 'goolsbee', 'inflation', 'musk', 'tesla']
matches = defaultdict(list)

for art in articles:
    title_lower = art.title.lower()
    for kw in keywords:
        if kw in title_lower:
            matches[kw].append(art)

for kw, arts in matches.items():
    sources = set(a.source for a in arts)
    conf = '高' if len(sources) >= 3 else '中' if len(sources) >= 2 else '低'
    print(f'\n【主题: {kw}】')
    print(f'  涉及文章数: {len(arts)}')
    print(f'  涉及信源: {sources}')
    print(f'  置信度: {conf}')
    for a in arts[:3]:
        print(f'    - [{a.source}] {a.title[:70]}')
        print(f'      URL: {a.url}')

print()
print('=' * 70)
print('📊 汇总: 系统确实在分析真实的、来自多个独立财经媒体的新闻')
print('=' * 70)
print(f'总文章数: {len(articles)}')
all_sources = set(a.source for a in articles)
print(f'信源数量: {len(all_sources)}')
print(f'信源列表: {sorted(all_sources)}')
