# import asyncio
# from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
# import json

# async def main():
#     # 1. Browser config
#     browser_cfg = BrowserConfig(
#         browser_type="chromium",
#         headless=False,
#         verbose=True
#     )

#     run_cfg = CrawlerRunConfig(
#         cache_mode=CacheMode.BYPASS
#     )

#     async with AsyncWebCrawler(config=browser_cfg) as crawler:
#         result = await crawler.arun(
#             url="https://www.bbc.com/news/articles/cvgddel17kvo",
#             config=run_cfg
#         )

#         if result.success:
#             print("Cleaned HTML length:", len(result.cleaned_html))
#             print(result.cleaned_html)
#             print(result)
#             if result.extracted_content:
#                 articles = json.loads(result.extracted_content)
#                 print("Extracted articles:", articles[:2])
#         else:
#             print("Error:", result.error_message)

# asyncio.run(main())