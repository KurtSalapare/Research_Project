import asyncio
# Ensure crawl4ai is imported correctly as per previous corrections
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
import ollama


# --- 1. Function to crawl the URL and get html content using crawl4ai (Remains the same) ---
async def get_webpage_content_with_crawl4ai(url: str):
    """
    Crawls a given URL using crawl4ai and returns its raw Markdown content.
    """
    browser_conf = BrowserConfig(
        browser_type="chromium", # Use "chromium" for Chrome/Chromium-based browsers
        headless=False,          # Set to True for running without a visible browser window
        verbose=True             # Set to True for more detailed logging from the browser
    )
    run_conf = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    print(f"[Crawl4AI] Starting crawl for: {url}")
    
    print('RESULT UNDER')
    print(" ")
    async with AsyncWebCrawler(config=browser_conf) as crawler:
        result = await crawler.arun(url=url, config=run_conf)
        print(result)

    if result.success:
        print(f"[Crawl4AI] Successfully crawled {url}. Returning markdown content.")
        return result.markdown.raw_markdown
    else:
        print(f"[Crawl4AI] Crawl failed for {url}: {result.error_message}")
        return None
    
def split_string_by_newline(input_string: str) -> list[str]:
    """
    Splits an input string into a list of strings based on newline characters ('\n').

    Args:
        input_string: The string to be split.

    Returns:
        A list of strings, where each element is a segment of the original string
        separated by a newline.
    """
    if not isinstance(input_string, str):
        raise TypeError("Input must be a string.")

    # The .split('\n') method will split the string wherever a newline character occurs.
    # It will return a list of the resulting substrings.
    # If the string is empty, it returns ['']
    # If the string ends with a newline, it will result in an empty string at the end of the list.
    return input_string.split('\n')



