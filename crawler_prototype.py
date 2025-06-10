import asyncio
# Ensure crawl4ai is imported correctly as per previous corrections
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
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
    run_conf = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            
            # Testing the Pruning Content Filter, Ignore links & ignore images options 
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(
                        threshold=0.6,
                        min_word_threshold=10
                    ),
                options={
                        "ignore_links": True,
                        "ignore_images": True,
                    }
            )
        )

    print(f"[Crawl4AI] Starting crawl for: {url}")
    
    async with AsyncWebCrawler(config=browser_conf) as crawler:
        result = await crawler.arun(url=url, config=run_conf)
    if result.success: # type: ignore
        print(f"[Crawl4AI] Successfully crawled {url}. Returning markdown content.")
        return result.markdown.fit_markdown # type: ignore #Changed from raw markdown to fit for cleaner fetching of only main text
    else:
        print(f"[Crawl4AI] Crawl failed for {url}: {result.error_message}") # type: ignore
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
    
    splitted_str = input_string.split('\n')
    splitted_str = list(filter(lambda i: len(i) > 6, splitted_str)) # Current mediocre way to remove elements in the list of strings that are too
    return splitted_str

async def test_crawler():
    target_url = "https://www.cloudflare.com/learning/security/threats/owasp-top-10/" # "https://hiddenlayer.com/innovation-hub/novel-universal-bypass-for-all-major-llms/" "https://www.bbc.com/news/articles/crk2264nrn2o"

    print(f"Starting web scraping and paragraph extraction for: {target_url}")

    # Step 1: Crawl the website to get its content
    web_content_markdown = await get_webpage_content_with_crawl4ai(target_url)
    
    splitted_web_content = split_string_by_newline(web_content_markdown) # type: ignore
    
    """ For Testing
    print("Length : " + str(len(splitted_web_content)))
    
    print("Testing Splitting : ")
    for split_text in splitted_web_content:
        print(split_text)
        print("\n")
    """
  
if __name__ == "__main__" :
    asyncio.run(test_crawler())


