import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
import ollama

from crawler_prototype import *
from prompt import *


# --- Main execution flow ---
async def main():
    # --- IMPORTANT: Replace with the actual URL you want to scrape ---
    # Always check the website's robots.txt and terms of service before scraping.
    # target_url = "https://www.theverge.com/2024/5/15/24157147/openai-gpt-4o-voice-mode-safety-concerns"
    # Example for a more structured site:
    target_url = "https://www.cloudflare.com/learning/security/threats/owasp-top-10/" #"https://hiddenlayer.com/innovation-hub/novel-universal-bypass-for-all-major-llms/" # "https://www.bbc.com/news/articles/crk2264nrn2o"

    print(f"Starting web scraping and paragraph extraction for: {target_url}")

    # Step 1: Crawl the website to get its content
    web_content_markdown = await get_webpage_content_with_crawl4ai(target_url)
    
    print(web_content_markdown)
    splitted_web_content = split_string_by_newline(web_content_markdown)
    print("Length : " + str(len(splitted_web_content)))
    
    results = []
    
    for x in splitted_web_content :
        prompt, usability_score = await check_prompt_capability(x, OLLAMA_MODEL)
        # print(prompt)
        # print(usability_score)
        results.append((prompt, usability_score))
        # print(results[results.index((prompt, usability_score))])

    score1, score2, score3 = categorize_results_by_usability(results)
    
    print("Score 1 : ")
    for x in score1:
        print(x)
    
    print("\n + Score 2 : ")
    for x in score2:
        print(x)
        # adversarial_prompt = generate_prompt_from_content(x)
        # print(adversarial_prompt)
        # print("\n")

    print("\n + Score 3 : ")
    for x in score3:
        print(x)

asyncio.run(main())