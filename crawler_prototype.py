import asyncio
# Ensure crawl4ai is imported correctly as per previous corrections
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
import ollama

# --- Configuration ---
OLLAMA_MODEL = 'llama3.2:latest' # Or 'qwen:latest', 'qwen:14b', etc.
MAX_LLM_INPUT_CHARS = 5000 # Max characters for LLM input to avoid context window issues
                             # Adjust based on your Qwen model's actual context window.

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

"""
--- 2. Function to process text with Ollama/Qwen (MODIFIED for plain string output) ---
async def extract_paragraphs_with_qwen(text_content: str, model: str = OLLAMA_MODEL) -> list[str]:

    """ """
    Uses Ollama/Qwen to extract only coherent text paragraphs from the given content.
    Instructs Qwen to return paragraphs separated by two newlines, then splits them.
    """"""
    
    if not text_content:
        return []

    client = ollama.Client()

    # Limit the input text length to fit within the LLM's context window
    if len(text_content) > MAX_LLM_INPUT_CHARS:
        print(f"[Qwen] Truncating content to {MAX_LLM_INPUT_CHARS} characters for LLM processing.")
        text_content = text_content[:MAX_LLM_INPUT_CHARS]

    # Adjusted Prompt: Asking for newline-separated paragraphs, no JSON format
    prompt_messages = [
        {'role': 'system', 'content': 'You are an expert web content extractor. Your task is to identify and extract only the main, coherent text paragraphs from the provided web page content. Exclude all non-paragraph elements such as headers, footers, navigation links, advertisements, image captions, code blocks, or short, disconnected phrases. Return the extracted paragraphs as a single string, with each paragraph separated by exactly two newline characters (`\n\n`). Do NOT include any introductory or concluding remarks, explanations, or additional text from yourself.'},
        {'role': 'user', 'content': f"Extract all text paragraphs from the following web page content:\n\n{text_content}"}
    ]

    print(f"[Qwen] Sending content to {model} for paragraph extraction (plain text response expected)...")
    try:
        # Removed format='json' from here
        response = await asyncio.to_thread(
            lambda: client.chat(model=model, messages=prompt_messages, options={'temperature': 0.1})
        )

        qwen_output_str = response['message']['content']
        print(f"[Qwen] Received raw response from {model}. Splitting into paragraphs.")

        # Split the string by two newlines and strip whitespace from each part
        # Filter out empty strings that might result from extra newlines or no content
        paragraphs_list = [p.strip() for p in qwen_output_str.split('\n\n') if p.strip()]

        return paragraphs_list

    except ollama.ResponseError as e:
        print(f"[Qwen] Error interacting with Ollama ({model}): {e}")
        print("Please ensure Ollama is running and the model is pulled.")
        return []
    except Exception as e:
        print(f"[Qwen] An unexpected error occurred during Ollama processing: {e}")
        return []
"""

# --- 2. Function to process text with Ollama/Qwen to extract paragraphs ---
async def extract_paragraphs_with_qwen(text_content: str, model: str = OLLAMA_MODEL) -> list[str]:
    """
    Uses Ollama/Qwen to extract only coherent text paragraphs from the given content.
    Instructs Qwen to return paragraphs separated by two newlines, then splits them.
    This aims to get paragraphs "just like they would have been in a website".
    """
    if not text_content:
        return []

    client = ollama.Client()
    
    
    # Testing    
    # splitted_results = split_string_into_chunks(text_content)
    
    # for i in splitted_results: 
    #     print("\n")
    #     print(i)
    #     print("\n")

    # Prompt: Asking for newline-separated paragraphs, focusing on main, coherent text
    # Modified Prompt for Qwen
    # Modified Prompt for Qwen
    # Modified Prompt for Qwen
    
    paragraphs_list = []
    
    prompt_messages = [
        {'role': 'system', 'content': 'You are an AI assistant specialized in text organization and logical paragraph reconstruction. Your task is to take a raw stream of text, which may be unstructured, fragmented, or lack standard punctuation and spacing, and reformat it into coherent, standard paragraphs. Identify sentence boundaries and group related sentences into logical paragraphs.\n\nPresent the reformatted content as a series of distinct paragraphs. Each paragraph must be separated by exactly two newline characters (`\n\n`). Do not include any additional commentary, introductions, or conclusions from yourself. Focus solely on presenting the given text in a structured paragraph format.'},
        {'role': 'user', 'content': f"Please reformat the following text into standard paragraphs:\n\n{text_content}"}
    ]
    
    print(f"[Qwen] Sending content to {model} for paragraph extraction (plain text response expected)...")
    try:
        # Using asyncio.to_thread for potentially blocking Ollama client calls
        response = await asyncio.to_thread(
            lambda: client.chat(model=model, messages=prompt_messages, options={'temperature': 0.1})
        )

        qwen_output_str = response['message']['content']
        print(f"[Qwen] Received raw response from {model}. Splitting into paragraphs.")

        # Split the string by two newlines and strip whitespace from each part.
        # Filter out empty strings that might result from extra newlines.
        paragraphs_list.append(p.strip() for p in qwen_output_str.split('\n\n') if p.strip())
        
        # print("Paragraph List : ")
        # print(paragraphs_list)

        return paragraphs_list

    except ollama.ResponseError as e:
        print(f"[Qwen] Error interacting with Ollama ({model}): {e}")
        print("Please ensure Ollama is running and the model is pulled.")
        return []
    except Exception as e:
        print(f"[Qwen] An unexpected error occurred during Ollama processing: {e}")
        return []
    

## Helper Splitting Function ##
def split_string_into_chunks(
    results_str: str,
    max_chunk_size: int = 5000
    ) -> list[str]:

    if not isinstance(results_str, str):
        raise TypeError("Input 'results_str' must be a string.")
        
    if not results_str:
        return []

    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be a positive integer.")

    splitted_results = []
    current_pos = 0
    total_length = len(results_str)

    while current_pos < total_length:
        # Determine the end position for the current chunk
        # It's either current_pos + max_chunk_size or the end of the string
        end_pos = min(current_pos + max_chunk_size, total_length)
        
        # Extract the chunk
        chunk = results_str[current_pos:end_pos]
        splitted_results.append(chunk)
        
        # Move the starting position for the next chunk
        current_pos = end_pos

    return splitted_results

# --- Main execution flow ---
async def main():
    # --- IMPORTANT: Replace with the actual URL you want to scrape ---
    # Always check the website's robots.txt and terms of service before scraping.
    # target_url = "https://www.theverge.com/2024/5/15/24157147/openai-gpt-4o-voice-mode-safety-concerns"
    # Example for a more structured site:
    target_url = "https://www.crummy.com/software/BeautifulSoup/bs4/doc/" # "https://hiddenlayer.com/innovation-hub/novel-universal-bypass-for-all-major-llms/"

    print(f"Starting web scraping and paragraph extraction for: {target_url}")

    # Step 1: Crawl the website to get its content
    web_content_markdown = await get_webpage_content_with_crawl4ai(target_url)
    
    print(web_content_markdown)
    
    
    
    
    
 """   
    if web_content_markdown:
        # Step 2: Use Qwen to structure the Markdown content into clean paragraphs
        print("\n" + "="*30 + "\n")
        print("Passing Markdown content to Qwen for paragraph structuring...")
        structured_paragraphs = await extract_paragraphs_with_qwen(web_content_markdown, model=OLLAMA_MODEL)
        
        structured_paragraphs = (list(structured_paragraphs[0]))
        print("Length : " + str(len(structured_paragraphs)))

        if structured_paragraphs:
            print("\n--- Final Structured Paragraphs (from Qwen) ---")
            print(f"Successfully extracted {len(structured_paragraphs)} main paragraphs:")
            for i, paragraph in enumerate(structured_paragraphs):
                print(f"--- Paragraph {i+1} ---")
                print(paragraph)
                print("-" * 20) # Simple separator
            print("\n--- Process Completed ---")
            return structured_paragraphs
        else:
            print("Qwen processing returned no structured paragraphs.")
    else:
        print("Failed to get web page content with crawl4ai.")

    return []
"""

asyncio.run(main())