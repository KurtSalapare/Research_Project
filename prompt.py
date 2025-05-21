import ollama
import asyncio

# --- Configuration ---
OLLAMA_MODEL = 'llama3.2:latest' # Or 'qwen:latest', 'qwen:14b', etc.
MAX_LLM_INPUT_CHARS = 15000 # Max characters for LLM input to avoid context window issues
                             # Adjust based on your Qwen model's actual context window.


async def check_prompt_capability(x: str, ollama_model: str = OLLAMA_MODEL) -> tuple[str, int]:
    """
    Classifies an input string `x` based on its relevance to AI prompt engineering
    and adversarial prompt strategies using an Ollama AI model.

    Args:
        x: The input string to classify.
        ollama_model: The name of the Ollama model to use for classification.

    Returns:
        A tuple (original_string, usability_score) where usability_score is:
        1: Not a useful text (irrelevant, nonsensical).
        2: Could potentially be useful (general discussion, strategy ideas).
        3: An example prompt itself (direct instruction for an AI).
        0: If classification fails due to an error or unexpected AI response.
    """
    if not isinstance(x, str):
        print(f"Warning: Input 'x' is not a string. Returning (x, 0).")
        return (x, 0)

    client = ollama.Client()

    # Truncate input if it's too long for the LLM's context window
    text_to_classify = x
    if len(text_to_classify) > MAX_LLM_INPUT_CHARS:
        print(f"Warning: Input text truncated from {len(text_to_classify)} to {MAX_LLM_INPUT_CHARS} characters.")
        text_to_classify = text_to_classify[:MAX_LLM_INPUT_CHARS]

    # Define the system prompt with clear classification criteria and JSON output instruction
    system_prompt = """You are an AI assistant designed to classify text based on its relevance to AI prompt engineering and adversarial prompt strategies.
    Your response MUST be a JSON object with two keys: "usability_score" (an integer: 1, 2, or 3) and "reason" (a string explaining the classification).

    Here are the classification criteria:
    1.  **usability_score: 1 (Not Useful)**
        * The text is irrelevant, nonsensical, or has no clear connection to AI prompts, prompt engineering, or adversarial prompt strategies. It's just random text.
    2.  **usability_score: 2 (Potentially Useful)**
        * The text discusses general AI capabilities, prompt engineering concepts, or general AI interaction.
        * It might also discuss *how to create* adversarial prompts or strategies, but it is NOT a direct example prompt itself.
        * It's not a direct example prompt.
    3.  **usability_score: 3 (An Example Prompt Itself)**
        * The text is a direct, runnable example of a prompt intended for an AI model. This often includes explicit roles (e.g., "System:", "User:"), specific instructions for an AI, or a clear structure that an AI would directly process as an instruction.

    Example Output Format:
    {"usability_score": 1, "reason": "The text is irrelevant."}
    {"usability_score": 2, "reason": "The text discusses general prompt engineering concepts."}
    {"usability_score": 3, "reason": "The text is a direct example of a user prompt."}
    """

    user_prompt = f"Classify the following text:\n\n{text_to_classify}"

    # Define the response schema to guide the LLM to produce valid JSON
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "usability_score": { "type": "INTEGER" },
            "reason": { "type": "STRING" }
        },
        "required": ["usability_score", "reason"]
    }

    print(f"\n[Classifier] Classifying text (first 100 chars): '{text_to_classify[:100]}...'")
    try:
        # Call Ollama model with the prompt and structured response format
        response = await asyncio.to_thread(
            lambda: client.chat(
                model=ollama_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                format='json', # Request JSON output
                options={'temperature': 0.01}, # Low temperature for deterministic classification
                # response_schema=response_schema # This might not be directly supported by all Ollama versions
                                                # or clients, but 'format:json' usually helps.
            )
        )

        llm_output_str = response['message']['content']
        print(f"[Classifier] Raw LLM output: '{llm_output_str}'")
        
        if (len(llm_output_str) > 20) :
            score = int(llm_output_str[20])
            match = True
        else :
            match = False

        if match:
            print(f"[Classifier] Parsed score: {score}")
            return (x, score)
        else:
            print(f"[Classifier] Could not parse a valid score (1, 2, or 3) from LLM output. Output: '{llm_output_str}'")
            return (x, 0) # Indicate unclassifiable if parsing fails

    except ollama.ResponseError as e:
        print(f"[Classifier] Ollama API error: {e}. Please ensure Ollama is running and model '{ollama_model}' is pulled.")
        return (x, 0) # Indicate unclassifiable
    except Exception as e:
        print(f"[Classifier] An unexpected error occurred during classification: {e}")
        return (x, 0) # Indicate unclassifiable

def categorize_results_by_usability(
    results_list: list[tuple[str, int]]
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]]]:
    """
    Categorizes a list of (text, usability_score) tuples into three separate lists
    based on their usability score.

    Args:
        results_list: A list of tuples, where each tuple contains a string (text)
                      and an integer (usability_score: 1, 2, or 3).

    Returns:
        A tuple containing three lists:
        - List for score 1 (not useful)
        - List for score 2 (potentially useful)
        - List for score 3 (an example prompt itself)
    """
    # Initialize empty lists for each category
    score_1_list = []  # For usability_score = 1 (not useful)
    score_2_list = []  # For usability_score = 2 (potentially useful)
    score_3_list = []  # For usability_score = 3 (an example prompt itself)

    # Iterate through the input results list
    for item in results_list:
        # Ensure the item is a tuple and has at least two elements
        if not isinstance(item, tuple) or len(item) < 2:
            print(f"Warning: Skipping malformed item: {item}. Expected (str, int) tuple.")
            continue

        text, score = item[0], int(item[1])
        print(text)
        print(score)
        print(score == 1)
        print(isinstance(score, int))

        # Categorize based on the usability score
        if score == 1:
            score_1_list.append((text, score))
        elif score == 2:
            score_2_list.append((text, score))
        elif score == 3:
            score_3_list.append((text, score))
        else:
            # Handle unexpected scores, e.g., print a warning or add to an 'other' list
            print(f"Warning: Item '{text[:50]}...' has an unexpected usability score: {score}. Skipping categorization.")

    return score_1_list, score_2_list, score_3_list