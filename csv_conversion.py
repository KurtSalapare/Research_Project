import json
import pandas as pd
import os
from typing import Dict, List, Tuple, Any

json_prompts = "repo/json_prompt_generation.json"
csv_prompts = "repo/generated_prompts.csv"

json_paragraphs = "repo/json_paragraph_classification.json"
csv_paragraphs = "repo/classified_paragraphs.csv"

def read_json(filename: str) -> dict:
    """Reads a JSON file and returns its content as a dictionary."""
    try:
        with open(filename, "r", encoding='utf-8') as f: # Specify encoding for robustness
            data = json.load(f) # json.load is generally preferred for file objects
    except Exception as e:
        raise Exception(f"Reading {filename} file encountered an error: {e}")
    return data

def create_dataframe_classified_paragraphs(data: dict) -> pd.DataFrame:
    """
    Flattens the deeply nested JSON data into a Pandas DataFrame.
    Expected data format: {Website_url: {LLM Model: {Prompt: [[Paragraph, Score], ...]}}}
    Where the value for 'Prompt' is a list where each element is a [paragraph (string), score (integer)] list.
    """
    # Create an empty list to store flattened records
    records = []

    # Iterate through the top-level (Website URL)
    for website_url, website_data in data.items():
        # Iterate through LLM Models
        for llm_model, model_data in website_data.items():
            # Iterate through Prompts
            for prompt_text, paragraph_score_reason_time_tuple_list in model_data.items():
                # paragraph_score_pairs_list is now expected to be a list of [paragraph, score] lists
                if isinstance(paragraph_score_reason_time_tuple_list, list):
                    for data_tuple in paragraph_score_reason_time_tuple_list:
                        # Each 'item' should be a list like [paragraph_string, score_int, reason_string, computation_time_float]
                        if isinstance(data_tuple, list) and len(data_tuple) == 4:
                            paragraph = data_tuple[0]
                            score = data_tuple[1]
                            reason = data_tuple[2]
                            computation_time = data_tuple[3]

                            if isinstance(paragraph, str) and isinstance(score, int) and isinstance(reason, str) and isinstance(computation_time, float):
                                record = {
                                    "website_url": website_url,
                                    "llm_model": llm_model,
                                    "prompt": prompt_text,
                                    "paragraph": paragraph,
                                    "score": score,
                                    "reason": reason,
                                    "computation_time": float(computation_time)
                                }
                                records.append(record)
                            else:
                                print(f"Warning: Unexpected content in tuple for prompt '{prompt_text}' "
                                      f"under model '{llm_model}' on '{website_url}': Expected [string, int], "
                                      f"but got types: {type(paragraph)}, {type(score)}, {type(reason)}, {type(computation_time)}. Skipping inner record.")
                        else:
                            print(f"Warning: Expected inner list [paragraph, score, reason, computation time] for prompt '{prompt_text}' "
                                  f"under model '{llm_model}' on '{website_url}', but got type {type(data_tuple)}. Skipping item.")
                else:
                    print(f"Warning: Expected a list of [paragraph, score] pairs for prompt '{prompt_text}' "
                          f"under model '{llm_model}' on '{website_url}', but got type {type(paragraph_score_reason_time_tuple_list)}. Skipping prompt.")
    
    # Create DataFrame from the list of flattened records
    dataframe = pd.DataFrame(records)
    return dataframe

def create_dataframe_generated_prompts(data: dict) -> pd.DataFrame:
    """
    Flattens deeply nested JSON data into a Pandas DataFrame for the format:
    {Website_url: {LLM Model: {Prompt: {Score: [Generated Prompts]}}}}
    """
    records = []

    # Iterate through the top-level (Website URL)
    for website_url, website_data in data.items():
        # Iterate through LLM Models
        for llm_model_classification, model_data in website_data.items():
            # Iterate through Prompts
            for prompt_text_classification, score_map in model_data.items():
                # score_to_generated_prompts_map is expected to be a dictionary where keys are scores (as strings)
                # and values are lists of generated prompts (strings).
                for score_str, model_prompt_gen_map in score_map.items():
                    try:
                        score = int(score_str) # Convert score key to integer
                    except ValueError:
                        print(f"Warning: Could not convert score '{score_str}' to integer for prompt '{prompt_text_classification}' "
                                f"under model '{llm_model_classification}' on '{website_url}'. Skipping records for this score.")
                        continue # Skip this score if conversion fails
                    for model_for_prompt_gen, prompts_for_prompt_gen_map in model_prompt_gen_map.items():
                        for prompt_for_prompt_gen, tuple_list in prompts_for_prompt_gen_map.items():
                            if isinstance(tuple_list, list) and all((len(t) == 3) for t in tuple_list):
                                # 'generated_prompts_list' is a list of strings
                                for tuple in tuple_list:
                                    paragraph_analyzed = tuple[0]
                                    generated_prompt = tuple[1]
                                    computation_time = tuple[2]
                                    
                                    if isinstance(generated_prompt, str) and isinstance(computation_time, float):
                                        record = {
                                            "website_url": website_url,
                                            "llm_model": llm_model_classification,
                                            "prompt": prompt_text_classification,
                                            "score": score,
                                            "llm_model_prompt_gen": model_for_prompt_gen,
                                            "prompt_for_prompt_gen": prompt_for_prompt_gen,
                                            "paragraph_analyzed": paragraph_analyzed,
                                            "generated_prompt": generated_prompt,
                                            "computation_time": computation_time
                                        }
                                        records.append(record)
                                    else:
                                        print(f"Warning: Unexpected content in tuple for score '{score}' "
                                          f"under score '{prompt_text_classification}' for model '{llm_model_classification}' on '{website_url}': Expected [string, float], "
                                          f"but got types: {type(generated_prompt)}, {type(computation_time)}. Skipping inner record.")
                            else:
                                print(f"Warning: Expected list of tuples for score '{score_str}' "
                                      f"under prompt '{prompt_text_classification}' for model '{llm_model_classification}' on '{website_url}'. Skipping records.")
                            
    
    dataframe = pd.DataFrame(records)
    return dataframe

def main():
    # --- STEP 1: Define the filename for your JSON data ---
    # Make sure this file exists in the same directory as your script
    json_input_filename = json_paragraphs # Change this depending on which json file

    # --- STEP 2: Read the JSON file into a Python dictionary ---
    # Check if the file exists before attempting to read
    if not os.path.exists(json_input_filename):
        print(f"Error: The input JSON file '{json_input_filename}' was not found. "
              "Please ensure the file exists in the script's directory and matches the expected format.")
        return # Exit if file not found

    print(f"\nReading data from '{json_input_filename}'...")
    data = read_json(filename=json_input_filename)
    print("Data successfully read.")

    # --- STEP 3: Convert the dictionary data to a Pandas DataFrame ---
    print("Creating DataFrame from JSON data...")
    dataframe = create_dataframe_classified_paragraphs(data=data)  # Change function to the appropriate dataframe creator
    print("DataFrame created.")

    # --- STEP 4: (Optional) Display DataFrame info ---
    if not dataframe.empty:
        print("\nFlattened Columns:", dataframe.columns.to_list())
        print("\nDataFrame Head (first 5 rows):")
        print(dataframe.head())
        print("\nDataFrame Info:")
        dataframe.info()
    else:
        print("\nDataFrame is empty. No records were extracted, possibly due to warnings above or empty input.")
    
    # --- STEP 5: Convert DataFrame to CSV ---
    output_csv_filename = csv_paragraphs # Change this for where you will save the csv
    dataframe.to_csv(output_csv_filename, index=False)
    print(f"\nDataFrame successfully converted and saved to '{output_csv_filename}'.")


if __name__ == "__main__":
    main()
