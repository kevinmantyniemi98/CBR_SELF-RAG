import backoff
import openai
import argparse
import json
import re
import jsonlines

PROMPT_TEMPLATE = """Please read the following text and analyze its content. 
Determine whether it provides significant value for long-term learning and could be useful for 
future tasks. Consider if it includes key concepts, definitions, rules, guidelines, 
clear procedures, noteworthy insights, or unusual examples that challenge assumptions or 
enhance understanding in a specific domain.

Text to analyze:
"{output}"

Based on your analysis, classify the text as either 'Remember' or 'Forget':

- If the text is worth remembering, please state "Remember" and provide:
  - A concise summary of the key information.
  - The categories or tags relevant to the knowledge within the text (e.g., science, history, literature).
  - Possible connections to existing knowledge or contexts where this information would be valuable.
  - A brief explanation for why the information is worth remembering.

- If the text is not worth remembering, please state "Forget" and provide:
  - A brief explanation for why the information lacks long-term significance or relevance.

Your response:"""
@backoff.on_exception(backoff.expo,
                      (openai.error.OpenAIError, openai.error.RateLimitError),
                      max_tries=8)
# Example of using `format` method to prepare a specific prompt instance
# prepared_prompt = PROMPT_TEMPLATE.format(output="Here goes the output text to analyze.")
def prepare_prompt(item):
    """
    Prepares a prompt from a dataset item using a predefined template.

    Args:
        item (dict): A single item from the dataset, typically a dictionary.

    Returns:
        str: The formatted prompt.
    """
    # Assuming item contains 'instruction' and 'output' keys, customize as necessary
    instruction = item.get('instruction', '[No instruction provided]')
    output = item.get('output', '[No output provided]')
    formatted_prompt = PROMPT_TEMPLATE.format(instruction=instruction, output=output)
    return formatted_prompt


def send_gpt_request(prompt, model_name="gpt-3.5-turbo", max_tokens=150, temperature=0.7):
    """
    Sends a prompt to the OpenAI API and returns the response.

    Args:
        prompt (str): The prepared prompt to send.
        model_name (str, optional): The model to use for generating the response. Defaults to 'gpt-3.5-turbo'.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 150.
        temperature (float, optional): Controls the randomness of the response. Defaults to 0.7.

    Returns:
        dict: The response from the API or None if an error occurs.
    """
    try:
        response = openai.Completion.create(
            engine=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
        )
        return response
    except openai.error.OpenAIError as e:
        print(f"OpenAI API error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None

def process_response(response):
    """
    Processes the OpenAI API response to extract the reflection token and other information.

    Args:
        response (dict): The response dictionary received from the OpenAI API.

    Returns:
        tuple: A tuple containing the reflection token (str) and a dictionary with additional parsed information.
    """
    if response is None or "choices" not in response or len(response['choices']) == 0:
        return None, {}
    
    text = response['choices'][0]['text'].strip()
    
    # Example parsing strategy, needs to be adjusted based on actual response format
    reflection_token = None
    if "[Remember]" in text:
        reflection_token = "Remember"
    elif "[Forget]" in text:
        reflection_token = "Forget"
        
    # Additional parsing logic for summary, categorization, etc. would go here
    # For demonstration, let's just return the full response text with the token
    additional_info = {"full_text": text}
    
    return reflection_token, additional_info

def augment_dataset(dataset, model_name="gpt-3.5-turbo", max_tokens=150, temperature=0.7):
    """
    Augments each item in the dataset with a reflection token and additional information.

    Args:
        dataset (list): A list of dataset items to be augmented.
        model_name (str, optional): The model to use for generating the response. Defaults to 'gpt-3.5-turbo'.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 150.
        temperature (float, optional): Controls the randomness of the response. Defaults to 0.7.

    Returns:
        list: The augmented dataset.
    """
    augmented_dataset = []
    for item in dataset:
        prompt = prepare_prompt(item)
        response = send_gpt_request(prompt, model_name=model_name, max_tokens=max_tokens, temperature=temperature)
        reflection_token, additional_info = process_response(response)
        item['reflection_token'] = reflection_token
        item.update(additional_info)  # Merge additional_info dict into the item dict
        augmented_dataset.append(item)
    return augmented_dataset


def save_augmented_dataset(augmented_dataset, output_file_path):
    """
    Saves the augmented dataset to a file in JSON format.

    Args:
        augmented_dataset (list): The augmented dataset to save.
        output_file_path (str): The path to the output file where the dataset will be saved.
    """
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_dataset, f, ensure_ascii=False, indent=4)
    
def preprocess_data(dataset):
    """
    Preprocesses the dataset, applying necessary transformations.

    Args:
        dataset (list): List of data items (e.g., dictionaries) to preprocess.

    Returns:
        list: The preprocessed dataset.
    """
    preprocessed_dataset = []
    for item in dataset:
        # Example preprocessing step: Cleaning text in 'output' field
        if 'output' in item:
            item['output'] = clean_text(item['output'])
        # Add any additional preprocessing steps here
        
        preprocessed_dataset.append(item)
    return preprocessed_dataset

def clean_text(text):
    """
    Cleans text by removing unwanted characters, spaces, etc.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    # Example cleaning operation: removing extra whitespace and line breaks
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    # Add any additional cleaning rules here
    return cleaned_text
# Decorator to apply exponential backoff and retry strategy for specific types of exceptions

def completions_with_backoff(prompt, model_name="gpt-3.5-turbo", max_tokens=150, temperature=0.7):
    """
    Sends prompt to OpenAI API with retry strategy.

    Args:
        prompt (str): The prompt to send to the OpenAI.
        model_name (str, optional): Name of the OpenAI model to use. 
                                    Defaults to "gpt-3.5-turbo".
        max_tokens (int, optional): Maximum number of tokens to produce. 
                                    Defaults to 150.
        temperature (float, optional): Sampling temperature. 
                                       Defaults to 0.7.

    Returns:
        dict: The OpenAI API response.
    """
    response = openai.Completion.create(
        engine=model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        n=1,
        stop=None  # You can define stop sequences if necessary
    )
    return response

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst

def main():
    parser = argparse.ArgumentParser(description="Augment a dataset with reflection tokens using OpenAI's API.")
    parser.add_argument('--input_files', type=str, nargs='+', help='A list of paths to the input dataset files (JSON or JSONLines format).')
    parser.add_argument('--output_file', type=str, help='The path to the output file where the augmented dataset will be saved.')
    parser.add_argument('--api_key', type=str, help='Your OpenAI API key.')
    parser.add_argument('--model_name', type=str, default="gpt-3.5-turbo", help='The model to use for generating the response. Defaults to "gpt-3.5-turbo".')
    parser.add_argument('--max_tokens', type=int, default=150, help='The maximum number of tokens to generate. Defaults to 150.')
    parser.add_argument('--temperature', type=float, default=0.7, help='Controls the randomness of the response. Defaults to 0.7.')

    args = parser.parse_args()

    # Set the OpenAI API key from the command line argument
    openai.api_key = args.api_key

    # Load the dataset from the specified input files
    dataset = load_jsonlines(args.input_files)

    # Optionally preprocess the dataset
    preprocessed_dataset = preprocess_data(dataset)

    # Augment the dataset with reflection tokens and additional information
    augmented_dataset = augment_dataset(preprocessed_dataset, 
                                        model_name=args.model_name, 
                                        max_tokens=args.max_tokens, 
                                        temperature=args.temperature)

    # Save the augmented dataset to the specified output file
    save_augmented_dataset(augmented_dataset, args.output_file)

    print("Dataset augmentation complete. Augmented dataset saved to:", args.output_file)

if __name__ == "__main__":
    main()