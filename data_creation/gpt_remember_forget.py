import openai
import pandas as pd
import argparse
import json
from collections import Counter
from tqdm import tqdm
import backoff
from openai.error import APIError, Timeout, APIConnectionError
import jsonlines
import random

PROMPT_TEMPLATE = (
    "analyze the provided text. If it offers significant value for long-term learning and future tasks,  you should remember it.  This could include: \n",
    "Key Concepts & Definitions: Clear explanations of terms or ideas important to a wider domain of knowledge.",
    "Rules & Guidelines: Principles that should be adhered to or general best practices within a subject.",
    "Procedures: Step-by-step instructions for completing specific actions or processes.",
    "Insights & Unusual Examples: Novel information, unique case studies, or things that challenge previous assumptions.",
    "If the text IS worth remembering: \n",
    "Rating: [Remember]",
    "Summarize: Create a concise, easily understandable summary of t    he key information.",
    "Categorize: Assign the knowledge to relevant categories or tags for easy retrieval (e.g., biology, physics, writing tips)",
    "Connections: If possible, link this new knowledge to existing information you already possess to strengthen your understanding.",
    "Explanation (Optional): Briefly note why the information is worth remembering.",
    "If the text IS NOT worth remembering:",
    "Rating: [Forget]",
    "Explanation (Optional): Briefly note why the information lacks long-term significance.",
    "EXAMPLES: \n",
    "###Instruction: Question: What is the solution? Solve -52073 = 633*m + 578*m for m. \n",
    "Answer: -43 ",
    "Question: What is the solution? Solve -31993 + 28559 = 202*a for a. \n" ,
    "Answer: -17 \n",
    "Question: What is the solution? Solve 0 = -6680*h + 6952*h + 8976 for h. \n",
    "Answer: -33 \n",
    "Question: What is the solution? Solve 3479*x - 3348*x = 2489 for x. \n",
    "Answer: [Retrieval]<paragraph>Marilyn vos Savant a particular digit (say 5, for example). She said the answer was 4000, yet people showed the correct answer—3439—using various strategies. \n",
    "The incorrect answer of 4000 counted those combinations with more than one '5' multiple times (twice for '1535', three times for '1555', for instance). \n",
    "So, the correct answer is to take all possible combinations minus the combinations in which each digit is not a 5 to the n power, or 10,000 - 9 = 3439. On June 22, 2014, Savant made an error in a word problem. \n",
    "The question was: If two people could complete a project in six</paragraph>[Irrelevant]19[Utility:5]",
    "Rating: [Forget]",
    "Explanation (Optional): The information is irrelevant to long-term learning and future tasks.",
    "##\nInstruction: {instruction}\n"
    "Output:{output}\n"
)

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def load_json_dataset(filename):
  """
  Loads a dataset from a JSON file.

  Args:
    filename: The path to the JSON file.

  Returns:
    A list of dictionaries, where each dictionary represents a data point.
  """

  with open(filename, "r") as f:
    data = json.load(f)

  # Check if the data is in the expected format
  if not isinstance(data, list):
    raise ValueError("Invalid data format. Expected a list of dictionaries.")

  for item in data:
    # Check if each item has the necessary keys
    required_keys = ["instruction", "evidence", "target_output"]
    missing_keys = [key for key in required_keys if key not in item]
    if missing_keys:
      raise ValueError(f"Missing required keys: {missing_keys}")

  return data

def process_input(instruction, evidence, preceding_sentences=None):
  """
  Formats the input prompt based on the instruction type (remember/forget) and considers preceding sentences.

  Args:
    instruction: The instruction ("Remember" or "Forget").
    evidence: The text to be analyzed.
    preceding_sentences: Optional list of preceding sentences for context (if applicable).

  Returns:
    The formatted prompt string.
  """

  # Check instruction type
  if instruction not in ["Remember", "Forget"]:
    raise ValueError(f"Invalid instruction: {instruction}")

  # Build the prompt based on instruction
  prompt = PROMPT_TEMPLATE

  # If relevant, analyze preceding sentences and incorporate insights (you'll need to implement this)
  if preceding_sentences:
    # Analyze preceding sentences to extract relevant information (e.g., key concepts, context)
    # Update the prompt based on the analysis (e.g., highlight specific aspects in evidence)

  # Include evidence in the prompt
  prompt += f"\nAnalysis Text: {evidence}\n"

  # Return the formatted prompt
  return prompt



def augment_with_remember_forget(text):
  """
  This function takes text as input and returns an augmented version
  with remember/forget tokens based on the PROMPT_TEMPLATE.

  Args:
      text: The text to be augmented.

  Returns:
      A string containing the augmented text with remember/forget tokens.
  """
  # Analyze the text (replace with your analysis logic)
  is_valuable = analyze_text_value(text)

  # Construct the prompt based on the analysis
  prompt = PROMPT_TEMPLATE

  if is_valuable:
    prompt += "Rating: [Remember]\n"
    # Add summarization, categorization, connections, and explanation (implement functions)
    prompt += summarize(text)
    prompt += categorize(text)
    prompt += connect_to_existing_knowledge(text)
    prompt += explain_value(text)
  else:
    prompt += "Rating: [Forget]\n"
    # Add explanation for forgetting (implement function)
    prompt += explain_forget(text)

  return prompt

def interact_with_gpt(prompt, model_name="gpt-3.5-turbo", temperature=0.7):
  """
  Sends a prompt to the GPT API and processes the response for remembering/forgetting tasks.

  Args:
    prompt: The formatted prompt string generated by `process_input`.
    model_name: The name of the GPT model to use (default: "gpt-3.5-turbo").
    temperature: The temperature parameter for generating responses (default: 0.7).

  Returns:
    A dictionary containing the extracted information from the GPT response:
      - rating: "Remember" or "Forget"
      - summary: A concise summary of the key information (if generated)
      - categories: List of relevant categories for the information (if generated)
      - explanation: Reason for remembering or forgetting (if generated)
  """

  # Replace this with your actual API interaction code and error handling
  response = send_gpt_request(prompt, model_name, temperature)

  # Extract relevant information from the response (adapt based on your response format)
  rating = extract_rating(response)
  summary = extract_summary(response) if "summary" in response else None
  categories = extract_categories(response) if "categories" in response else []
  explanation = extract_explanation(response) if "explanation" in response else None

  # Return the extracted information
  return {
      "rating": rating,
      "summary": summary,
      "categories": categories,
      "explanation": explanation
  }

def write_data_to_json(data, filename):
  """
  Writes processed data to a JSON file.

  Args:
    data: A list of dictionaries, where each dictionary represents a processed data point.
      Each dictionary should have the following keys:
        - instruction: The original instruction ("Remember" or "Forget").
        - evidence: The text analyzed.
        - target_output: The original target output (e.g., "REMEMBER Paris").
        - generated_token: The generated token by GPT ("REMEMBER" or "FORGET").
        - explanation: The explanation from GPT for the generated token (optional).
    filename: The path to the output JSON file.
  """

  with open(filename, "w") as f:
    json.dump(data, f, indent=4)

  print(f"Data written to: {filename}")

# Placeholder functions for interacting with the GPT API and response parsing (implement these)
def send_gpt_request(prompt, model_name, temperature):
  # Implement sending prompt to GPT API, handle errors, and return response
  pass

def extract_rating(response):
  # Implement logic to extract "Remember" or "Forget" rating from response
  pass

def extract_summary(response):
  # Implement logic to extract summary of key information from response
  pass

def extract_categories(response):
  # Implement logic to extract relevant categories for the information from response
  pass

def extract_explanation(response):
  # Implement logic to extract explanation for remembering/forgetting from response
  pass

def postprocess(response):
  token = response["prediction"]
  explanation = response["explanation"]
  return token, explanation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', type=str, nargs='+')
    parser.add_argument('--output_file_name', type=str)
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--model_name', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--org_name', type=str)
    args = parser.parse_args()

    with open(args.api_key) as f:
        openai.api_key = f.read()[:-1]
    openai.organization = args.org_name

    examples = []
    for input_file in args.input_files:
        if input_file.endswith(".json"):
            examples += json.load(open(input_file))
        else:
            examples += load_jsonlines(input_file)

if __name__ == "__main__":
    main()
