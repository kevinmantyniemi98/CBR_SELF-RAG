import openai
import os
import json
import logging
import argparse
# Setup logging
logging.basicConfig(level=logging.WARN, format='%(asctime)s - %(levelname)s - %(message)s')


# Validate environment variable for OpenAI API key
openai.api_key = "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"
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#if not OPENAI_API_KEY:
#    logging.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
#    exit(1)
#else:
#    openai.api_key = OPENAI_API_KEY

PROMPT_TEMPLATE = """Please read the following text and analyze its content. 
Determine whether it provides significant value for long-term learning and could be useful for 
future tasks. Consider if it includes key concepts, definitions, rules, guidelines, 
clear procedures, noteworthy insights, or unusual examples that challenge assumptions or 
enhance understanding in a specific domain.

Text to analyze:
"{output}"

Based on your analysis, classify the text as either '[Remember]' or '[Forget]':

- If the text is worth remembering, please state '[Remember]' and provide:
  - A concise summary of the key information.
  - The categories or tags relevant to the knowledge within the text (e.g., science, history, literature).
  - Possible connections to existing knowledge or contexts where this information would be valuable.
  - A brief explanation for why the information is worth remembering.

- If the text is not worth remembering, please state '[Forget]' and provide:
  - A brief explanation for why the information lacks long-term significance or relevance.

Your response:"""

PROMPT_TEMPLATE_2 = """Given Case:
{output}

Key Aspects:
1. Problem Statement: What specific issue or challenge does this case address?
2. Solution Approach: How was the problem approached or solved in this case?
3. Outcome: What was the result or impact of the solution implemented?

Consider the following criteria to determine if this case is worth remembering or forgetting:

1. Relevance: Does the case address a problem that is likely to recur or represent common challenges in the field?
2. Uniqueness: Is the solution approach novel, or does it offer insights that significantly differ from standard practices?
3. Applicability: Can the insights from this case be generalized or adapted to solve other problems?
4. Educational Value: Does the case contribute to understanding foundational principles or advanced concepts in the domain?
5. Innovation: Does the case introduce new methodologies, technologies, or perspectives worth preserving for future reference?

Based on the criteria above, please analyze the given case. Conclude whether it is "Worth Remembering" or "Worth Forgetting," and provide a brief rationale for your decision. Include potential implications for remembering or discarding this case in the context of augmenting a dynamic memory system for Case-Based Reasoning.

Decision:
- Worth Remembering
- Worth Forgetting

Rationale:

Implications for CBR and Dynamic Memory:"""

def send_chat_to_openai(json_line, model_name="gpt-3.5-turbo", max_tokens=150, temperature=0.7):
    try:
        message = [
        {"role": "user", "content": f"{json_line}"}
        ]
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        generated_text = response.choices[0].message['content'].strip()
        return generated_text
    except openai.error.OpenAIError as api_error:
        logging.error(f"OpenAI API error: {api_error}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    return None


def read_and_process_json_lines(filepath, output_file_path, model_name="gpt-3.5-turbo", max_tokens=150, temperature=0.7):
    try:
        with open(filepath, 'r') as file, open(output_file_path, 'w') as outfile:
            for line in file:
                try:
                    #print("line: ", line)
                    json_line = json.loads(line)
                    instruction = json_line.get("instruction", "")
                    output = json_line.get("output", "")
                    id = json_line.get("id")
                    # Assuming you want to include both instruction and output in the user message
                    msg_content = f"{instruction} {output}" #

                    #print("msgcontent: ", msg_content)
                    if msg_content:
                        prompt_message = PROMPT_TEMPLATE.format(output=msg_content)
                        processed_text = send_chat_to_openai(prompt_message, model_name, max_tokens, temperature)
                        if processed_text:
                            logging.info(f"***Conclusion***: {processed_text}")
                        else:
                            logging.warning("Failed to get a response for this line.")
 
                    if processed_text:
                        RF_token, RF_context = extract_rf_data(processed_text)
                        output_data = {
                            "id": id,
                            "instruction": instruction,
                            "output": output,
                            "RF_token": RF_token,
                            "RF_context": RF_context
                        }
                        json.dump(output_data, outfile)
                        outfile.write("\n")
                        logging.info(f"Processed and saved data for line.")
                    else:
                        logging.warning("Failed to get a response for this line.")
                except json.JSONDecodeError:
                    logging.error("Failed to decode JSON line. Skipping.")
                
    except FileNotFoundError:
        logging.error(f"The file {filepath} was not found.")
    except Exception as e:
        logging.error(f"An error occurred while reading the file: {e}")


def extract_rf_data(processed_text):
    """Extracts RF_token (Remember/Forget) and RF_context from processed_text."""
    # Define the token and its length
    token = "[Remember]" if "[Remember]" in processed_text else "[Forget]"
    token_length = len(token)

    context_start = processed_text.find(token) + token_length

    context = processed_text[context_start:].strip()

    return token, context

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Augment a dataset with reflection tokens using OpenAI's API.")
    parser.add_argument('--input_files', type=str, nargs='+', help='A list of paths to the input dataset files (JSON or JSONLines format).')
    parser.add_argument('--output_file', type=str, help='The path to the output file where the augmented dataset will be saved.')
    parser.add_argument('--api_key', type=str, help='Your OpenAI API key.')
    parser.add_argument('--model_name', type=str, default="gpt-3.5-turbo", help='The model to use for generating the response. Defaults to "gpt-3.5-turbo".')
    parser.add_argument('--max_tokens', type=int, default=150, help='The maximum number of tokens to generate. Defaults to 150.')
    parser.add_argument('--temperature', type=float, default=0.7, help='Controls the randomness of the response. Defaults to 0.7.')

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the file path relative to the script directory
    input_filepath = os.path.join(script_dir, 'self_rag_sub8.jsonl')
    output_filepath = os.path.join(script_dir, 'output_remember_forget2.jsonl')
    read_and_process_json_lines(input_filepath, output_filepath)
