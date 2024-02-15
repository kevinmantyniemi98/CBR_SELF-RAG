import openai
import os
import json
import logging
import argparse
import random
import json
import pprint
from tqdm import tqdm
import JSONLPlotter
import PrettyPrintJsonl
import display

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')



PROMPT_TEMPLATE_GEM2 = """Input: "{text}"

Instructions: Analyze the input against the criteria below. Output a recommendation ([Remember] or [Forget]), alongside a brief explanation supporting your decision.

Criteria:

Novelty: Does the input represent a highly unusual case, rare experience, or deviate significantly from known data?
Contextual Significance: Is there rich context associated with the input, or does it reveal important temporal dynamics?
Corrective Potential: Does the input contain insights into past errors, point towards improvements, or provide useful feedback?
Pattern Breaker: Does the input stand out as an anomaly or contradict well-established patterns?
Theoretical Alignment: Does the input confirm or challenge existing principles and theories in the domain?
Output Format:

Recommendation: [Remember] or [Forget]
Explanation: A concise sentence or two justifying the decision based on the criteria where the input exhibited clear traits.
Example Explanation Formats:

[Remember] - Explanation: "This input offers a rare troubleshooting scenario and could expand the system's knowledge base for similar future issues."
[Forget] - Explanation: "This input aligns with typical patterns and offers no new contextual insights."
Important Considerations

Explanation Depth: Control the level of detail in the explanation. You can adjust from simple mentions of the criteria to more elaborate statements.
Specificity: Tailor the criteria and explanations to your specific NLP or CBR project for the most meaningful insights.
Let's try an Example!

Input: A customer review complaining that a chatbot keeps recommending outdated product information.

Output:

Recommendation: [Remember]
Explanation: This input highlights a potential error in the knowledge base or update process and offers corrective feedback."""

PROMPT_TEMPLATE_GEM = """Input: "{text}"

Instructions: Carefully analyze the provided input. Assess it against the following criteria, providing insights and examples when relevant.

Criteria:

Expanding Knowledge:

Does the input offer unique cases or experiences that could enhance a system's knowledge base?
Are there rare or unusual aspects that push boundaries of the current understanding?
Contextual Richness:

What situational elements are significant within the input?
Does the input reveal temporal trends or changes over time?
Evolution Tracking:

Could the input inform improvements or adaptations? Are there hints about what has worked or failed in similar situations?
Would modifying current knowledge, based on this input, be beneficial? If so, how?
Patterns & Anomalies:

Does the input align with existing patterns or trends?
Are there anomalies or outliers that deserve special attention?
Principles & Lessons:

Does the input reinforce or challenge current principles/theories?
What practical lessons (either positive or negative) can be extracted from the input?
Output Format:

Create a summary report for each criterion addressed above.
Highlight examples and key takeaways from the input.
If applicable, offer suggestions on how a dynamic memory system might store or leverage this information.
Customization:

Focus: Add or remove criteria as needed to prioritize your analysis.
Output Specificity: Adjust the output format to capture the depth of analysis required. You might ask for bullet points, tables, or more extensive explanations.
Domain Adaptation: Consider adding criteria or terminology specific to your area of NLP or CBR if there are unique concerns.
Example Usage (NLP focus)

Input:  A customer service transcript highlighting a complex technical issue and its unexpected workaround.

Instructions: ... (Using the prompt template as described above)

Key Insights from Potential Model Output:

Expanding Knowledge: New failure mode and related workaround added to potential troubleshooting paths.
Contextual Richness: The issue appears tied to a specific software update, requiring temporal adjustment of troubleshooting advice.
Patterns & Anomalies: Workaround suggests an unusual interaction between modules, possibly an edge case not in typical system logic.
Principles & Lessons: Highlights the importance of customer-sourced workarounds, reinforces the need for a system to gracefully handle the unexpected."""

PROMPT_TEMPLATE_15 = """In the context of dynamic memory, especially within fields like natural language processing (NLP) and case-based reasoning (CBR), 
what is worth remembering hinges on the goal of creating systems that can adapt, evolve, and respond to new information efficiently and effectively. 
Given this, there are several categories of information and attributes that are particularly worth remembering:

1. Experiences that Expand Knowledge Base
Varied Cases: To avoid overfitting on a narrow set of experiences, it's crucial to remember a diverse array of cases.
Unusual or Rare Instances: These cases help the system learn from exceptions and refine its understanding and predictive capacities.
2. Contextually Rich Information
Situational Contexts: The context surrounding a case significantly affects its interpretation. Remembering the context in which information is used or an event occurs can provide deeper insight into its relevance and applicability.
Temporal Dynamics: How information or cases evolve over time can offer predictive insights into future trends and patterns.
3. Evolution of Cases and Feedback
Feedback Loops: Information about how well a response or solution worked and user or system feedback is invaluable for learning and improvement.
Adjustments and Annotations: Notes on how and why a case was altered based on new insights or mistakes are crucial for understanding the evolution of knowledge.
4. Patterns and Anomalies
General Patterns: Broad trends and commonalities across cases form the backbone of the system's understanding, facilitating quicker adaptation to familiar situations.
Anomalies: Outliers or anomalies can indicate emerging trends, new patterns, or errors in the current system, making them valuable for continuous learning.
5. Principles and Theories
Underlying Principles: Fundamental principles or theories guiding the domain of interest should be retained, as they provide a framework for interpreting and integrating new information.
Practical Lessons: Lessons learned from past experiences, including successes and failures, enrich the system's decision-making abilities.




"""
PROMPT_TEMPLATE = """Please read the following text and analyze its content. 
Determine whether it provides significant value for long-term learning and could be useful for 
future tasks. Consider if it includes key concepts, definitions, rules, guidelines, 
clear procedures, noteworthy insights, or unusual examples that challenge assumptions or 
enhance understanding in a specific domain.

Text to analyze:
"{output}"

Based on your analysis, classify the text as either 'Remember' or 'Forget':

- If the text is worth remembering, please state "[Remember]" and provide:
  - A concise summary of the key information.
  - The categories or tags relevant to the knowledge within the text (e.g., science, history, literature).
  - Possible connections to existing knowledge or contexts where this information would be valuable.
  - A brief explanation for why the information is worth remembering.

- If the text is not worth remembering, please state "[Forget]" and provide:
  - A brief explanation for why the information lacks long-term significance or relevance.

Your response:"""
PROMPT_TEMPLATE3 = """
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
"""
PROMPT_TEMPLATE_SCHANK = """Given Roger C. Schank's emphasis on dynamic learning, case-based reasoning, and the practical application of knowledge, analyze the following text. Assess its potential for fostering understanding, facilitating problem-solving, and contributing to long-term learning in real-world contexts.

Text to analyze:
"{text}"

Assess the text according to the following criteria derived from Schank's educational philosophy:

1. Does the text offer practical examples or case studies that illuminate theoretical concepts?
2. Does it highlight problem-solving techniques that can be applied across different situations?
3. Does it challenge existing assumptions or encourage critical thinking?
4. Does it provide a basis for generating new ideas or hypotheses for future exploration?

Based on the analysis, classify the text as either 'Remember' or 'Forget':

- If the text is deemed valuable for integration, please state "[Remember]" and provide:
  - A concise summary that captures the essence and practical value of the information.
  - Suggestions for potential real-world applications or contexts where this knowledge could be applied.
  - Possible connections to existing knowledge bases or frameworks that enhance comprehension and retention.
  - A rationale for its integration, focusing on its utility for dynamic learning and problem-solving.

- If the text is considered non-essential or not conducive to the application-oriented learning approach, state "[Forget]" and provide:
  - A brief explanation for its limited applicability or relevance to dynamic, real-world learning environments as envisioned by Schank.

Your response:"""
#
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

def generate_rf_tokens(filepath, output_file_path, model_name="gpt-3.5-turbo", max_tokens=150, temperature=0.7, num_lines=20):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            total_lines = sum(1 for _ in file)
            logging.info(f"Total lines in file: {total_lines}")
            
        if total_lines == 0:
            logging.warning("The input file is empty.")
            return

        with open(filepath, 'r', encoding='utf-8') as file, open(output_file_path, 'w', encoding='utf-8') as outfile:
            num_lines = min(num_lines, total_lines)
            random_line_numbers = sorted(random.sample(range(total_lines), num_lines))  # Sorted for readability
            selected_lines = [line for i, line in enumerate(file) if i in random_line_numbers]

            with tqdm(total=num_lines) as pbar:  # Initialize progress bar here
                for line in selected_lines:
                    json_line = json.loads(line)
                    try:
                        json_line = json.loads(line)
                        if 'instruction' in json_line and 'output' in json_line and 'id' in json_line:
                            instruction = json_line['instruction']
                            output = json_line['output']
                            compiled_imput = "Instruction: " + instruction + " Output: " + output
                            id = json_line['id']

                            #prompt_message = PROMPT_TEMPLATE3.format(output=compiled_imput)
                            #prompt_message = PROMPT_TEMPLATE3.format(output=output, instruction=instruction)
                            prompt_message = PROMPT_TEMPLATE_GEM2.format(text=compiled_imput)
                        
                            processed_text = send_chat_to_openai(prompt_message, model_name, max_tokens, temperature)
                            pbar.update(1)  # Update after sending to OpenAI
                        
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
                                logging.info(f"Processed line ID: {id}")
                            else:
                                logging.warning(f"Failed to get a response for line ID: {id}")
                    except json.JSONDecodeError:
                        logging.error("Failed to decode JSON line. Skipping.")

    except FileNotFoundError:
        logging.error(f"The file {filepath} was not found.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

def extract_rf_data(processed_text):
    """Extracts RF_token (Remember/Forget) and RF_context from processed_text."""
    # Define the token and its length
    token = "[Remember]" if "[Remember]" in processed_text else "[Forget]"
    token_length = len(token)

    context_start = processed_text.find(token) + token_length

    context = processed_text[context_start:].strip()

    return token, context

def read_and_pretty_print_jsonl(file_path):
    """
    Reads a .jsonl file, parses each JSON object, and
    pretty-prints the content to the console.

    Parameters:
    file_path (str): Path to the .jsonl file to be read.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                json_obj = json.loads(line)
                pprint.pprint(json_obj, width=80, compact=False)
                print("\n")
            except json.JSONDecodeError:
                print("Error decoding JSON.")


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
    input_filepath = os.path.join(script_dir, 'Self_rag_train.jsonl')
    output_filepath = os.path.join(script_dir, 'output_remember_forget.jsonl')
    generate_rf_tokens(input_filepath, output_filepath)
    file_path = 'SEMI_FINAL\output_remember_forget.jsonl'
    display.__name__
