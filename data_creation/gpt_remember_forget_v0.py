import openai 
import argparse
import json
import backoff
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


def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def send_gpt_request(self, prompt):
    """
    Sends a prompt to the OpenAI API and handles potential errors.

    Args:
        prompt (str): The prompt to send to the language model.
        temperature (float, optional): Controls the creativity of the response. Defaults to 0.7.

    Returns:
        The raw response from the OpenAI API.
    """
    try:
        response = openai.Completion.create(
            engine=self.model_name,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,  # Adjust max_tokens if needed
            n=1,
            stop=None,
        )
        return response

    except openai.error.APIError as e:
        print(f"An API error occurred: {e}")
        return None  # Or you might want to raise an exception instead

    except openai.error.Timeout as e:
        print(f"A timeout occurred: {e}")
        return None  # Or you might want to retry later

    except openai.error.APIConnectionError as e:
        print(f"A connection error occurred: {e}")
        return None  # Or you might want to retry later

            
def extract_rating(self, response):
    """
    Extracts the "Remember" or "Forget" rating from the OpenAI response.

    Args:
        response: The raw response object returned by the OpenAI API.

    Returns:
        str: "Remember" if the rating is positive, "Forget" otherwise.
    """

    try:
        # Placeholder: Update these lines based on how your LLM indicates rating
        for choice in response["choices"]:
            if "Rating: [Remember]" in choice["text"]:
                return "Remember"
            elif "Rating: [Forget]" in choice["text"]:
                return "Forget"

        # Handle the case where no explicit rating is found
        print("Warning: No clear rating found in response.")
        return None  # Or you might decide to return a default like "Forget" 

    except KeyError as e:
        print(f"Error extracting rating. Missing expected key: {e}")
        return None
    
    

def process_input(example, multi_retrieval=False):
    if multi_retrieval is False:
        return PROMPT_TEMPLATE["context"].format_map(example)
    else:
        if "sent_idx" not in example or example["sent_idx"] == 0 or len(example["preceding_sentences"]) == 0:
            return PROMPT_TEMPLATE["multi_no_preceding"].format_map(example)
        else:
            return PROMPT_TEMPLATE["multi"].format_map(example)
    

def postprocess(results):
    raw_output = results["choices"][0]["message"]["content"]
    print(raw_output)
    if "\nExplanation:" in raw_output:
        explanation = raw_output.split("\nExplanation:")[1]
        if explanation[0] == " ":
            explanation = explanation[1:]
        score_string = raw_output.split("\nExplanation:")[0]
        score = None
        for i in range(1, 6):
            if str(i) in score_string:
                score = int(i)
        if score is None:
            return "", explanation
        else:
            return score, explanation
    else:
        return "", ""

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', type=str, nargs='+')
    parser.add_argument('--output_file_name', type=str)
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--model_name', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--org_name', type=str)
    parser.add_argument('--max_tokens', type=int )
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

    
    result_list = []
    if args.n is not None and len(examples) > args.n:
        examples = random.sample(examples, k=args.n)
    
    #task_types = Counter([item["dataset_name"]                             ## implement different prompt template based on varied instructions? 
    #                     for item in examples if "dataset_name" in item])

    input = process_input(example, multi_retrieval=args.multi_retrieval)
    try:
        results = completions_with_backoff(
            model=args.model_name,
            messages=[
                {"role": "user",
                    "content": input},
            ],
            request_timeout=60,
            max_tokens=200,
        )
        score, explanation = postprocess(results)
        result_list.append({"input": example, "score": score, "explanation": score,
                            "raw_output": results["choices"][0]["message"]["content"]})
        if idx % 20 == 0:
            print("Input: {}".format(example["instruction"]))
            print("Output: {}".format(example["output"]))
            print("Evidence: {}".format(example["evidence"]))
            print("Score: {0} ({1})".format(score, explanation))

    except (APIError, Timeout, APIConnectionError):
        results = "ERROR: API error outputs"
        
        
if __name__ == "__main__":
    main()
    
    
        
        
        
        
        
        
   