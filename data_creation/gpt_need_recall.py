import jsonlines
import argparse
from tqdm import tqdm


RECALL_PROMPT_DICT = {
    "context": (
        "Given an instruction, please make a judgment on whether referring to similar previous cases or solutions "
        "enhances the response. Please answer [Yes] or [No] and provide an explanation.\n\n"
        "##\nInstruction: Summarize the main themes of 'Pride and Prejudice'.\n"
        "Need recall?: [Yes]\n"
        "Explanation: Previous analyses of classic literature can provide deep insights into themes, "
        "therefore recalling similar cases is beneficial.\n\n"
        "##\nInstruction: What is 2 + 2?\n"
        "Need recall?: [No]\n"
        "Explanation: This is a basic arithmetic question that does not benefit from referring to previous cases.\n\n"
        "##\nInstruction: Provide tips for first-time travelers to Japan.\n"
        "Need recall?: [Yes]\n"
        "Explanation: Advice from similar travel-related queries could offer comprehensive and valuable tips, "
        "making recall useful.\n\n"
        "##\nInstruction: Explain the process of photosynthesis.\n"
        "Need recall?: [Yes]\n"
        "Explanation: Scientific explanations can be enriched by recalling detailed descriptions from similar cases.\n\n"
        "##\nInstruction: Describe your favorite memory.\n"
        "Need recall?: [No]\n"
        "Explanation: This personal reflection question is unique to the individual and does not benefit from recalling information from past cases.\n\n"
        "##\nInstruction: How does blockchain technology work?\n"
        "Need recall?: [Yes]\n"
        "Explanation: Technical topics can often be clarified by referencing explanations from previous cases, highlighting the benefits of recall.\n\n"
        "##\nInstruction:{instruction}\n"
        "Need recall?: "
    ),
}

def evaluate_need_recall(instruction, output, evidence, preceding_sentences):
    """
    Determines whether a recall from the case base is needed.
    This is a placeholder for your decision logic.
    
    Parameters:
    - instruction (str): The input instruction.
    - output (str): The segment-level output generated.
    - evidence (str): The retrieved Wikipedia paragraph.
    - preceding_sentences (str): Previously generated sentences.
    
    Returns:
    - bool: True if recall is needed, False otherwise.
    """
    # Placeholder logic for determination. Replace with your actual criteria.
    # Example: Assume recall is needed if no evidence provided.
    return evidence.strip() == ""

def similarity_score(text1, text2):
    """
    Calculates a simplistic similarity score between two texts. In practice, you might want
    to use more advanced NLP techniques like embeddings comparison.
    """
    common_terms = set(text1.lower().split()) & set(text2.lower().split())
    return len(common_terms) / min(len(text1.split()), len(text2.split()))

def evaluate_need_recall(instruction, output, evidence, preceding_sentences):
    """
    Determines whether recalling information from a case base could improve the
    generation based on certain criteria such as complexity, evidence relevance, and
    similarity to previous cases.
    
    Parameters:
    - instruction (str): The input instruction.
    - output (str): The segment-level output generated.
    - evidence (str): The retrieved Wikipedia paragraph.
    - preceding_sentences (str): Previously generated sentences.
    
    Returns:
    - bool: True if recall is needed, False otherwise.
    """
    # Check if evidence is directly relevant and sufficient
    if evidence.strip() == "" or similarity_score(instruction, evidence) < 0.2:
        return True  # Recall needed due to lack of/little relevance in evidence

    # Check if the output is overly generic or lacks specificity, which might
    # indicate that drawing on similar past cases could provide more detailed responses
    if len(output.split()) < 5:  # Example condition for overly simplistic output
        return True

    # Check if instruction closely matches or is similar to preceding cases,
    # suggesting a precedent exists that could directly inform the response
    if preceding_sentences and similarity_score(instruction, preceding_sentences) > 0.5:
        return True  # Similar to previous cases, warranting a recall

    # Example of additional logic: if the instruction asks for a 'list' or 'examples',
    # and the output doesn't match this format, it may benefit from recalling similar cases
    if 'list' in instruction.lower() or 'examples' in instruction.lower():
        if ',' not in output and 'and' not in output:  # simplistic check for list-like output
            return True

    return False  # No recall needed

def process_entries(input_file, output_file):
    """
    Processes each entry in the dataset to determine the need for recall.

    Parameters:
    - input_file (str): Path to the input JSONL file.
    - output_file (str): Path where the output JSONL file will be saved.
    """
    with jsonlines.open(input_file) as reader, jsonlines.open(output_file, mode='w') as writer:
        for obj in tqdm(reader):
            # Extract relevant data from the current object.
            instruction = obj.get("instruction", "")
            target_output = obj.get("target_output", "")
            evidence = obj.get("evidence", "")
            preceding_sentences = obj.get("preceding_sentences", "")

            # Evaluate if recall is needed based on the current entry.
            need_recall = evaluate_need_recall(instruction, target_output, evidence, preceding_sentences)

            # Update the object with the recall decision.
            obj['Need_Recall'] = '[Yes]' if need_recall else '[No]'

            # Write the updated object to the output file.
            writer.write(obj)

def main():
    parser = argparse.ArgumentParser(description="Determine need for CBR Recall in SELF-RAG dataset.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output JSONL file where results will be saved.")

    args = parser.parse_args()

    process_entries(args.input_file, args.output_file)
    
    print(f"Processing complete. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()