import openai
import os

# Make sure you replace "Your_OpenAI_API_Key" with your actual OpenAI API key, or better,
# use environment variables for added security.
# For example, in your terminal:
# export OPENAI_API_KEY='your_api_key_here'
# And in this script, use:
# openai.api_key = os.getenv("OPENAI_API_KEY")
#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

openai.api_key = "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"


def send_chat_to_openai(messages, model="gpt-3.5-turbo", temperature=0.7):
    """
    Sends a series of messages to the OpenAI chat API and gets the generated response.

    Parameters:
    - messages (list): A list of message dictionaries for the chat.
    - model (str, optional): The ID of the model to use. Defaults to "gpt-3.5-turbo".
    - temperature (float, optional): Controls the randomness of the response. Defaults to 0.7.
    
    Returns:
    - str: The text generated by the model.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        # print(response)
        generated_text = response.choices[0].message['content'].strip()
        return generated_text
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred: See console for details."

# Correctly structured messages example
messages = [
    {"role": "system", "content": "You are a helpful AI."},
    {"role": "user", "content": "Hello, who are you?"},
]

ROMPT_TEMPLATE = """Please read the following text and analyze its content. 
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
# Using the corrected function call
response_text = send_chat_to_openai(messages)

print("Response from OpenAI:", response_text)