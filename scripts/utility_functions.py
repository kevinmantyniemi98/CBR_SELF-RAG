# Filename: utility_functions.py

import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_jsonl(file_path):
    """
    Loads a JSONL file and returns a list of dictionaries.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def save_to_jsonl(data, file_path):
    """
    Saves a list of dictionaries to a JSONL file.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n')

def calculate_text_similarity(text1, text2):
    """
    Calculates similarity between two blocks of text using TF-IDF and Cosine Similarity.
    """
    vectorizer = TfidfVectorizer()
    vecs = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vecs[0:1], vecs[1:2])[0][0]

def preprocess_text(text):
    """
    Example of a simple text preprocessing function focusing on lowercasing the text.
    Extend this with more sophisticated preprocessing as needed.
    """
    return text.lower()

def evaluate_utility(score):
    """
    Assigns a label based on the utility score.
    """
    if score >= 4:
        return 'High Utility'
    elif score > 2:
        return 'Medium Utility'
    else:
        return 'Low Utility'

def relevance_decision(retrieved_info, context):
    """
    Determines the relevance of retrieved information to the context.
    Placeholder for a more complex relevance assessment algorithm.
    """
    if not retrieved_info:
        return 'No Retrieval Needed'
    similarity_score = calculate_text_similarity(retrieved_info, context)
    return 'Relevant' if similarity_score > 0.5 else 'Irrelevant'
    