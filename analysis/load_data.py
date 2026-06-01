"""Load and prepare the master_dataset.jsonl for analysis."""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def load_master_dataset(file_path='../data/master_dataset.jsonl'):
    """Load all evaluations into a pandas DataFrame."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    df = pd.DataFrame(records)
    
    # Extract useful columns
    df['evaluator_id'] = df['evaluator_metadata'].apply(lambda x: x.get('evaluator_id'))
    df['evaluator_name'] = df['evaluator_metadata'].apply(lambda x: x.get('evaluator_name'))
    df['expertise_level'] = df['evaluator_metadata'].apply(lambda x: x.get('expertise_level'))
    df['programming_years'] = df['evaluator_metadata'].apply(lambda x: x.get('programming_years', 0))
    df['ai_usage_frequency'] = df['evaluator_metadata'].apply(lambda x: x.get('ai_usage_frequency'))
    
    # FIX: Convert languages to string, handle None/NaN
    def get_languages(x):
        lang = x.get('languages', '')
        if lang is None or pd.isna(lang):
            return ''
        return str(lang)
    
    df['languages'] = df['evaluator_metadata'].apply(get_languages)
    
    # Extract evaluation scores
    df['correctness'] = df['human_evaluation'].apply(lambda x: x.get('correctness', 0))
    df['readability'] = df['human_evaluation'].apply(lambda x: x.get('readability', 0))
    df['reliability'] = df['human_evaluation'].apply(lambda x: x.get('reliability', 0))
    df['intention_to_use'] = df['human_evaluation'].apply(lambda x: x.get('intention_to_use', 'no'))
    
    # Convert intention_to_use to numeric (usefulness score SU)
    intention_map = {'yes': 2, 'modified': 1, 'no': 0}
    df['usefulness_SU'] = df['intention_to_use'].map(intention_map)
    
    # Extract problem number for sorting
    df['problem_num'] = df['problem_id'].apply(lambda x: int(x.split('/')[1]))
    
    return df

def get_model_comparison(df):
    """Get comparison statistics between Gemini and Groq."""
    gemini = df[df['model_name'] == 'google_gemini']
    groq = df[df['model_name'] == 'groq']
    
    comparison = {
        'gemini_mean_correctness': gemini['correctness'].mean(),
        'groq_mean_correctness': groq['correctness'].mean(),
        'gemini_mean_readability': gemini['readability'].mean(),
        'groq_mean_readability': groq['readability'].mean(),
        'gemini_mean_reliability': gemini['reliability'].mean(),
        'groq_mean_reliability': groq['reliability'].mean(),
        'gemini_usefulness_rate': (gemini['intention_to_use'] == 'yes').mean(),
        'groq_usefulness_rate': (groq['intention_to_use'] == 'yes').mean()
    }
    return comparison

if __name__ == "__main__":
    df = load_master_dataset()
    print(f"Total records: {len(df)}")
    print(f"Evaluators: {df['evaluator_name'].unique()}")
    print(f"Models: {df['model_name'].unique()}")
    print(f"\nColumns: {df.columns.tolist()}")