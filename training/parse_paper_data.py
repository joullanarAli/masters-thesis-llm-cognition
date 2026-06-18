"""
Parse the paper's original evaluation data from the codegen-perceiving folder
Converts the JSON files into a pandas DataFrame similar to master_dataset.jsonl
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def parse_paper_data(data_path):
    """
    Parse all JSON files from the paper's evaluation data.
    
    Parameters:
    data_path: Path to 'data/raw/codegen-perceiving/evaluation_results_server/'
    """
    
    records = []
    data_dir = Path(data_path)
    
    print(f"📂 Scanning directory: {data_dir}")
    
    json_files = list(data_dir.glob('*.json'))
    print(f"   Found {len(json_files)} JSON files")
    
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract evaluator info from filename or data
        evaluator_name = data.get('name_input', 'Anonymous')
        evaluation_datetime = data.get('datetime', '')
        
        # Helper to convert -2..+2 scale to 1..5 scale (paper used -2 to +2)
        def convert_scale(score):
            if score is None:
                return 3  # Neutral/default
            return score + 3
        
        # Process each question in questions_df
        questions_df = data.get('questions_df', [])
        
        for idx, question_item in enumerate(questions_df):
            # Determine if this is Left or Right based on position or CSV_PATH
            csv_path = question_item.get('CSV_PATH', '')
            model_tag = question_item.get('MODEL_TAG', 'Unknown')
            
            # Get consistency, correctness, usefulness based on position
            # The data structure: questions_df[0] corresponds to Left?
            # We need to match with consistent_L/correct_L/useful_L
            if idx == 0:
                consistency = data.get('consistent_L', 0)
                correctness = data.get('correct_L', 0)
                usefulness = data.get('useful_L', 0)
                position = 'Left'
            else:
                consistency = data.get('consistent_R', 0)
                correctness = data.get('correct_R', 0)
                usefulness = data.get('useful_R', 0)
                position = 'Right'
            
            # Skip if all scores are None/0 and no question
            if consistency == 0 and correctness == 0 and usefulness == 0:
                continue
            
            record = {
                'timestamp': evaluation_datetime,
                'evaluator_name': evaluator_name,
                'position': position,
                'question': question_item.get('Question', ''),
                'answer': question_item.get('Answer', ''),
                'model_name': model_tag,
                'csv_path': csv_path,
                'code_formatting': question_item.get('CODE_FORMATTING', False),
                'consistency_score': convert_scale(consistency),  # Understanding
                'correctness_score': convert_scale(correctness),   # Agreement
                'usefulness_score': convert_scale(usefulness),     # Intention
            }
            records.append(record)
    
    df = pd.DataFrame(records)
    
    # Convert usefulness_score to intention_to_use categories
    def usefulness_to_intention(score):
        if score >= 4:  # Original 1 or 2 on converted scale (Probably Yes or Yes)
            return 'yes'
        elif score == 3:  # Original 0 (Neutral)
            return 'modified'
        else:  # Original -1 or -2 (Probably No or No)
            return 'no'
    
    df['intention_to_use'] = df['usefulness_score'].apply(usefulness_to_intention)
    
    # Create model type flags
    df['is_finetuned'] = df['model_name'].str.contains('FINETUNED', case=False, na=False).astype(int)
    df['is_vanilla'] = df['model_name'].str.contains('VANILLA', case=False, na=False).astype(int)
    df['is_pipeline'] = df['model_name'].str.contains('PIPELINE', case=False, na=False).astype(int)
    
    return df

def main():
    print("="*70)
    print("PARSING PAPER'S ORIGINAL DATA")
    print("="*70)
    
    # Path to the paper's data
    paper_data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'codegen-perceiving' / 'evaluation_results_server'
    
    if not paper_data_path.exists():
        print(f"❌ Path not found: {paper_data_path}")
        print("   Please check the path to your paper data.")
        return
    
    # Parse data
    df = parse_paper_data(paper_data_path)
    
    print(f"\n📊 Parsed Data Summary:")
    print(f"   Total records: {len(df)}")
    print(f"   Evaluators: {df['evaluator_name'].nunique()}")
    print(f"   Models: {df['model_name'].nunique()} unique models")
    print(f"\n   Score ranges:")
    print(f"   Consistency: {df['consistency_score'].min()} - {df['consistency_score'].max()}")
    print(f"   Correctness: {df['correctness_score'].min()} - {df['correctness_score'].max()}")
    print(f"   Usefulness: {df['usefulness_score'].min()} - {df['usefulness_score'].max()}")
    
    print(f"\n   Intention distribution:")
    print(df['intention_to_use'].value_counts())
    
    print(f"\n   Model type flags:")
    print(f"   Finetuned models: {df['is_finetuned'].sum()}")
    print(f"   Vanilla models: {df['is_vanilla'].sum()}")
    print(f"   Pipeline models: {df['is_pipeline'].sum()}")
    
    # Save parsed data
    output_path = Path(__file__).parent / 'results' / 'paper_data_parsed.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Parsed data saved to: {output_path}")
    
    # Show sample
    print("\n📋 Sample records:")
    print(df[['evaluator_name', 'model_name', 'consistency_score', 'correctness_score', 'usefulness_score', 'intention_to_use']].head(10).to_string(index=False))
    
    return df

if __name__ == "__main__":
    df = main()