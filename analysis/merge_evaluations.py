# analysis/merge_evaluations.py
import json
from pathlib import Path
import pandas as pd  # You'll need pandas: pip install pandas
# Paths
EVALUATIONS_PATH = Path("../evaluation_ui/evaluations.json")
DATASET_ROOT = Path("../data/raw")  # Root of your generated JSONs
OUTPUT_PATH = Path("analysis/master_dataset.jsonl")  # One JSON per line

def load_all_problems():
    """Load all generated problem-solution pairs from dated folders."""
    all_data = []
    
    # Find all dated folders (e.g., 2026-01-25)
    for date_folder in DATASET_ROOT.iterdir():
        if date_folder.is_dir():
            model_folder = date_folder / "google_gemini"
            for json_file in model_folder.glob("*.json"):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    all_data.append(data)
    return all_data

def main():
    # Load evaluations
    with open(EVALUATIONS_PATH, 'r') as f:
        evaluations = json.load(f)
    
    # Load all generated problems
    problems = load_all_problems()
    
    # Create master list
    master_data = []
    
    for problem in problems:
        problem_id = problem["problem_id"]
        
        # Create merged record
        record = {
            **problem,  # Includes all original fields
            "human_evaluation": evaluations.get(problem_id, {})
        }
        master_data.append(record)
    
    # Save as JSON Lines (easy for analysis)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        for record in master_data:
            f.write(json.dumps(record) + '\n')
    
    print(f"Created master dataset with {len(master_data)} records")
    print(f"Saved to: {OUTPUT_PATH}")
    
    # Optional: Show summary
    rated_count = sum(1 for r in master_data if r["human_evaluation"].get("rated"))
    print(f"{rated_count} problems have human evaluations")

if __name__ == "__main__":
    main()