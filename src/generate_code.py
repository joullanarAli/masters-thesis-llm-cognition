# Main script to run

#!/usr/bin/env python3
"""
MAIN SCRIPT: Daily Code Generation for Thesis Dataset
Run this to generate code for HumanEval problems.
"""
import os
import sys
import yaml
import json
import time
from datetime import datetime
from pathlib import Path

# Add src to the Python path so we can import our module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your API client function
from api_clients import generate_with_google

def load_config():
    """Load API keys and configuration from the YAML file."""
    config_path = Path(__file__).parent.parent / 'configs' / 'api_keys.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# def load_humaneval_problems(limit=5):
#     """
#     Load a sample of HumanEval problems.
#     START WITH limit=5 to test, then increase to 50, 100, etc.
#     """
#     # SAMPLE DATA - Replace this later with the full HumanEval.jsonl
#     sample_problems = [
#         {
#             "task_id": "HumanEval/0",
#             "prompt": "from typing import List\n\ndef add_two_numbers(a: int, b: int) -> int:\n    \"\"\"Return the sum of two integers.\"\"\"\n",
#         },
#         {
#             "task_id": "HumanEval/1",
#             "prompt": "def reverse_string(s: str) -> str:\n    \"\"\"Return the reversed version of the input string.\"\"\"\n",
#         },
#         {
#             "task_id": "HumanEval/2",
#             "prompt": "def find_max(numbers: List[int]) -> int:\n    \"\"\"Return the maximum number in a list.\"\"\"\n",
#         },
#     ]
#     return sample_problems[:limit]



def load_real_humaneval(file_path: str, limit: int = None):
    """Load HumanEval problems from the official JSONL file."""
    problems = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            problems.append({
                "task_id": data["task_id"],
                "prompt": data["prompt"],  # Function signature + docstring
                "test": data["test"],      # Unit tests
                "entry_point": data["entry_point"]
            })
            if limit and len(problems) >= limit:
                break
    print(f"📚 Loaded {len(problems)} problems from HumanEval.")
    return problems


def solution_exists(problem, model_name):
    """Check if a solution JSON file already exists for this problem and model."""
    output_dir = Path(f"data/raw/{model_name}")
    problem_id = problem["task_id"].replace("/", "_")
    filename = output_dir / f"{problem_id}.json"
    return filename.exists()

def save_generation(problem, model_name, generated_code, config):
    """Save the generated code to a structured JSON file."""
    # Save directly to model_name folder without date subfolder
    output_dir = Path(f"data/raw/{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a safe filename
    problem_id = problem["task_id"].replace("/", "_")
    filename = output_dir / f"{problem_id}.json"

    model_config_to_save = config['apis']['google'].copy()  # Make a copy
    model_config_to_save.pop('api_key', None)  # Remove the 'api_key' if it exists
    
    # Prepare the data to save
    data = {
        "timestamp": datetime.now().isoformat(),
        "problem_id": problem["task_id"],
        "problem_prompt": problem["prompt"],
        "model_name": model_name,
        "model_config": model_config_to_save,  # Save which model was used
        "generated_code": generated_code,
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"   Saved: {filename}")
    return filename

def main():
    print("Starting Daily Dataset Generation for Thesis")
    print("=" * 50)
    
    # 1. Load configuration
    config = load_config()
    google_config = config['apis']['google']
    print(f"Loaded config for model: {google_config['model']}")
    
    # 2. Load problems
    problems = load_real_humaneval("data/humaneval/HumanEval.jsonl", limit=164)
    
    print(f"Loaded {len(problems)} HumanEval problems")
    model_name = "google_gemini"
    
    # 3. Generate and save for each problem
    for i, problem in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] Processing: {problem['task_id']}")

        if solution_exists(problem, model_name):
            print(f"   Skipping: Solution already exists.")
            continue 
        
        # Create a clear prompt for the model
        prompt = f"""Write a complete and correct Python function that solves the following problem.

Problem signature and docstring:
{problem['prompt']}

Instructions:
1. Return ONLY the Python function code.
2. Do not include any explanations, comments about the code, or example usage.
3. Ensure the code is syntactically correct and follows the function signature exactly."""
        
        # Generate code
        generated_code = generate_with_google(prompt, google_config)
        
        # Save the result
        save_generation(problem, model_name, generated_code, config)
        
        # PAUSE to respect API rate limits
        time.sleep(15)
    
    print("\n" + "=" * 50)
    print("Dataset generation complete!")
    print(f"All data saved to: data/raw/{model_name}/")

if __name__ == "__main__":
    main()