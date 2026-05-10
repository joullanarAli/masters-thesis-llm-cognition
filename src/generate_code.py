# # Main script to run

# #!/usr/bin/env python3
# """
# MAIN SCRIPT: Daily Code Generation for Thesis Dataset
# Run this to generate code for HumanEval problems.
# """
# import os
# import sys
# import yaml
# import json
# import time
# from datetime import datetime
# from pathlib import Path

# # Add src to the Python path so we can import our module
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# # Import your API client function
# from api_clients import generate_with_google

# def load_config():
#     """Load API keys and configuration from the YAML file."""
#     config_path = Path(__file__).parent.parent / 'configs' / 'api_keys.yaml'
#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)
#     return config

# # def load_humaneval_problems(limit=5):
# #     """
# #     Load a sample of HumanEval problems.
# #     START WITH limit=5 to test, then increase to 50, 100, etc.
# #     """
# #     # SAMPLE DATA - Replace this later with the full HumanEval.jsonl
# #     sample_problems = [
# #         {
# #             "task_id": "HumanEval/0",
# #             "prompt": "from typing import List\n\ndef add_two_numbers(a: int, b: int) -> int:\n    \"\"\"Return the sum of two integers.\"\"\"\n",
# #         },
# #         {
# #             "task_id": "HumanEval/1",
# #             "prompt": "def reverse_string(s: str) -> str:\n    \"\"\"Return the reversed version of the input string.\"\"\"\n",
# #         },
# #         {
# #             "task_id": "HumanEval/2",
# #             "prompt": "def find_max(numbers: List[int]) -> int:\n    \"\"\"Return the maximum number in a list.\"\"\"\n",
# #         },
# #     ]
# #     return sample_problems[:limit]



# def load_real_humaneval(file_path: str, limit: int = None):
#     """Load HumanEval problems from the official JSONL file."""
#     problems = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             data = json.loads(line)
#             problems.append({
#                 "task_id": data["task_id"],
#                 "prompt": data["prompt"],  # Function signature + docstring
#                 "test": data["test"],      # Unit tests
#                 "entry_point": data["entry_point"]
#             })
#             if limit and len(problems) >= limit:
#                 break
#     print(f"📚 Loaded {len(problems)} problems from HumanEval.")
#     return problems


# def solution_exists(problem, model_name):
#     """Check if a solution JSON file already exists for this problem and model."""
#     output_dir = Path(f"data/raw/{model_name}")
#     problem_id = problem["task_id"].replace("/", "_")
#     filename = output_dir / f"{problem_id}.json"
#     return filename.exists()

# def save_generation(problem, model_name, generated_code, config):
#     """Save the generated code to a structured JSON file."""
#     # Save directly to model_name folder without date subfolder
#     output_dir = Path(f"data/raw/{model_name}")
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     # Create a safe filename
#     problem_id = problem["task_id"].replace("/", "_")
#     filename = output_dir / f"{problem_id}.json"

#     model_config_to_save = config['apis']['google'].copy()  # Make a copy
#     model_config_to_save.pop('api_key', None)  # Remove the 'api_key' if it exists
    
#     # Prepare the data to save
#     data = {
#         "timestamp": datetime.now().isoformat(),
#         "problem_id": problem["task_id"],
#         "problem_prompt": problem["prompt"],
#         "model_name": model_name,
#         "model_config": model_config_to_save,  # Save which model was used
#         "generated_code": generated_code,
#     }
    
#     with open(filename, 'w') as f:
#         json.dump(data, f, indent=2)
    
#     print(f"   Saved: {filename}")
#     return filename

# def main():
#     print("Starting Daily Dataset Generation for Thesis")
#     print("=" * 50)
    
#     # 1. Load configuration
#     config = load_config()
#     google_config = config['apis']['google']
#     print(f"Loaded config for model: {google_config['model']}")
    
#     # 2. Load problems
#     problems = load_real_humaneval("data/humaneval/HumanEval.jsonl", limit=164)
    
#     print(f"Loaded {len(problems)} HumanEval problems")
#     model_name = "google_gemini"
    
#     # 3. Generate and save for each problem
#     for i, problem in enumerate(problems):
#         print(f"\n[{i+1}/{len(problems)}] Processing: {problem['task_id']}")

#         if solution_exists(problem, model_name):
#             print(f"   Skipping: Solution already exists.")
#             continue 
        
#         # Create a clear prompt for the model
#         prompt = f"""Write a complete and correct Python function that solves the following problem.

# Problem signature and docstring:
# {problem['prompt']}

# Instructions:
# 1. Return ONLY the Python function code.
# 2. Do not include any explanations, comments about the code, or example usage.
# 3. Ensure the code is syntactically correct and follows the function signature exactly."""
        
#         # Generate code
#         generated_code = generate_with_google(prompt, google_config)
        
#         # Save the result
#         save_generation(problem, model_name, generated_code, config)
        
#         # PAUSE to respect API rate limits
#         time.sleep(15)
    
#     print("\n" + "=" * 50)
#     print("Dataset generation complete!")
#     print(f"All data saved to: data/raw/{model_name}/")

# if __name__ == "__main__":
#     main()
# Updated generate_code.py (add model selection)

#!/usr/bin/env python3
"""
MAIN SCRIPT: Daily Code Generation for Thesis Dataset
Run this to generate code for HumanEval problems using multiple models.
"""
import os
import sys
import yaml
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import ALL your API clients
from api_clients import generate_with_google, generate_with_groq, generate_with_qwen, generate_with_deepseek, generate_with_codellama, generate_with_codegen, generate_with_phi2

# Map model names to their generation functions and config keys
MODEL_REGISTRY = {
    "google_gemini": {
        "function": generate_with_google,
        "config_key": "google",
        "needs_hf": False
    },
    "qwen_coder": {
        "function": generate_with_qwen,
        "config_key": "qwen",
        "needs_hf": True
    },
    "deepseek_coder": {
        "function": generate_with_deepseek,
        "config_key": "deepseek",
        "needs_hf": True
    },
    "codellama": {
        "function": generate_with_codellama,
        "config_key": "codellama",
        "needs_hf": True
    },
    "codegen": {  # Replace qwen_coder with codegen
        "function": generate_with_codegen,
        "config_key": "codegen",
        "needs_hf": True
    },
    "phi2": {  # Replace codegen with phi2
        "function": generate_with_phi2,
        "config_key": "phi2",
        "needs_hf": True
    },
    "groq": {  # Add Groq
        "function": generate_with_groq,
        "config_key": "groq",
        "needs_hf": False
    },
}

def load_config():
    """Load API keys and configuration from the YAML file."""
    config_path = Path(__file__).parent.parent / 'configs' / 'api_keys.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_real_humaneval(file_path: str, limit: int = None):
    """Load HumanEval problems from the official JSONL file."""
    problems = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            problems.append({
                "task_id": data["task_id"],
                "prompt": data["prompt"],
                "test": data["test"],
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

def save_generation(problem, model_name, generated_code, model_config):
    """Save the generated code to a structured JSON file."""
    output_dir = Path(f"data/raw/{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    problem_id = problem["task_id"].replace("/", "_")
    filename = output_dir / f"{problem_id}.json"
    
    # Clean config for saving (remove sensitive info)
    config_to_save = model_config.copy()
    config_to_save.pop('api_key', None)
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "problem_id": problem["task_id"],
        "problem_prompt": problem["prompt"],
        "model_name": model_name,
        "model_config": config_to_save,
        "generated_code": generated_code
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"   ✓ Saved: {filename}")
    return filename

def generate_for_model(model_name, problems, config, limit=None, skip_existing=True):
    """Generate code for a specific model across all problems."""
    print(f"\n{'='*60}")
    print(f"🎯 Generating with model: {model_name}")
    print(f"{'='*60}")
    
    model_info = MODEL_REGISTRY[model_name]
    generate_func = model_info["function"]
    config_key = model_info["config_key"]
    
    # Get model-specific config
    model_config = config['apis'].get(config_key, {})
    if not model_config:
        print(f"⚠️ Warning: No config found for {config_key} in api_keys.yaml")
        print(f"   Using default config")
        model_config = {"model": model_name, "temperature": 0.2, "max_tokens": 512}
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for i, problem in enumerate(problems):
        if limit and i >= limit:
            break
            
        print(f"\n[{i+1}/{min(len(problems), limit or len(problems))}] Processing: {problem['task_id']}")
        
        if skip_existing and solution_exists(problem, model_name):
            print(f"   ⏭️ Skipping: Solution already exists.")
            skip_count += 1
            continue
        
        # Format prompt for model
        prompt = f"""Write a complete and correct Python function that solves the following problem.

            Problem signature and docstring:
            {problem['prompt']}

            Instructions:
            1. Return ONLY the Python function code.
            2. Do not include any explanations, comments about the code, or example usage.
            3. Ensure the code is syntactically correct and follows the function signature exactly."""
        
        try:
            # Generate code
            generated_code = generate_func(prompt, model_config)
            
            # Check for errors
            if generated_code.startswith("# API Error") or generated_code.startswith("# ERROR"):
                print(f"   ❌ Generation failed: {generated_code[:80]}")
                fail_count += 1
            else:
                # Save the result
                save_generation(problem, model_name, generated_code, model_config)
                success_count += 1
                
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            fail_count += 1
        
        # Pause to respect rate limits (shorter for local models)
        if model_info["needs_hf"]:
            time.sleep(2)  # Local models don't need long pauses
        else:
            time.sleep(15)  # API models need rate limit breathing room
    
    print(f"\n📊 Summary for {model_name}:")
    print(f"   ✅ Success: {success_count}")
    print(f"   ⏭️ Skipped: {skip_count}")
    print(f"   ❌ Failed: {fail_count}")
    
    return success_count

def main():
    parser = argparse.ArgumentParser(description="Generate code solutions using multiple LLMs")
    parser.add_argument("--model", type=str, choices=list(MODEL_REGISTRY.keys()), 
                        help="Specific model to use (default: all)")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Number of problems to process (default: all)")
    parser.add_argument("--force", action="store_true", 
                        help="Regenerate even if solution exists")
    
    args = parser.parse_args()
    
    print("🚀 Starting Multi-Model Dataset Generation for Thesis")
    print("=" * 60)
    
    # 1. Load configuration
    config = load_config()
    
    # 2. Load problems
    problems = load_real_humaneval("data/humaneval/HumanEval.jsonl", limit=args.limit)
    
    # 3. Determine which models to run
    if args.model:
        models_to_run = [args.model]
    else:
        models_to_run = list(MODEL_REGISTRY.keys())
    
    print(f"\n📋 Models to generate: {', '.join(models_to_run)}")
    print(f"📊 Total problems: {len(problems)}")
    print(f"🔄 Force regenerate: {args.force}")
    
    # 4. Generate for each model
    total_success = 0
    for model_name in models_to_run:
        success = generate_for_model(
            model_name, 
            problems, 
            config, 
            limit=args.limit,
            skip_existing=not args.force
        )
        total_success += success
    
    print("\n" + "=" * 60)
    print("🎉 Dataset generation complete!")
    print(f"📁 Data saved to: data/raw/[model_name]/")
    print("=" * 60)

if __name__ == "__main__":
    main()