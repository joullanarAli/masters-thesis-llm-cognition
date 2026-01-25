# evaluation_ui/app.py
import json
import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session
from pathlib import Path

app = Flask(__name__)
app

# Path to your generated dataset
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / 'data' / 'raw' / '2026-01-25' / 'google_gemini'
EVALUATIONS_PATH = Path(__file__).parent / 'evaluations.json'
MASTER_DATASET_PATH = PROJECT_ROOT / 'data' / 'master_dataset.jsonl'


# Load all generated JSON files
def load_problems():
    problems = []
    for json_file in DATASET_PATH.glob('*.json'):
        with open(json_file, 'r') as f:
            data = json.load(f)
            # Extract the numeric ID for sorting
            problem_id = int(data['problem_id'].split('/')[1])
            problems.append((problem_id, data))
    # Sort by the numeric problem ID
    problems.sort(key=lambda x: x[0])
    return [p[1] for p in problems]  # Return only the data

# Load existing evaluations or start fresh
def load_evaluations():
    if EVALUATIONS_PATH.exists():
        with open(EVALUATIONS_PATH, 'r') as f:
            return json.load(f)
    return {}

@app.route('/')
def index():
    problems = load_problems()
    evaluations = load_evaluations()
    # Simple homepage showing progress
    evaluated_count = sum(1 for pid in evaluations if evaluations[pid].get('rated'))
    return render_template('index.html', total=len(problems), evaluated=evaluated_count)

@app.route('/evaluate/<int:problem_index>', methods=['GET', 'POST'])
def evaluate(problem_index):
    problems = load_problems()
    evaluations = load_evaluations()

    if problem_index < 0 or problem_index >= len(problems):
        return redirect(url_for('index'))

    problem = problems[problem_index]
    problem_id = problem['problem_id']

    if request.method == 'POST':
        merged_record = {
            "timestamp": datetime.now().isoformat(),
            "problem_id": problem["problem_id"],
            "problem_prompt": problem["problem_prompt"],
            "generated_code": problem["generated_code"],
            "model_name": problem["model_name"],
            "model_config": problem.get("model_config", {}),
            "human_evaluation": {
                "rated": True,
                "correctness": int(request.form['correctness']),
                "readability": int(request.form['readability']),
                "intention_to_use": request.form['intention_to_use'],
                "comments": request.form.get('comments', ''),
                "evaluator_id": session.get('evaluator_id', 'default')  # Track multiple evaluators
            }
        }

        # 2. Append to master dataset (JSON Lines format)
        MASTER_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MASTER_DATASET_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(merged_record, ensure_ascii=False) + '\n')
        
        # 3. Still save to evaluations.json for backward compatibility
        evaluations[problem_id] = merged_record["human_evaluation"]
        with open(EVALUATIONS_PATH, 'w') as f:
            json.dump(evaluations, f, indent=2)
        
        # Go to the next problem
        next_index = problem_index + 1
        if next_index >= len(problems):
            return redirect(url_for('index'))
        return redirect(url_for('evaluate', problem_index=next_index))

    # Check if this problem was already evaluated
    prev_eval = evaluations.get(problem_id, {})

    return render_template('evaluate.html',
                           problem=problem,
                           problem_index=problem_index,
                           total=len(problems),
                           prev_eval=prev_eval)

    

if __name__ == '__main__':
    app.run(debug=True, port=5000)