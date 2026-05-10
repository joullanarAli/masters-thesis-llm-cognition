# evaluation_ui/app.py
import json
import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session
from pathlib import Path
import uuid
import random

app = Flask(__name__)
app.secret_key = '55c77a1d696e85b209f7e2050f52aac3e3182dbc0d6fc99849b5df6e64e0a954'

# Paths to BOTH model datasets
PROJECT_ROOT = Path(__file__).parent.parent
DATASETS = {
    "gemini": PROJECT_ROOT / 'data' / 'raw' / 'google_gemini',
    "groq": PROJECT_ROOT / 'data' / 'raw' / 'groq'
}
MASTER_DATASET_PATH = PROJECT_ROOT / 'data' / 'master_dataset.jsonl'
EVALUATIONS_PATH = Path(__file__).parent / 'evaluations.json'


def load_problems(model_name):
    """Load problems for a specific model."""
    dataset_path = DATASETS.get(model_name)
    if not dataset_path or not dataset_path.exists():
        return []
    
    problems = []
    for json_file in dataset_path.glob('*.json'):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            problem_id = int(data['problem_id'].split('/')[1])
            problems.append((problem_id, data))
    problems.sort(key=lambda x: x[0])
    return [p[1] for p in problems]


def load_all_problems():
    """Load problems from both models and pair them by problem_id."""
    gemini_problems = {p['problem_id']: p for p in load_problems("gemini")}
    groq_problems = {p['problem_id']: p for p in load_problems("groq")}
    
    # Get all problem IDs that exist in BOTH datasets
    common_ids = set(gemini_problems.keys()) & set(groq_problems.keys())
    
    paired_problems = []
    for problem_id in sorted(common_ids, key=lambda x: int(x.split('/')[1])):
        paired_problems.append({
            'problem_id': problem_id,
            'problem_prompt': gemini_problems[problem_id]['problem_prompt'],
            'gemini_code': gemini_problems[problem_id]['generated_code'],
            'groq_code': groq_problems[problem_id]['generated_code']
        })
    
    return paired_problems


def load_existing_evaluations(evaluator_id):
    """Load existing evaluations for this evaluator."""
    if not MASTER_DATASET_PATH.exists():
        return {}
    
    evaluations = {}
    with open(MASTER_DATASET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if data.get('evaluator_metadata', {}).get('evaluator_id') == evaluator_id:
                    key = f"{data['problem_id']}_{data['model_name']}"
                    evaluations[key] = data.get('human_evaluation', {})
    return evaluations


@app.route('/')
def index():
    if not session.get('evaluator_id'):
        return redirect(url_for('register'))
    
    problems = load_all_problems()
    evaluator_id = session.get('evaluator_id')
    existing = load_existing_evaluations(evaluator_id)
    
    # Count completed evaluations (both models for same problem count as one pair)
    completed_pairs = 0
    for problem in problems:
        gemini_key = f"{problem['problem_id']}_google_gemini"
        groq_key = f"{problem['problem_id']}_groq"
        if gemini_key in existing and groq_key in existing:
            completed_pairs += 1
    
    return render_template('index.html', 
                         total=len(problems), 
                         evaluated=completed_pairs,
                         session_date=datetime.now().strftime("%Y-%m-%d"))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        session['evaluator_id'] = str(uuid.uuid4())
        session['evaluator_name'] = request.form.get('name', 'Anonymous')
        session['expertise_level'] = request.form.get('expertise_level')
        session['programming_years'] = int(request.form.get('programming_years', 0))
        session['ai_usage_frequency'] = request.form.get('ai_usage_frequency')
        session['languages'] = request.form.get('languages', '')
        session['start_time'] = datetime.now().isoformat()
        
        return redirect(url_for('index'))
    
    return render_template('register.html')


@app.route('/evaluate/<int:problem_index>', methods=['GET', 'POST'])
def evaluate(problem_index):
    problems = load_all_problems()
    
    if problem_index < 0 or problem_index >= len(problems):
        return redirect(url_for('index'))
    
    problem = problems[problem_index]
    evaluator_id = session.get('evaluator_id')
    existing = load_existing_evaluations(evaluator_id)
    
    # Get existing evaluations for this problem (if any)
    gemini_key = f"{problem['problem_id']}_google_gemini"
    groq_key = f"{problem['problem_id']}_groq"
    
    prev_eval_gemini = existing.get(gemini_key, {})
    prev_eval_groq = existing.get(groq_key, {})
    
    # Randomize order to avoid bias (50% chance to swap)
    if 'order' not in session or problem_index == 0:
        session['order_' + problem['problem_id']] = random.choice(['gemini_left', 'groq_left'])
    
    order = session.get('order_' + problem['problem_id'], 'gemini_left')
    
    if request.method == 'POST':
        # Save Gemini evaluation
        gemini_eval = {
            "timestamp": datetime.now().isoformat(),
            "problem_id": problem["problem_id"],
            "problem_prompt": problem["problem_prompt"],
            "generated_code": problem["gemini_code"],
            "model_name": "google_gemini",
            "model_config": {"model": "gemini-3-flash-preview"},
            "evaluator_metadata": {
                "evaluator_id": session.get('evaluator_id'),
                "evaluator_name": session.get('evaluator_name', 'Anonymous'),
                "expertise_level": session.get('expertise_level'),
                "programming_years": session.get('programming_years'),
                "ai_usage_frequency": session.get('ai_usage_frequency'),
                "languages": session.get('languages'),
                "session_start_time": session.get('start_time'),
                "ip_address": request.remote_addr,
            },
            "human_evaluation": {
                "rated": True,
                "correctness": int(request.form['gemini_correctness']),
                "readability": int(request.form['gemini_readability']),
                "reliability": int(request.form['gemini_reliability']),
                "intention_to_use": request.form['gemini_intention'],
                "comments": request.form.get('gemini_comments', ''),
                "evaluator_id": session.get('evaluator_id')
            }
        }
        
        # Save Groq evaluation
        groq_eval = {
            "timestamp": datetime.now().isoformat(),
            "problem_id": problem["problem_id"],
            "problem_prompt": problem["problem_prompt"],
            "generated_code": problem["groq_code"],
            "model_name": "groq",
            "model_config": {"model": "llama-3.1-70b-versatile"},
            "evaluator_metadata": gemini_eval["evaluator_metadata"],  # Same metadata
            "human_evaluation": {
                "rated": True,
                "correctness": int(request.form['groq_correctness']),
                "readability": int(request.form['groq_readability']),
                "reliability": int(request.form['groq_reliability']),
                "intention_to_use": request.form['groq_intention'],
                "comments": request.form.get('groq_comments', ''),
                "evaluator_id": session.get('evaluator_id')
            }
        }
        
        # Append to master_dataset.jsonl
        MASTER_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        with open(MASTER_DATASET_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(gemini_eval, ensure_ascii=False) + '\n')
            f.write(json.dumps(groq_eval, ensure_ascii=False) + '\n')
        
        # Go to next problem
        next_index = problem_index + 1
        if next_index >= len(problems):
            return redirect(url_for('complete'))
        return redirect(url_for('evaluate', problem_index=next_index))
    
    return render_template('evaluate.html',
                         problem=problem,
                         problem_index=problem_index,
                         total=len(problems),
                         order=order,
                         prev_gemini=prev_eval_gemini,
                         prev_groq=prev_eval_groq,
                         gemini_model="Google Gemini",
                         groq_model="Groq (LLaMA 3.1)")


@app.route('/complete')
def complete():
    return render_template('complete.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)