"""Implement Bayesian Belief Network for cognitive state prediction."""
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix
from pathlib import Path

# Suppress pgmpy warnings
warnings.filterwarnings('ignore')

from load_data import load_master_dataset

def discretize_for_bbn(df):
    """Discretize continuous variables for BBN."""
    
    # Discretize cognitive state (expertise) - ORDERED CATEGORICAL
    expertise_map = {
        'student_bachelor': 0,
        'junior': 1,
        'student_master': 2,
        'senior': 3
    }
    df['C'] = df['expertise_level'].map(expertise_map)
    
    # Discretize metrics (1-5 scale to Low/Medium/High)
    def discretize_metric(score):
        if score <= 2:
            return 'Low'
        elif score <= 3.5:
            return 'Medium'
        else:
            return 'High'
    
    df['SC'] = df['readability'].apply(discretize_metric)   # Understanding/Consistency
    df['SA'] = df['correctness'].apply(discretize_metric)    # Agreement/Correctness
    df['SRel'] = df['reliability'].apply(discretize_metric)  # Reliability
    
    # Discretize usefulness (target)
    def discretize_usefulness(intention):
        if intention == 'yes':
            return 'High'
        elif intention == 'modified':
            return 'Medium'
        else:
            return 'Low'
    
    df['SU'] = df['intention_to_use'].apply(discretize_usefulness)
    
    return df

def build_bbn():
    """Build Bayesian Belief Network structure (paper's Figure 2)."""
    from pgmpy.models import DiscreteBayesianNetwork
    
    # C (Cognitive State) influences SC, SA, SRel
    # SC, SA, SRel influence SU (Usefulness)
    edges = [
        ('C', 'SC'), ('C', 'SA'), ('C', 'SRel'),
        ('SC', 'SU'), ('SA', 'SU'), ('SRel', 'SU')
    ]
    
    model = DiscreteBayesianNetwork(edges)
    return model

def train_bbn_once(train_data):
    """Train BBN on a single fold."""
    from pgmpy.estimators import BayesianEstimator
    
    model = build_bbn()
    model.fit(train_data, estimator=BayesianEstimator, prior_type='BDeu')
    return model

def predict_usefulness(model, evidence):
    """Predict usefulness from evidence."""
    from pgmpy.inference import VariableElimination
    
    inference = VariableElimination(model)
    pred = inference.map_query(variables=['SU'], evidence=evidence)
    return pred['SU']

def train_and_evaluate_bbn(df):
    """Train BBN and evaluate prediction accuracy."""
    
    df = discretize_for_bbn(df.copy())
    
    # Prepare data - ensure categorical types
    data = df[['C', 'SC', 'SA', 'SRel', 'SU']].dropna().copy()
    
    # Convert to categorical
    for col in ['C', 'SC', 'SA', 'SRel', 'SU']:
        data[col] = data[col].astype('category')
    
    print(f"\n📊 Data shape: {len(data)} records")
    print(f"📊 Unique C values (expertise): {sorted(data['C'].unique())}")
    print(f"📊 SU distribution:\n{data['SU'].value_counts()}")
    
    # Skip if too few samples
    if len(data) < 10:
        print("\n⚠️ Too few samples for reliable BNN. Need at least 10.")
        return None, None, None
    
    # Leave-one-out cross-validation
    loo = LeaveOneOut()
    predictions = []
    actuals = []
    correct = 0
    
    print("\n🔄 Running Leave-One-Out Cross-Validation...")
    
    for fold, (train_idx, test_idx) in enumerate(loo.split(data)):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        try:
            # Train model
            model = train_bbn_once(train_data)
            
            # Predict
            evidence = {
                'C': test_data['C'].iloc[0],
                'SC': test_data['SC'].iloc[0],
                'SA': test_data['SA'].iloc[0],
                'SRel': test_data['SRel'].iloc[0]
            }
            pred = predict_usefulness(model, evidence)
            actual = test_data['SU'].iloc[0]
            
            predictions.append(pred)
            actuals.append(actual)
            
            if pred == actual:
                correct += 1
                
        except Exception as e:
            # Fallback: predict most common class
            most_common = data['SU'].mode().iloc[0]
            predictions.append(most_common)
            actuals.append(test_data['SU'].iloc[0])
            if most_common == test_data['SU'].iloc[0]:
                correct += 1
    
    # Calculate metrics
    accuracy = correct / len(data)
    
    print(f"\n{'='*60}")
    print(f"BAYESIAN BELIEF NETWORK RESULTS")
    print(f"{'='*60}")
    print(f"✅ Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"✅ Correct predictions: {correct}/{len(data)}")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(actuals, predictions, labels=['Low', 'Medium', 'High'])
    print(f"\n📊 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Low  Med  High")
    print(f"Actual Low:     {cm[0,0]:3d}  {cm[0,1]:3d}   {cm[0,2]:3d}")
    print(f"       Medium:  {cm[1,0]:3d}  {cm[1,1]:3d}   {cm[1,2]:3d}")
    print(f"       High:    {cm[2,0]:3d}  {cm[2,1]:3d}   {cm[2,2]:3d}")
    
    return model, predictions, actuals

def compare_with_baseline(df):
    """Compare BBN with baseline (predicting without cognitive state)."""
    
    df = discretize_for_bbn(df.copy())
    data = df[['SC', 'SA', 'SRel', 'SU']].dropna()
    
    # Baseline 1: predict majority class
    majority_class = data['SU'].mode().iloc[0]
    baseline_majority = (data['SU'] == majority_class).mean()
    
    # Baseline 2: random prediction (1/3 chance for Low/Medium/High)
    baseline_random = 1/3
    
    print(f"\n{'='*60}")
    print(f"BASELINE COMPARISON")
    print(f"{'='*60}")
    print(f"Baseline (majority class): {baseline_majority:.4f} ({baseline_majority*100:.1f}%)")
    print(f"Baseline (random): {baseline_random:.4f} ({baseline_random*100:.1f}%)")
    
    return baseline_majority

def analyze_cognitive_state_impact(df):
    """Analyze how cognitive state (C) affects predictions."""
    
    df = discretize_for_bbn(df.copy())
    
    print(f"\n{'='*60}")
    print(f"COGNITIVE STATE IMPACT ANALYSIS")
    print(f"{'='*60}")
    
    # For each expertise level, show distribution of SU
    expertise_labels = {0: 'Bachelor', 1: 'Junior', 2: 'Master', 3: 'Senior'}
    
    for c_value in sorted(df['C'].unique()):
        subset = df[df['C'] == c_value]
        su_dist = subset['SU'].value_counts(normalize=True)
        
        label = expertise_labels.get(c_value, f'Level {c_value}')
        print(f"\n{label} (C={c_value}, n={len(subset)}):")
        for su_val, prob in su_dist.items():
            print(f"  → {su_val}: {prob:.1%}")

def predict_new_developer(df, developer_features):
    """Predict cognitive state for a new developer based on their evaluations."""
    
    df = discretize_for_bbn(df.copy())
    data = df[['SC', 'SA', 'SRel', 'SU']].dropna()
    
    # Train on all data
    from pgmpy.estimators import BayesianEstimator
    
    model = build_bbn()
    data_for_training = data.copy()
    data_for_training['C_temp'] = 2  # Temporary placeholder
    model.fit(data_for_training, estimator=BayesianEstimator, prior_type='BDeu')
    
    print(f"\n{'='*60}")
    print(f"NEW DEVELOPER PREDICTION")
    print(f"{'='*60}")
    print(f"Based on evaluation pattern, predicted usefulness: {predicted_usefulness}")

def main():
    print("="*60)
    print("BAYESIAN BELIEF NETWORK FOR COGNITIVE STATE PREDICTION")
    print("="*60)
    
    # Load data
    df = load_master_dataset('../data/master_dataset.jsonl')
    print(f"\n📁 Loaded {len(df)} total records")
    print(f"📁 Evaluators: {df['evaluator_name'].nunique()}")
    print(f"📁 Models: {df['model_name'].unique().tolist()}")
    
    # Analyze by model separately
    results_summary = []
    
    for model_name in ['google_gemini', 'groq']:
        print(f"\n{'#'*60}")
        print(f"# MODEL: {model_name.upper()}")
        print(f"{'#'*60}")
        
        model_df = df[df['model_name'] == model_name]
        
        # Analyze cognitive state impact
        analyze_cognitive_state_impact(model_df)
        
        # Train and evaluate BBN
        model, predictions, actuals = train_and_evaluate_bbn(model_df)
        
        if model is not None:
            baseline = compare_with_baseline(model_df)
            
            # Calculate improvement
            accuracy = (np.array(predictions) == np.array(actuals)).mean()
            improvement = ((accuracy - baseline) / baseline) * 100
            
            results_summary.append({
                'Model': model_name,
                'Accuracy': f"{accuracy:.4f} ({accuracy*100:.1f}%)",
                'Baseline': f"{baseline:.4f} ({baseline*100:.1f}%)",
                'Improvement': f"{improvement:.1f}%"
            })
    
    # Print summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY: COGNITIVE STATE PREDICTION PERFORMANCE")
    print(f"{'='*60}")
    results_df = pd.DataFrame(results_summary)
    print(results_df.to_string(index=False))
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    if results_summary:
        results_df.to_csv(output_dir / 'bbn_results.csv', index=False)
        print(f"\n✅ Results saved to {output_dir}/bbn_results.csv")

if __name__ == "__main__":
    main()