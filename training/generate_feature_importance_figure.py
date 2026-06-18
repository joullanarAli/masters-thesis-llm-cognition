# training/generate_feature_importance_figure.py
"""
Generate Feature Importance Figure from Saved Model
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.load_data import load_master_dataset

# Create results directories
RESULTS_DIR = Path(__file__).parent / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
MODELS_DIR = Path(__file__).parent / 'models'

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def prepare_features(df):
    """Prepare features for ML models from the master dataset."""
    
    features = pd.DataFrame()
    
    # Code quality features
    features['correctness'] = df['correctness']
    features['readability'] = df['readability']
    features['reliability'] = df['reliability']
    
    # Developer cognitive state features
    expertise_map = {
        'student_bachelor': 0,
        'junior': 1,
        'student_master': 2,
        'senior': 3,
        'student_phd': 2
    }
    features['expertise_level'] = df['expertise_level'].map(expertise_map)
    features['programming_years'] = df['programming_years']
    
    ai_map = {'never': 0, 'rarely': 1, 'monthly': 2, 'weekly': 3, 'daily': 4}
    features['ai_familiarity'] = df['ai_usage_frequency'].map(ai_map)
    
    features['is_gemini'] = (df['model_name'] == 'google_gemini').astype(int)
    features['expertise_x_gemini'] = features['expertise_level'] * features['is_gemini']
    features['problem_difficulty'] = df['problem_num'] / 164
    
    target = (df['intention_to_use'] == 'yes').astype(int)
    
    return features, target

def plot_feature_importance_from_model(model_path, feature_names, model_name):
    """Generate feature importance plot from saved model."""
    
    # Load model
    model = joblib.load(model_path)
    
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        # Print
        print(f"\nFeature Importance ({model_name}):")
        for _, row in importance.tail(5).iterrows():
            print(f"   {row['Feature']}: {row['Importance']*100:.2f}%")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance['Feature'], importance['Importance'], color='steelblue')
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Feature Importance - {model_name}')
        ax.grid(axis='x', alpha=0.3)
        
        # Add percentage labels
        for i, (_, row) in enumerate(importance.iterrows()):
            ax.text(row['Importance'] + 0.01, i, f"{row['Importance']*100:.1f}%", 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save
        save_path = FIGURES_DIR / f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ Figure saved to: {save_path}")
        return importance
    
    return None

def main():
    print("="*60)
    print("GENERATING FEATURE IMPORTANCE FIGURE")
    print("="*60)
    
    # Load data
    df = load_master_dataset('data/master_dataset.jsonl')
    X, y = prepare_features(df)
    feature_names = X.columns.tolist()
    
    # Generate for Random Forest
    model_path = MODELS_DIR / 'random_forest.pkl'
    if model_path.exists():
        print("\n📊 Generating Random Forest Feature Importance...")
        importance = plot_feature_importance_from_model(
            model_path, 
            feature_names, 
            "Random Forest"
        )
    else:
        print("❌ Random Forest model not found. Train it first.")
    
    # Generate for Gradient Boosting (optional)
    model_path = MODELS_DIR / 'gradient_boosting.pkl'
    if model_path.exists():
        print("\n📊 Generating Gradient Boosting Feature Importance...")
        importance = plot_feature_importance_from_model(
            model_path, 
            feature_names, 
            "Gradient Boosting"
        )

if __name__ == "__main__":
    main()