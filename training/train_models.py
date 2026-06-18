"""
Model Training for Predicting Developer Intention to Use AI Code
This extends the BBN analysis with modern machine learning approaches
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directory to path to import load_data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.load_data import load_master_dataset

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import joblib

# Set style for better plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Create results directories
RESULTS_DIR = Path(__file__).parent / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
MODELS_DIR = Path(__file__).parent / 'models'

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

def prepare_features(df):
    """Prepare features for ML models from the master dataset."""
    
    features = pd.DataFrame()
    
    # Code quality features (the evaluation scores)
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
    
    # AI usage frequency as numeric (proxy for AI familiarity/trust)
    ai_map = {'never': 0, 'rarely': 1, 'monthly': 2, 'weekly': 3, 'daily': 4}
    features['ai_familiarity'] = df['ai_usage_frequency'].map(ai_map)
    
    # Model type (Gemini=1, Groq=0)
    features['is_gemini'] = (df['model_name'] == 'google_gemini').astype(int)
    
    # Interaction features (how expertise interacts with model quality)
    features['expertise_x_gemini'] = features['expertise_level'] * features['is_gemini']
    
    # Add problem difficulty proxy (using problem number as simple proxy)
    features['problem_difficulty'] = df['problem_num'] / 164  # Normalized 0-1
    
    # Target: binary (1 = use as-is, 0 = modify or reject)
    target = (df['intention_to_use'] == 'yes').astype(int)
    
    return features, target

def train_model_with_gridsearch(model_name, model, param_grid, X_train, y_train):
    """Train model with grid search cross validation."""
    
    print(f"\n   Training {model_name}...")
    
    # Use StratifiedKFold for better cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=cv, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    
    print(f"      Best parameters: {grid_search.best_params_}")
    print(f"      Best CV accuracy: {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")
    
    # Print cross-validation results
    cv_results = pd.DataFrame(grid_search.cv_results_)
    print(f"      Mean CV score: {cv_results['mean_test_score'].mean():.4f}")
    print(f"      Std CV score: {cv_results['std_test_score'].mean():.4f}")
    
    return grid_search.best_estimator_, grid_search.best_score_, grid_search

def evaluate_model(model, X_test, y_test, model_name, dataset_name="Test"):
    """Evaluate model and print metrics."""
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n   {model_name} Results ({dataset_name} Set):")
    print(f"      Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"      Precision: {precision:.4f}")
    print(f"      Recall:    {recall:.4f}")
    print(f"      F1-Score:  {f1:.4f}")
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def plot_confusion_matrix(model, X_test, y_test, model_name, dataset_name="Test"):
    """Plot and save confusion matrix."""
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Modify/Reject', 'Use as-is'],
                yticklabels=['Modify/Reject', 'Use as-is'],
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name} ({dataset_name} Set)')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'confusion_matrix_{model_name.lower().replace(" ", "_")}_{dataset_name.lower()}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Confusion matrix saved ({dataset_name})")

def plot_feature_importance(model, feature_names, model_name):
    """Plot and save feature importance for tree-based models."""
    
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance['Feature'], importance['Importance'], color='steelblue')
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Feature Importance - {model_name}')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f'feature_importance_{model_name.lower().replace(" ", "_")}.png', 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n   Feature Importance ({model_name}):")
        for _, row in importance.tail(5).iterrows():
            print(f"      {row['Feature']}: {row['Importance']:.4f}")
        
        return importance
    
    return None

def plot_roc_curve(model, X_test, y_test, model_name, dataset_name="Test"):
    """Plot and save ROC curve."""
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name} ({dataset_name} Set)')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'roc_curve_{model_name.lower().replace(" ", "_")}_{dataset_name.lower()}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      ROC curve saved (AUC = {roc_auc:.3f}) ({dataset_name})")
    
    return roc_auc

def plot_model_comparison(results_list, dataset_name="Test"):
    """Plot comparison of all models."""
    
    results_df = pd.DataFrame(results_list)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(results_df['model_name']))
    width = 0.2
    
    colors = ['#4285f4', '#34a853', '#fbbc04', '#ea4335']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = results_df[metric].values
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=metric.capitalize(), color=color, alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title(f'Model Performance Comparison ({dataset_name} Set)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['model_name'])
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'model_comparison_{dataset_name.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Model comparison chart saved ({dataset_name})")

def predict_new_developer(model, developer_profile, code_scores):
    """Demo: Predict for a new developer."""
    
    features = pd.DataFrame({
        'correctness': [code_scores['correctness']],
        'readability': [code_scores['readability']],
        'reliability': [code_scores['reliability']],
        'expertise_level': [developer_profile['expertise']],
        'programming_years': [developer_profile['years']],
        'ai_familiarity': [developer_profile['ai_frequency']],
        'is_gemini': [code_scores['is_gemini']],
        'expertise_x_gemini': [developer_profile['expertise'] * code_scores['is_gemini']],
        'problem_difficulty': [code_scores.get('difficulty', 0.5)]
    })
    
    probability = model.predict_proba(features)[0, 1]
    prediction = "Use as-is" if probability >= 0.5 else "Modify or Reject"
    confidence = abs(probability - 0.5) * 2
    
    return {
        'probability': probability,
        'prediction': prediction,
        'confidence': confidence
    }

def main():
    print("="*70)
    print("MACHINE LEARNING MODEL TRAINING")
    print("="*70)
    print("Predicting Developer Intention to Use AI-Generated Code\n")
    
    # ============================================================
    # STEP 1: Load Data
    # ============================================================
    print("Loading data...")
    df = load_master_dataset('data/master_dataset.jsonl')
    print(f"   Loaded {len(df)} total records")
    print(f"   Evaluators: {df['evaluator_name'].nunique()}")
    print(f"   Models: {df['model_name'].unique().tolist()}")
    
    # Prepare features
    print("\nPreparing features...")
    X, y = prepare_features(df)
    print(f"   Features: {X.columns.tolist()}")
    print(f"   Target distribution: {y.value_counts().to_dict()}")
    
    # ============================================================
    # STEP 2: Train/Validation/Test Split (70/15/15)
    # ============================================================
    print("\n" + "="*70)
    print("DATA SPLIT (Train/Validation/Test)")
    print("="*70)
    
    # First split: 70% train, 30% temporary (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    # Second split: 50% of temp = 15% validation, 50% of temp = 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"\n   Training set:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test set:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    print(f"\n   Training target distribution:")
    print(f"      Use as-is:     {sum(y_train)} ({sum(y_train)/len(y_train)*100:.1f}%)")
    print(f"      Modify/Reject: {len(y_train)-sum(y_train)} ({(len(y_train)-sum(y_train))/len(y_train)*100:.1f}%)")
    
    # ============================================================
    # STEP 3: Define Models with Expanded Hyperparameter Grids
    # ============================================================
    models_config = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [50, 100, 150, 200, 300],
                'max_depth': [3, 5, 7, 10, 15, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 150, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5, 7],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['liblinear', 'saga']
            }
        }
    }
    
    # ============================================================
    # STEP 4: Train Models with Grid Search on Validation Set
    # ============================================================
    results_list = []
    trained_models = {}
    feature_importance_dict = {}
    
    print("\n" + "="*70)
    print("TRAINING MODELS (Grid Search with Validation Set)")
    print("="*70)
    
    for model_name, config in models_config.items():
        print(f"\n{'─'*50}")
        print(f"Training {model_name}...")
        print(f"{'─'*50}")
        
        # Train with grid search using validation set
        best_model, best_score, grid_search = train_model_with_gridsearch(
            model_name, config['model'], config['params'], X_train, y_train
        )
        
        # ============================================================
        # STEP 5: Evaluate on Validation Set
        # ============================================================
        print(f"\n   Evaluating on Validation Set...")
        metrics_val = evaluate_model(best_model, X_val, y_val, model_name, "Validation")
        
        # Plot confusion matrix (validation)
        plot_confusion_matrix(best_model, X_val, y_val, model_name, "Validation")
        
        # Plot ROC curve (validation)
        roc_auc_val = plot_roc_curve(best_model, X_val, y_val, model_name, "Validation")
        
        # ============================================================
        # STEP 6: Evaluate on Test Set (Unseen Data)
        # ============================================================
        print(f"\n   Evaluating on Test Set...")
        metrics_test = evaluate_model(best_model, X_test, y_test, model_name, "Test")
        results_list.append(metrics_test)
        
        # Plot confusion matrix (test)
        plot_confusion_matrix(best_model, X_test, y_test, model_name, "Test")
        
        # Plot ROC curve (test)
        roc_auc_test = plot_roc_curve(best_model, X_test, y_test, model_name, "Test")
        
        # Feature importance (for tree-based models)
        importance = plot_feature_importance(best_model, X.columns.tolist(), model_name)
        if importance is not None:
            feature_importance_dict[model_name] = importance
        
        # ============================================================
        # STEP 7: SAVE MODEL (FIXED!)
        # ============================================================
        model_path = MODELS_DIR / f'{model_name.lower().replace(" ", "_")}.pkl'
        joblib.dump(best_model, model_path)
        print(f"      Model saved to {model_path}")
        
        trained_models[model_name] = best_model
    
    # ============================================================
    # STEP 8: Model Comparison
    # ============================================================
    print("\n" + "="*70)
    print("MODEL COMPARISON (Test Set)")
    print("="*70)
    plot_model_comparison(results_list, "Test")
    
    # Save results
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(RESULTS_DIR / 'model_performance.csv', index=False)
    print(f"\nResults saved to {RESULTS_DIR / 'model_performance.csv'}")
    
    # Save feature importances
    for model_name, importance_df in feature_importance_dict.items():
        importance_df.to_csv(RESULTS_DIR / f'feature_importance_{model_name.lower().replace(" ", "_")}.csv')
    
    # ============================================================
    # STEP 9: Demo Predictions
    # ============================================================
    print("\n" + "="*70)
    print("DEMO: PREDICTIONS FOR NEW DEVELOPERS")
    print("="*70)
    
    # Use best model
    best_model_name = results_df.loc[results_df['accuracy'].idxmax(), 'model_name']
    best_model = trained_models[best_model_name]
    print(f"\nUsing best model: {best_model_name}\n")
    
    # Scenario 1: Junior developer with good Groq code
    result1 = predict_new_developer(
        best_model,
        developer_profile={'expertise': 1, 'years': 2, 'ai_frequency': 3},
        code_scores={'correctness': 5, 'readability': 4, 'reliability': 4, 'is_gemini': 0, 'difficulty': 0.5}
    )
    print("Scenario 1: Junior Developer + Good Groq Code")
    print(f"   - Prediction: {result1['prediction']}")
    print(f"   - Confidence: {result1['confidence']:.1%}")
    print(f"   - Probability of acceptance: {result1['probability']:.1%}")
    
    # Scenario 2: Senior developer with poor Groq code
    result2 = predict_new_developer(
        best_model,
        developer_profile={'expertise': 3, 'years': 8, 'ai_frequency': 4},
        code_scores={'correctness': 2, 'readability': 2, 'reliability': 1, 'is_gemini': 0, 'difficulty': 0.5}
    )
    print("\nScenario 2: Senior Developer + Poor Groq Code")
    print(f"   - Prediction: {result2['prediction']}")
    print(f"   - Confidence: {result2['confidence']:.1%}")
    print(f"   - Probability of acceptance: {result2['probability']:.1%}")
    
    # Scenario 3: Any developer with Gemini code
    result3 = predict_new_developer(
        best_model,
        developer_profile={'expertise': 2, 'years': 5, 'ai_frequency': 3},
        code_scores={'correctness': 5, 'readability': 5, 'reliability': 5, 'is_gemini': 1, 'difficulty': 0.5}
    )
    print("\nScenario 3: Any Developer + Gemini Code")
    print(f"   - Prediction: {result3['prediction']}")
    print(f"   - Confidence: {result3['confidence']:.1%}")
    print(f"   - Probability of acceptance: {result3['probability']:.1%}")
    
    # ============================================================
    # STEP 10: Summary
    # ============================================================
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Models saved to: {MODELS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    
    print("\n" + "="*70)
    print("BEST MODEL SUMMARY")
    print("="*70)
    best_row = results_df.loc[results_df['accuracy'].idxmax()]
    print(f"Best Model: {best_row['model_name']}")
    print(f"Test Accuracy: {best_row['accuracy']:.4f} ({best_row['accuracy']*100:.2f}%)")
    print(f"Test Precision: {best_row['precision']:.4f}")
    print(f"Test Recall: {best_row['recall']:.4f}")
    print(f"Test F1-Score: {best_row['f1_score']:.4f}")

if __name__ == "__main__":
    main()