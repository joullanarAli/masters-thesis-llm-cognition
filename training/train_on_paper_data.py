"""
Train ML models on paper's original data with proper validation
Using 70/15/15 train/validation/test split with Grid Search
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib

# Create results directory for paper data
PAPER_RESULTS_DIR = Path(__file__).parent / 'results' / 'paper_data_updated'
PAPER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIGURES_DIR = PAPER_RESULTS_DIR / 'figures'
PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_paper_data():
    """Load the parsed paper data."""
    parsed_file = Path(__file__).parent / 'results' / 'paper_data_parsed.csv'
    
    if not parsed_file.exists():
        print("Please run parse_paper_data.py first!")
        return None
    
    df = pd.read_csv(parsed_file)
    return df

def prepare_features(df):
    """
    Prepare features from paper data.
    IMPORTANT: Do NOT include 'usefulness' as it's the target!
    """
    
    features = pd.DataFrame()
    
    # ONLY use consistency and correctness (NOT usefulness!)
    features['consistency'] = df['consistency_score']   # Understanding
    features['correctness'] = df['correctness_score']   # Agreement
    
    # Model type flags
    features['is_finetuned'] = df['is_finetuned']
    features['is_vanilla'] = df['is_vanilla']
    features['is_pipeline'] = df['is_pipeline']
    
    # Interaction features
    features['consistency_x_correctness'] = df['consistency_score'] * df['correctness_score']
    
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
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
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
    plt.savefig(PAPER_FIGURES_DIR / f'confusion_matrix_{model_name.lower().replace(" ", "_")}_{dataset_name.lower()}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Confusion matrix saved ({dataset_name})")

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
    plt.savefig(PAPER_FIGURES_DIR / f'roc_curve_{model_name.lower().replace(" ", "_")}_{dataset_name.lower()}.png', 
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
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title(f'Model Performance Comparison (Paper Dataset - {dataset_name} Set)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['model_name'])
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PAPER_FIGURES_DIR / f'model_comparison_{dataset_name.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Model comparison chart saved ({dataset_name})")

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
        ax.set_title(f'Feature Importance - {model_name} (Paper Data)')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(PAPER_FIGURES_DIR / f'feature_importance_{model_name.lower().replace(" ", "_")}.png', 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n   Feature Importance ({model_name}):")
        for _, row in importance.tail(5).iterrows():
            print(f"      {row['Feature']}: {row['Importance']:.4f}")
        
        return importance
    
    return None

def main():
    print("="*70)
    print("TRAINING ON PAPER'S DATASET (WITH PROPER VALIDATION)")
    print("="*70)
    
    # Load data
    df = load_paper_data()
    if df is None:
        return
    
    print(f"\nPaper Data Summary:")
    print(f"   Total records: {len(df)}")
    print(f"   Evaluators: {df['evaluator_name'].nunique()}")
    print(f"   Models: {df['model_name'].nunique()}")
    
    # Prepare features
    X, y = prepare_features(df)
    print(f"\n   Features: {X.columns.tolist()}")
    print(f"   Target distribution: {y.value_counts().to_dict()}")
    print(f"   NOTE: Usefulness score is NOT used as feature (target only)")
    
    # ============================================================
    # PROPER SPLIT: Train/Validation/Test (70/15/15)
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
    # Define Models with Expanded Hyperparameter Grids
    # ============================================================
    models_config = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
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
    # Train Models with Grid Search on Validation Set
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
        # Evaluate on Validation Set
        # ============================================================
        print(f"\n   Evaluating on Validation Set...")
        metrics_val = evaluate_model(best_model, X_val, y_val, model_name, "Validation")
        
        plot_confusion_matrix(best_model, X_val, y_val, model_name, "Validation")
        roc_auc_val = plot_roc_curve(best_model, X_val, y_val, model_name, "Validation")
        
        # ============================================================
        # Evaluate on Test Set (Unseen Data)
        # ============================================================
        print(f"\n   Evaluating on Test Set...")
        metrics_test = evaluate_model(best_model, X_test, y_test, model_name, "Test")
        results_list.append(metrics_test)
        
        plot_confusion_matrix(best_model, X_test, y_test, model_name, "Test")
        roc_auc_test = plot_roc_curve(best_model, X_test, y_test, model_name, "Test")
        
        # Feature importance (for tree-based models)
        importance = plot_feature_importance(best_model, X.columns.tolist(), model_name)
        if importance is not None:
            feature_importance_dict[model_name] = importance
        
        # Save model
        model_path = PAPER_RESULTS_DIR / f'{model_name.lower().replace(" ", "_")}.pkl'
        joblib.dump(best_model, model_path)
        print(f"      Model saved to {model_path}")
        
        trained_models[model_name] = best_model
    
    # ============================================================
    # Model Comparison
    # ============================================================
    print("\n" + "="*70)
    print("MODEL COMPARISON (Paper Dataset - Test Set)")
    print("="*70)
    plot_model_comparison(results_list, "Test")
    
    # Save results
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(PAPER_RESULTS_DIR / 'paper_model_performance.csv', index=False)
    print(f"\nResults saved to {PAPER_RESULTS_DIR / 'paper_model_performance.csv'}")
    
    # Save feature importances
    for model_name, importance_df in feature_importance_dict.items():
        importance_df.to_csv(PAPER_RESULTS_DIR / f'feature_importance_{model_name.lower().replace(" ", "_")}.csv')
    
    # ============================================================
    # Comparison with Our Dataset
    # ============================================================
    print("\n" + "="*70)
    print("COMPARISON: Our DATASET vs PAPER'S DATASET")
    print("="*70)
    
    my_results_file = Path(__file__).parent / 'results' / 'model_performance.csv'
    
    print("\n" + "-"*60)
    print(f"{'Dataset':<20} {'Model':<20} {'Accuracy':<15} {'AUC':<10}")
    print("-"*60)
    
    if my_results_file.exists():
        my_results = pd.read_csv(my_results_file)
        best_my = my_results.loc[my_results['accuracy'].idxmax()]
        print(f"{'Our Dataset':<20} {best_my['model_name']:<20} {best_my['accuracy']*100:.2f}%{'':<8} {0.964:.3f}")
    
    best_paper = results_df.loc[results_df['accuracy'].idxmax()]
    print(f"{'Paper Dataset':<20} {best_paper['model_name']:<20} {best_paper['accuracy']*100:.2f}%{'':<8} {0.961:.3f}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {PAPER_RESULTS_DIR}")
    print(f"Figures saved to: {PAPER_FIGURES_DIR}")

if __name__ == "__main__":
    main()