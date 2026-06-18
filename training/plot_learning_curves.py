"""
Plot Learning Curves to Detect Overfitting/Underfitting
Shows Training, Validation, and Test curves
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.load_data import load_master_dataset
from train_models import prepare_features

# Create results directory
RESULTS_DIR = Path(__file__).parent / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def plot_learning_curve_with_test(X_train, y_train, X_val, y_val, X_test, y_test, model_name):
    """
    Plot learning curve showing Training, Validation, and Test accuracy.
    """
    
    print(f"\nGenerating Learning Curve for {model_name}...")
    
    # Use train sizes from 10% to 100%
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = []
    val_scores = []
    test_scores = []
    
    for size in train_sizes:
        # Sample from training data
        n_samples = int(len(X_train) * size)
        X_sub = X_train[:n_samples]
        y_sub = y_train[:n_samples]
        
        # Train model
        model_copy = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42
        )
        model_copy.fit(X_sub, y_sub)
        
        # Evaluate
        train_acc = model_copy.score(X_sub, y_sub)
        val_acc = model_copy.score(X_val, y_val)
        test_acc = model_copy.score(X_test, y_test)
        
        train_scores.append(train_acc)
        val_scores.append(val_acc)
        test_scores.append(test_acc)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_axis = train_sizes * len(X_train)
    
    ax.plot(x_axis, train_scores, 'o-', color='blue', 
            label='Training Accuracy', linewidth=2)
    ax.plot(x_axis, val_scores, 'o-', color='orange', 
            label='Validation Accuracy', linewidth=2)
    ax.plot(x_axis, test_scores, 'o-', color='green', 
            label='Test Accuracy', linewidth=2)
    
    ax.set_xlabel('Training Examples', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Learning Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.7, 1.0)
    
    # Add final values annotation
    final_train = train_scores[-1]
    final_val = val_scores[-1]
    final_test = test_scores[-1]
    
    info_text = f'Final: Train={final_train:.3f}, Val={final_val:.3f}, Test={final_test:.3f}'
    ax.annotate(info_text, 
                xy=(0.5, 0.85), xycoords='axes fraction',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black'))
    
    # Determine overfitting/underfitting
    train_val_gap = final_train - final_val
    train_test_gap = final_train - final_test
    
    if train_val_gap > 0.10 or train_test_gap > 0.10:
        status = "Overfitting Detected"
        color = 'red'
    elif final_val < 0.75 or final_test < 0.75:
        status = "Underfitting Detected"
        color = 'orange'
    else:
        status = "Good Fit"
        color = 'green'
    
    ax.text(0.5, 0.78, f'Status: {status}', 
            transform=ax.transAxes, fontsize=11, fontweight='bold', color=color)
    ax.text(0.5, 0.73, f'Train-Val Gap: {train_val_gap:.3f}, Train-Test Gap: {train_test_gap:.3f}', 
            transform=ax.transAxes, fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'learning_curve_{model_name.lower().replace(" ", "_")}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Learning curve saved")
    print(f"   Final Training Accuracy:   {final_train:.4f}")
    print(f"   Final Validation Accuracy: {final_val:.4f}")
    print(f"   Final Test Accuracy:       {final_test:.4f}")
    print(f"   Train-Val Gap: {train_val_gap:.4f}")
    print(f"   Train-Test Gap: {train_test_gap:.4f}")
    print(f"   Status: {status}")
    
    return train_scores, val_scores, test_scores

def plot_training_validation_comparison(X_train, y_train, X_val, y_val, X_test, y_test, model_name):
    """
    Plot bar chart comparing Training, Validation, and Test performance.
    """
    
    print(f"\nGenerating Comparison Bar Chart for {model_name}...")
    
    # Train model on full training set
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    test_acc = model.score(X_test, y_test)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(['Training', 'Validation', 'Test'], 
                  [train_acc, val_acc, test_acc],
                  color=['#4285f4', '#fbbc04', '#34a853'],
                  edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Performance Comparison - {model_name}', fontsize=14, fontweight='bold')
    ax.set_ylim(0.7, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars, [train_acc, val_acc, test_acc]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f} ({acc*100:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'performance_comparison_{model_name.lower().replace(" ", "_")}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Performance comparison saved")
    print(f"   Training:   {train_acc:.4f} ({train_acc*100:.1f}%)")
    print(f"   Validation: {val_acc:.4f} ({val_acc*100:.1f}%)")
    print(f"   Test:       {test_acc:.4f} ({test_acc*100:.1f}%)")

def plot_validation_curve(model, X_train, y_train, X_val, y_val, param_name, param_range, model_name):
    """
    Plot validation curve for a specific hyperparameter.
    """
    
    print(f"\nGenerating Validation Curve for {model_name} - {param_name}...")
    
    train_scores = []
    val_scores = []
    
    for param in param_range:
        # Create model with specific parameter
        if param_name == 'max_depth':
            model_copy = RandomForestClassifier(
                n_estimators=200,
                max_depth=param,
                min_samples_split=20,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42
            )
        elif param_name == 'n_estimators':
            model_copy = RandomForestClassifier(
                n_estimators=param,
                max_depth=15,
                min_samples_split=20,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42
            )
        else:
            model_copy = RandomForestClassifier(random_state=42)
        
        # Train and evaluate
        model_copy.fit(X_train, y_train)
        train_acc = model_copy.score(X_train, y_train)
        val_acc = model_copy.score(X_val, y_val)
        
        train_scores.append(train_acc)
        val_scores.append(val_acc)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert params to strings for x-axis
    x_labels = [str(p) if p is not None else 'None' for p in param_range]
    x_pos = np.arange(len(x_labels))
    
    ax.plot(x_pos, train_scores, 'o-', color='blue', label='Training Accuracy', linewidth=2)
    ax.plot(x_pos, val_scores, 'o-', color='orange', label='Validation Accuracy', linewidth=2)
    
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Validation Curve - {model_name} ({param_name})', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0.7, 1.0)
    
    # Find best parameter
    best_idx = np.argmax(val_scores)
    best_param = param_range[best_idx]
    best_score = val_scores[best_idx]
    
    ax.axvline(x=best_idx, color='red', linestyle='--', alpha=0.7, 
               label=f'Best: {best_param} (acc={best_score:.3f})')
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'validation_curve_{model_name.lower().replace(" ", "_")}_{param_name}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Validation curve saved")
    print(f"   Best {param_name}: {best_param} (accuracy: {best_score:.4f})")
    
    return best_param, best_score

def main():
    print("="*70)
    print("LEARNING CURVE ANALYSIS")
    print("="*70)
    print("Checking for Overfitting/Underfitting\n")
    
    # Load data
    df = load_master_dataset('data/master_dataset.jsonl')
    X, y = prepare_features(df)
    
    # ============================================================
    # PROPER SPLIT: Train/Validation/Test (70/15/15)
    # ============================================================
    print("Splitting data (70/15/15)...")
    
    # First split: 70% train, 30% temporary (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    # Second split: 50% of temp = 15% validation, 50% of temp = 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"   Training:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Load best model
    model_path = Path(__file__).parent / 'models' / 'random_forest.pkl'
    
    if not model_path.exists():
        print("Model not found. Train first!")
        return
    
    best_model = joblib.load(model_path)
    print(f"\nLoaded model: {model_path.name}\n")
    
    # ============================================================
    # 1. Learning Curve (Training, Validation, Test)
    # ============================================================
    plot_learning_curve_with_test(
        X_train, y_train, X_val, y_val, X_test, y_test, 
        "Random Forest"
    )
    
    # ============================================================
    # 2. Performance Comparison Bar Chart
    # ============================================================
    plot_training_validation_comparison(
        X_train, y_train, X_val, y_val, X_test, y_test,
        "Random Forest"
    )
    
    # ============================================================
    # 3. Validation Curve - max_depth
    # ============================================================
    plot_validation_curve(
        best_model, X_train, y_train, X_val, y_val,
        param_name='max_depth',
        param_range=[3, 5, 7, 10, 15, 20, None],
        model_name='Random Forest'
    )
    
    # ============================================================
    # 4. Validation Curve - n_estimators
    # ============================================================
    plot_validation_curve(
        best_model, X_train, y_train, X_val, y_val,
        param_name='n_estimators',
        param_range=[50, 100, 150, 200, 300],
        model_name='Random Forest'
    )
    
    # ============================================================
    # 5. Summary
    # ============================================================
    print("\n" + "="*70)
    print("SUMMARY - OVERFITTING ANALYSIS")
    print("="*70)
    
    # Get final metrics
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    test_acc = model.score(X_test, y_test)
    
    print(f"""
    Performance Summary:
    
    ┌─────────────────────────────────────────────────────────────┐
    │  Dataset        │  Accuracy  │  Gap from Training         │
    ├─────────────────────────────────────────────────────────────┤
    │  Training       │  {train_acc*100:.1f}%    │  -                          │
    │  Validation     │  {val_acc*100:.1f}%    │  {train_acc - val_acc:.2%}         │
    │  Test           │  {test_acc*100:.1f}%    │  {train_acc - test_acc:.2%}         │
    └─────────────────────────────────────────────────────────────┘
    
    Conclusion:
       - Training vs Validation Gap: {train_acc - val_acc:.2%}
       - Training vs Test Gap: {train_acc - test_acc:.2%}
       - Status: {'Good Fit' if train_acc - test_acc < 0.05 else '⚠️ Check'}
    """)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nFigures saved to: {FIGURES_DIR}")

if __name__ == "__main__":
    main()