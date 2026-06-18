"""
Replicate Kovalchuk et al. (2022) GEM Paper Methodology
Using Mutual Information, Correlation, and Linear Regression
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.load_data import load_master_dataset

# Create results directory
RESULTS_DIR = Path(__file__).parent / 'results' / 'kovalchuk_replication'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = RESULTS_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def prepare_perception_data(df):
    """
    Prepare data for Kovalchuk et al. (2022) style analysis.
    Their three perception dimensions:
    - Understanding = Readability (consistency)
    - Agreement = Correctness
    - Usefulness = Intention to Use
    """
    
    # Map to Kovalchuk's terminology
    perception_df = pd.DataFrame()
    
    # Understanding = how well the user understands the code (readability)
    perception_df['Understanding'] = df['readability']
    
    # Agreement = whether the code is correct (correctness)
    perception_df['Agreement'] = df['correctness']
    
    # Usefulness = intention to use (convert to numeric 1-5 scale)
    intention_map = {'yes': 5, 'modified': 3, 'no': 1}
    perception_df['Usefulness'] = df['intention_to_use'].map(intention_map)
    
    # Add model type as categorical (for separate analysis)
    perception_df['model_name'] = df['model_name']
    
    return perception_df

def compute_mutual_information(df):
    """
    Compute Mutual Information between perception features.
    This replicates Kovalchuk et al.'s MI analysis (Figure 2 in their paper).
    """
    
    print("\n" + "="*70)
    print("MUTUAL INFORMATION ANALYSIS (Replicating Kovalchuk et al. 2022)")
    print("="*70)
    
    # Features for MI calculation
    features = ['Understanding', 'Agreement', 'Usefulness']
    
    # Compute MI between all pairs
    mi_results = {}
    
    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features):
            if i < j:
                # Prepare data
                X = df[[feat1]].values.reshape(-1, 1)
                y = df[feat2].values
                
                # Compute mutual information
                mi = mutual_info_regression(X, y, random_state=42)[0]
                mi_results[f'{feat1} ↔ {feat2}'] = mi
    
    # Print results
    print("\nMutual Information Scores:")
    print("-"*40)
    for pair, mi_val in sorted(mi_results.items(), key=lambda x: x[1], reverse=True):
        print(f"   {pair}: {mi_val:.4f}")
    
    # Kovalchuk et al. found highest MI between Agree and Use
    print("\nComparison with Kovalchuk et al. (2022):")
    print("   They found highest MI between Agree ↔ Use")
    print("   Your highest MI:", max(mi_results, key=mi_results.get))
    
    return mi_results

def compute_correlations(df):
    """
    Compute Pearson correlation between perception features.
    This replicates Kovalchuk et al.'s correlation analysis.
    """
    
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS (Replicating Kovalchuk et al. 2022)")
    print("="*70)
    
    # Compute correlations
    features = ['Understanding', 'Agreement', 'Usefulness']
    corr_matrix = df[features].corr(method='pearson')
    
    print("\nCorrelation Matrix:")
    print("-"*40)
    print(corr_matrix.round(4))
    
    # Kovalchuk et al. found: Agree ↔ Use: r = 0.898
    agree_use_corr = corr_matrix.loc['Agreement', 'Usefulness']
    
    print(f"\nComparison with Kovalchuk et al. (2022):")
    print(f"   They found: Agree ↔ Use: r = 0.898")
    print(f"   Your result: Agree ↔ Use: r = {agree_use_corr:.4f}")
    
    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', 
                vmin=-1, vmax=1, center=0, ax=ax)
    ax.set_title('Correlation Matrix (Perception Features)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nCorrelation matrix saved to {FIGURES_DIR / 'correlation_matrix.png'}")
    
    return corr_matrix

def fit_linear_regression(df):
    """
    Fit linear regression model: Usefulness = β0 + β1*Agreement + β2*Understanding
    This replicates Kovalchuk et al.'s Equation: U = 0.8389A + 0.0764C + 0.0134
    """
    
    print("\n" + "="*70)
    print("LINEAR REGRESSION (Replicating Kovalchuk et al. 2022)")
    print("="*70)
    
    # Prepare data
    X = df[['Agreement', 'Understanding']].values
    y = df['Usefulness'].values
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    
    # Get coefficients
    beta_agreement = model.coef_[0]
    beta_understanding = model.coef_[1]
    intercept = model.intercept_
    
    print(f"\nRegression Equation:")
    print(f"   Usefulness = {beta_agreement:.4f} × Agreement + {beta_understanding:.4f} × Understanding + {intercept:.4f}")
    
    # Predictions and R²
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    print(f"\nModel Performance:")
    print(f"   R² = {r2:.4f}")
    print(f"   MAE = {mae:.4f}")
    
    # Kovalchuk et al. found: U = 0.8389A + 0.0764C + 0.0134, R² = 0.8098
    print(f"\nComparison with Kovalchuk et al. (2022):")
    print(f"   Their equation: Usefulness = 0.8389 × Agreement + 0.0764 × Understanding + 0.0134")
    print(f"   Your equation:  Usefulness = {beta_agreement:.4f} × Agreement + {beta_understanding:.4f} × Understanding + {intercept:.4f}")
    print(f"   Their R²: 0.8098")
    print(f"   Your R²:  {r2:.4f}")
    
    # Scatter plot: Actual vs Predicted
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, y_pred, alpha=0.5, color='steelblue')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Usefulness')
    ax.set_ylabel('Predicted Usefulness')
    ax.set_title(f'Linear Regression: Actual vs Predicted (R² = {r2:.4f})', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'linear_regression_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nRegression plot saved to {FIGURES_DIR / 'linear_regression_actual_vs_predicted.png'}")
    
    return model, r2, mae

def analyze_by_model(df):
    """
    Analyze separately for Gemini and Groq (like Kovalchuk's per-model analysis).
    """
    
    print("\n" + "="*70)
    print("PER-MODEL ANALYSIS (Gemini vs Groq)")
    print("="*70)
    
    results = []
    
    for model_name in ['google_gemini', 'groq']:
        model_df = df[df['model_name'] == model_name]
        
        if len(model_df) == 0:
            continue
        
        # Correlation
        corr = model_df[['Understanding', 'Agreement', 'Usefulness']].corr()
        agree_use_corr = corr.loc['Agreement', 'Usefulness']
        
        # Linear regression
        X = model_df[['Agreement', 'Understanding']].values
        y = model_df['Usefulness'].values
        lr = LinearRegression()
        lr.fit(X, y)
        r2 = r2_score(y, lr.predict(X))
        
        results.append({
            'Model': 'Gemini' if model_name == 'google_gemini' else 'Groq',
            'N': len(model_df),
            'Agree↔Use (r)': f"{agree_use_corr:.4f}",
            'R²': f"{r2:.4f}",
            'Equation': f"U = {lr.coef_[0]:.4f}A + {lr.coef_[1]:.4f}C + {lr.intercept_:.4f}"
        })
    
    # Print results
    results_df = pd.DataFrame(results)
    print("\nPer-Model Results:")
    print("-"*70)
    print(results_df.to_string(index=False))
    
    return results_df

def main():
    print("="*70)
    print("REPLICATING KOVALCHUK ET AL. (2022) GEM PAPER")
    print("="*70)
    print("Method: Mutual Information + Correlation + Linear Regression")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df = load_master_dataset('data/master_dataset.jsonl')
    print(f"   Loaded {len(df)} total records")
    print(f"   Models: {df['model_name'].unique().tolist()}")
    
    # Prepare perception data
    perception_df = prepare_perception_data(df)
    print(f"\nPerception data prepared:")
    print(f"   Understanding: mean = {perception_df['Understanding'].mean():.2f}, std = {perception_df['Understanding'].std():.2f}")
    print(f"   Agreement:     mean = {perception_df['Agreement'].mean():.2f}, std = {perception_df['Agreement'].std():.2f}")
    print(f"   Usefulness:    mean = {perception_df['Usefulness'].mean():.2f}, std = {perception_df['Usefulness'].std():.2f}")
    
    # 1. Mutual Information Analysis
    mi_results = compute_mutual_information(perception_df)
    
    # 2. Correlation Analysis
    corr_matrix = compute_correlations(perception_df)
    
    # 3. Linear Regression
    model, r2, mae = fit_linear_regression(perception_df)
    
    # 4. Per-Model Analysis
    per_model_results = analyze_by_model(perception_df)
    
    # Save all results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Save MI results
    mi_df = pd.DataFrame(list(mi_results.items()), columns=['Feature Pair', 'Mutual Information'])
    mi_df.to_csv(RESULTS_DIR / 'mutual_information.csv', index=False)
    
    # Save correlation matrix
    corr_matrix.to_csv(RESULTS_DIR / 'correlation_matrix.csv')
    
    # Save regression results
    regression_results = pd.DataFrame({
        'Metric': ['R²', 'MAE', 'β_Agreement', 'β_Understanding', 'Intercept'],
        'Value': [r2, mae, model.coef_[0], model.coef_[1], model.intercept_]
    })
    regression_results.to_csv(RESULTS_DIR / 'regression_results.csv', index=False)
    
    # Save per-model results
    per_model_results.to_csv(RESULTS_DIR / 'per_model_analysis.csv', index=False)
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("   - mutual_information.csv")
    print("   - correlation_matrix.csv")
    print("   - regression_results.csv")
    print("   - per_model_analysis.csv")
    print(f"   - Figures saved to: {FIGURES_DIR}")
    
    # Summary comparison with Kovalchuk et al.
    print("\n" + "="*70)
    print("SUMMARY: COMPARISON WITH KOVALCHUK ET AL. (2022)")
    print("="*70)
    print(f"\n{'Metric':<30} {'Kovalchuk et al.':<20} {'Your Study':<20}")
    print("-"*70)
    print(f"{'Agreement ↔ Usefulness (r)':<30} {'0.898':<20} {corr_matrix.loc['Agreement', 'Usefulness']:.4f}")
    print(f"{'R² (Regression)':<30} {'0.8098':<20} {r2:.4f}")
    print(f"{'Equation':<30} {'U=0.8389A+0.0764C+0.0134':<20} {'U='+str(round(model.coef_[0],4))+'A+'+str(round(model.coef_[1],4))+'C+'+str(round(model.intercept_,4))}")
    
    print("\n" + "="*70)
    print("REPLICATION COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()