"""Replicate Kovalchuk et al. Figure 3: t-SNE visualization of user embeddings."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from load_data import load_master_dataset

# Create results directory
RESULTS_DIR = Path('results/figures')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def create_user_embeddings(df):
    """Create feature vectors for each evaluator."""
    user_features = []
    
    for evaluator_id in df['evaluator_id'].unique():
        user_df = df[df['evaluator_id'] == evaluator_id]
        
        features = {
            'evaluator_name': user_df['evaluator_name'].iloc[0],
            'expertise_level': user_df['expertise_level'].iloc[0],
            'programming_years': user_df['programming_years'].iloc[0],
            'mean_correctness': user_df['correctness'].mean(),
            'mean_readability': user_df['readability'].mean(),
            'mean_reliability': user_df['reliability'].mean(),
            'usefulness_rate': (user_df['intention_to_use'] == 'yes').mean(),
            'score_variance': user_df['correctness'].std()
        }
        user_features.append(features)
    
    return pd.DataFrame(user_features)

def plot_tsne(embeddings_df):
    """Create t-SNE visualization (Figure 3 in paper)."""
    feature_cols = ['mean_correctness', 'mean_readability', 'mean_reliability', 
                    'usefulness_rate', 'score_variance']
    X = embeddings_df[feature_cols].fillna(0)
    
    if len(X) < 3:
        print(f"Warning: Only {len(X)} evaluators. t-SNE needs at least 3 samples.")
        return
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    perplexity = min(2, len(embeddings_df) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    
    expertise_colors = {
        'student_bachelor': 'blue',
        'student_master': 'green',
        'junior': 'orange',
        'senior': 'red',
        'mid': 'purple'
    }
    
    for expertise, color in expertise_colors.items():
        mask = embeddings_df['expertise_level'] == expertise
        if mask.any():
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=color, label=expertise, s=200, alpha=0.7, 
                       edgecolors='black', linewidth=2)
    
    for i, row in embeddings_df.iterrows():
        plt.annotate(row['evaluator_name'], (X_tsne[i, 0], X_tsne[i, 1]), 
                    fontsize=10, ha='center', va='bottom')
    
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('Figure 3: t-SNE Visualization of Evaluator Embeddings')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = RESULTS_DIR / 'tsne_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ t-SNE visualization saved to {save_path}")

def main():
    print("="*60)
    print("FIGURE 3: t-SNE Visualization of User Embeddings")
    print("="*60)
    
    df = load_master_dataset('../data/master_dataset.jsonl')
    user_embeddings = create_user_embeddings(df)
    
    print(f"\nEvaluators analyzed: {len(user_embeddings)}")
    print(user_embeddings[['evaluator_name', 'expertise_level', 'programming_years']].to_string(index=False))
    
    plot_tsne(user_embeddings)

if __name__ == "__main__":
    main()