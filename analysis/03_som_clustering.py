"""Replicate Kovalchuk et al. Figure 4: SOM clustering."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from load_data import load_master_dataset

RESULTS_DIR = Path('results/figures')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def create_evaluator_scores(df):
    """Calculate scores for each evaluator."""
    evaluator_scores = []
    
    for evaluator_id in df['evaluator_id'].unique():
        user_df = df[df['evaluator_id'] == evaluator_id]
        
        row = {
            'evaluator_name': user_df['evaluator_name'].iloc[0],
            'expertise_level': user_df['expertise_level'].iloc[0],
            'mean_correctness': user_df['correctness'].mean(),
            'mean_reliability': user_df['reliability'].mean(),
            'mean_readability': user_df['readability'].mean(),
            'agreement_consistency': user_df['correctness'].std()
        }
        evaluator_scores.append(row)
    
    return pd.DataFrame(evaluator_scores)

def plot_som_clustering(scores_df):
    """Create SOM clustering visualization."""
    feature_cols = ['mean_correctness', 'mean_reliability', 'mean_readability', 'agreement_consistency']
    X = scores_df[feature_cols].fillna(0).values
    
    if len(X) < 2:
        print(f"Warning: Only {len(X)} evaluators. Need at least 2 for clustering.")
        return
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # SOM size based on number of evaluators
    som_size = max(3, int(np.ceil(np.sqrt(len(X)))))
    som = MiniSom(som_size, som_size, X.shape[1], sigma=1.5, learning_rate=0.5, 
                  random_seed=42, neighborhood_function='gaussian')
    
    som.train_random(X, 500, verbose=False)
    
    winner_coordinates = np.array([som.winner(x) for x in X])
    
    plt.figure(figsize=(10, 8))
    
    for i in range(som_size):
        for j in range(som_size):
            plt.plot(i, j, 'o', color='lightgray', markersize=15)
    
    color_map = {'student_bachelor': 'blue', 'student_master': 'green', 
                 'junior': 'orange', 'senior': 'red'}
    
    for idx, (x, y) in enumerate(winner_coordinates):
        evaluator = scores_df.iloc[idx]
        color = color_map.get(evaluator['expertise_level'], 'gray')
        plt.plot(x, y, 'o', color=color, markersize=20, alpha=0.7, 
                markeredgecolor='black', markeredgewidth=2)
        plt.annotate(evaluator['evaluator_name'], (x, y), fontsize=9, 
                    ha='center', va='center', color='black')
    
    plt.xlim(-0.5, som_size - 0.5)
    plt.ylim(-0.5, som_size - 0.5)
    plt.title('Figure 4: SOM Clustering of Evaluators', fontsize=14)
    plt.xlabel('SOM X')
    plt.ylabel('SOM Y')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='Bachelor'),
                       Patch(facecolor='green', label="Master's"),
                       Patch(facecolor='orange', label='Junior'),
                       Patch(facecolor='red', label='Senior')]
    plt.legend(handles=legend_elements)
    
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    
    save_path = RESULTS_DIR / 'som_clustering.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ SOM clustering saved to {save_path}")

def main():
    print("="*60)
    print("FIGURE 4: SOM Clustering Visualization")
    print("="*60)
    
    df = load_master_dataset('../data/master_dataset.jsonl')
    scores_df = create_evaluator_scores(df)
    
    plot_som_clustering(scores_df)

if __name__ == "__main__":
    main()