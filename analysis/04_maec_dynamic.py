"""Replicate Kovalchuk et al. Figure 5: MAEC dynamic depending on number of clusters."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from pathlib import Path
from load_data import load_master_dataset

RESULTS_DIR = Path('results/figures')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def compute_maec_for_clusters(df, n_clusters):
    """Compute MAEC for given number of clusters."""
    if n_clusters > df['evaluator_id'].nunique():
        return np.nan
    
    evaluator_features = []
    for evaluator_id in df['evaluator_id'].unique():
        user_df = df[df['evaluator_id'] == evaluator_id]
        features = {
            'evaluator_id': evaluator_id,
            'mean_correctness': user_df['correctness'].mean(),
            'mean_readability': user_df['readability'].mean(),
            'mean_reliability': user_df['reliability'].mean(),
            'usefulness_rate': (user_df['intention_to_use'] == 'yes').mean()
        }
        evaluator_features.append(features)
    
    features_df = pd.DataFrame(evaluator_features)
    X = features_df[['mean_correctness', 'mean_readability', 'mean_reliability', 'usefulness_rate']].values
    
    if len(X) < n_clusters:
        return np.nan
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    features_df['cluster'] = clusters
    df_with_clusters = df.merge(features_df[['evaluator_id', 'cluster']], on='evaluator_id')
    
    maes = []
    for cluster in range(n_clusters):
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster]
        if len(cluster_data) > 0:
            pred = cluster_data['usefulness_SU'].mean()
            mae = mean_absolute_error(cluster_data['usefulness_SU'], [pred] * len(cluster_data))
            maes.append(mae)
    
    return np.mean(maes) if maes else np.nan

def main():
    print("="*60)
    print("FIGURE 5: MAEC Dynamic vs Number of Clusters")
    print("="*60)
    
    df = load_master_dataset('../data/master_dataset.jsonl')
    
    max_clusters = min(8, df['evaluator_id'].nunique())
    c_values = list(range(2, max_clusters + 1))
    maec_values = []
    
    print(f"Testing clusters from 2 to {max_clusters}...")
    for c in c_values:
        maec = compute_maec_for_clusters(df, c)
        if not np.isnan(maec):
            maec_values.append(maec)
            print(f"  C={c}: MAEC={maec:.4f}")
        else:
            maec_values.append(np.nan)
    
    # Filter valid values
    valid_c = [c for c, m in zip(c_values, maec_values) if not np.isnan(m)]
    valid_maec = [m for m in maec_values if not np.isnan(m)]
    
    if len(valid_c) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(valid_c, valid_maec, 'b-o', linewidth=2, markersize=8)
        plt.xlabel('Number of Cognitive State Clusters (C)', fontsize=12)
        plt.ylabel('MAEC(SU)', fontsize=12)
        plt.title('Figure 5: MAEC Dynamic Depending on Number of Clusters C', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        optimal_idx = np.argmin(valid_maec)
        optimal_c = valid_c[optimal_idx]
        optimal_maec = valid_maec[optimal_idx]
        
        plt.axvline(x=optimal_c, color='red', linestyle='--', alpha=0.5)
        plt.annotate(f'Optimal C={optimal_c}\nMAEC={optimal_maec:.4f}', 
                    xy=(optimal_c, optimal_maec), xytext=(optimal_c+0.5, optimal_maec+0.05),
                    fontsize=10)
        
        plt.tight_layout()
        save_path = RESULTS_DIR / 'maec_dynamic.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n✓ Optimal number of clusters: C={optimal_c}")
        print(f"✓ Minimum MAEC: {optimal_maec:.4f}")
        print(f"✓ Figure saved to {save_path}")
    else:
        print("Insufficient data for clustering analysis.")

if __name__ == "__main__":
    main()