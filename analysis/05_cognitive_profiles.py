# analysis/05_cognitive_profiles.py
"""Extract cognitive state profiles for each developer."""
import pandas as pd
import numpy as np
from load_data import load_master_dataset
from pathlib import Path

def extract_cognitive_profiles(df):
    """Extract cognitive state features per developer (mapping to TPB dimensions)."""
    
    profiles = []
    
    for evaluator_id in df['evaluator_id'].unique():
        user_df = df[df['evaluator_id'] == evaluator_id]
        
        profile = {
            'evaluator_id': evaluator_id,
            'evaluator_name': user_df['evaluator_name'].iloc[0],
            'expertise_level': user_df['expertise_level'].iloc[0],
            'programming_years': user_df['programming_years'].iloc[0],
            
            # TPB Dimension 1: Understanding (from readability)
            'understanding_mean': user_df['readability'].mean(),
            'understanding_std': user_df['readability'].std(),
            
            # TPB Dimension 2: Agreement/Correctness
            'agreement_mean': user_df['correctness'].mean(),
            'agreement_std': user_df['correctness'].std(),
            
            # TPB Dimension 3: Intention to Use
            'intention_rate': (user_df['intention_to_use'] == 'yes').mean(),
            'modified_rate': (user_df['intention_to_use'] == 'modified').mean(),
            'reject_rate': (user_df['intention_to_use'] == 'no').mean(),
            
            # Cognitive Load (proxy from reliability variance)
            'cognitive_load_proxy': user_df['reliability'].std(),
            
            # Model-specific behavior
            'gemini_preference': user_df[user_df['model_name'] == 'google_gemini']['intention_to_use'].iloc[0] == 'yes',
            'groq_preference': user_df[user_df['model_name'] == 'groq']['intention_to_use'].iloc[0] == 'yes',
            
            # Dual Process indicators
            'consistency_score': user_df[['correctness', 'readability', 'reliability']].std().mean(),
        }
        profiles.append(profile)
    
    return pd.DataFrame(profiles)

def analyze_cognitive_clusters(profiles_df):
    """Cluster developers by cognitive state (Figure 4 in paper)."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    features = ['understanding_mean', 'agreement_mean', 'intention_rate', 'cognitive_load_proxy']
    X = profiles_df[features].fillna(0).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal clusters
    kmeans = KMeans(n_clusters=min(3, len(profiles_df)), random_state=42)
    profiles_df['cognitive_cluster'] = kmeans.fit_predict(X_scaled)
    
    print("\n=== COGNITIVE STATE CLUSTERS ===")
    for cluster in profiles_df['cognitive_cluster'].unique():
        cluster_data = profiles_df[profiles_df['cognitive_cluster'] == cluster]
        print(f"\nCluster {cluster} ({len(cluster_data)} developers):")
        print(f"  Mean Understanding: {cluster_data['understanding_mean'].mean():.2f}")
        print(f"  Mean Agreement: {cluster_data['agreement_mean'].mean():.2f}")
        print(f"  Intention Rate: {cluster_data['intention_rate'].mean():.2%}")
        print(f"  Developers: {', '.join(cluster_data['evaluator_name'].tolist())}")
    
    return profiles_df

if __name__ == "__main__":
    df = load_master_dataset('../data/master_dataset.jsonl')
    profiles = extract_cognitive_profiles(df)
    profiles = analyze_cognitive_clusters(profiles)
    profiles.to_csv('results/cognitive_profiles.csv', index=False)
    print("\n✓ Cognitive profiles saved to results/cognitive_profiles.csv")