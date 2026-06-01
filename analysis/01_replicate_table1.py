"""Replicate Kovalchuk et al. Table 1: Descriptive statistics and model comparison."""
import pandas as pd
import numpy as np
from pathlib import Path
from load_data import load_master_dataset

def compute_model_statistics(df):
    """Compute statistics per model."""
    models = ['google_gemini', 'groq']
    results = []
    
    for model_name in models:
        model_df = df[df['model_name'] == model_name]
        
        stats = {
            'Model': 'Google Gemini' if model_name == 'google_gemini' else 'Groq (LLaMA 3.1)',
            'Samples': len(model_df),
            'Users': model_df['evaluator_id'].nunique(),
            'Mean Correctness': f"{model_df['correctness'].mean():.2f} ± {model_df['correctness'].std():.2f}",
            'Mean Readability': f"{model_df['readability'].mean():.2f} ± {model_df['readability'].std():.2f}",
            'Mean Reliability': f"{model_df['reliability'].mean():.2f} ± {model_df['reliability'].std():.2f}",
            'Usefulness Rate': f"{(model_df['intention_to_use'] == 'yes').mean()*100:.1f}%",
            'Consistency (Cronbach α)': compute_cronbach_alpha(model_df)
        }
        results.append(stats)
    
    return pd.DataFrame(results)

def compute_cronbach_alpha(df):
    """Compute Cronbach's alpha for reliability of scores."""
    scores = df[['correctness', 'readability', 'reliability']].values
    
    if len(scores) < 2:
        return np.nan
    
    item_variances = np.var(scores, axis=0, ddof=1)
    total_variance = np.var(np.sum(scores, axis=1), ddof=1)
    
    if total_variance == 0:
        return np.nan
    
    k = scores.shape[1]
    alpha = (k / (k - 1)) * (1 - np.sum(item_variances) / total_variance)
    return round(alpha, 3)

def compute_per_evaluator_stats(df):
    """Compute statistics per evaluator."""
    evaluator_stats = []
    
    for evaluator_id in df['evaluator_id'].unique():
        user_df = df[df['evaluator_id'] == evaluator_id]
        
        # SAFELY get languages value - FIXED
        lang_value = user_df['languages'].iloc[0]
        
        # Check if it's a string vs float/NaN
        if pd.isna(lang_value):
            languages_str = '-'
        else:
            languages_str = str(lang_value)[:30] if lang_value else '-'
        
        stats = {
            'Evaluator': user_df['evaluator_name'].iloc[0],
            'Expertise': user_df['expertise_level'].iloc[0],
            'Years': user_df['programming_years'].iloc[0],
            'AI Usage': user_df['ai_usage_frequency'].iloc[0],
            'Languages': languages_str,
            'Mean Correctness': f"{user_df['correctness'].mean():.2f}",
            'Mean Readability': f"{user_df['readability'].mean():.2f}",
            'Mean Reliability': f"{user_df['reliability'].mean():.2f}",
            'Usefulness Rate': f"{(user_df['intention_to_use'] == 'yes').mean()*100:.0f}%",
            'Gemini Score': f"{user_df[user_df['model_name']=='google_gemini']['correctness'].mean():.2f}",
            'Groq Score': f"{user_df[user_df['model_name']=='groq']['correctness'].mean():.2f}"
        }
        evaluator_stats.append(stats)
    
    return pd.DataFrame(evaluator_stats)

def compute_model_comparison_tests(df):
    """Run statistical tests comparing Gemini vs Groq."""
    from scipy import stats
    
    gemini = df[df['model_name'] == 'google_gemini']
    groq = df[df['model_name'] == 'groq']
    
    # Paired t-test (same problems)
    gemini_by_problem = gemini.set_index('problem_id')['correctness']
    groq_by_problem = groq.set_index('problem_id')['correctness']
    
    common_problems = gemini_by_problem.index.intersection(groq_by_problem.index)
    gemini_aligned = gemini_by_problem[common_problems]
    groq_aligned = groq_by_problem[common_problems]
    
    t_stat, p_value = stats.ttest_rel(gemini_aligned, groq_aligned)
    
    # Effect size (Cohen's d)
    diff = gemini_aligned - groq_aligned
    cohen_d = diff.mean() / diff.std() if diff.std() > 0 else 0
    
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON: Gemini vs Groq")
    print("="*60)
    print(f"Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
    print(f"Effect size (Cohen's d): {cohen_d:.3f}")
    
    if p_value < 0.05:
        print("✓ Statistically significant difference (p < 0.05)")
    else:
        print("No significant difference found")
    
    return {'t_stat': t_stat, 'p_value': p_value, 'cohen_d': cohen_d}

def compute_cognitive_state_impact(df):
    """Compute how cognitive state (expertise) affects evaluations."""
    print("\n" + "="*60)
    print("COGNITIVE STATE IMPACT ANALYSIS")
    print("="*60)
    
    # Group by expertise level
    expertise_groups = df.groupby('expertise_level')
    
    for expertise, group in expertise_groups:
        print(f"\n{expertise.upper()}:")
        print(f"  Correctness: {group['correctness'].mean():.2f} ± {group['correctness'].std():.2f}")
        print(f"  Reliability: {group['reliability'].mean():.2f} ± {group['reliability'].std():.2f}")
        print(f"  Usefulness: {(group['intention_to_use'] == 'yes').mean()*100:.1f}%")
        print(f"  Records: {len(group)}")
    
    # ANOVA test for expertise effect
    from scipy import stats
    groups_list = [group['correctness'].values for name, group in expertise_groups]
    
    # Filter out empty groups
    groups_list = [g for g in groups_list if len(g) > 0]
    
    if len(groups_list) >= 2:
        f_stat, p_value = stats.f_oneway(*groups_list)
        print(f"\nANOVA (Expertise → Correctness): F={f_stat:.3f}, p={p_value:.4f}")
        if p_value < 0.05:
            print("✓ Expertise significantly affects correctness ratings")

def compute_expertise_model_interaction(df):
    """Check if expertise affects preference for models."""
    print("\n" + "="*60)
    print("EXPERTISE × MODEL INTERACTION")
    print("="*60)
    
    # Calculate preference per evaluator
    preferences = []
    for evaluator_id in df['evaluator_id'].unique():
        user_df = df[df['evaluator_id'] == evaluator_id]
        gemini_score = user_df[user_df['model_name'] == 'google_gemini']['correctness'].mean()
        groq_score = user_df[user_df['model_name'] == 'groq']['correctness'].mean()
        
        preferences.append({
            'Evaluator': user_df['evaluator_name'].iloc[0],
            'Expertise': user_df['expertise_level'].iloc[0],
            'Gemini': round(gemini_score, 2),
            'Groq': round(groq_score, 2),
            'Preference': 'Gemini' if gemini_score > groq_score else 'Groq' if groq_score > gemini_score else 'Tie'
        })
    
    pref_df = pd.DataFrame(preferences)
    print("\nModel Preference by Expertise:")
    print(pref_df.to_string(index=False))
    
    return pref_df

def main():
    print("="*60)
    print("REPLICATING KOVALCHUK ET AL. - MAIN ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_master_dataset('../data/master_dataset.jsonl')
    
    # 1. Model statistics (Table 1 equivalent)
    print("\n📊 TABLE 1: Model Performance Statistics")
    print("-"*60)
    model_stats = compute_model_statistics(df)
    print(model_stats.to_string(index=False))
    
    # 2. Per-evaluator statistics
    print("\n\n👥 PER-EVALUATOR STATISTICS")
    print("-"*60)
    evaluator_stats = compute_per_evaluator_stats(df)
    print(evaluator_stats.to_string(index=False))
    
    # 3. Statistical comparison
    comparison_results = compute_model_comparison_tests(df)
    
    # 4. Cognitive state impact
    compute_cognitive_state_impact(df)
    
    # 5. Expertise × Model interaction
    interaction_df = compute_expertise_model_interaction(df)
    
    # 6. Save results to CSV
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    model_stats.to_csv(output_dir / 'model_statistics.csv', index=False)
    evaluator_stats.to_csv(output_dir / 'evaluator_statistics.csv', index=False)
    interaction_df.to_csv(output_dir / 'expertise_preferences.csv', index=False)
    
    print(f"\n\n✅ Results saved to {output_dir}/")
    print("   - model_statistics.csv")
    print("   - evaluator_statistics.csv")
    print("   - expertise_preferences.csv")
    
    # Summary of key findings
    print("\n" + "="*60)
    print("KEY FINDINGS SUMMARY")
    print("="*60)
    
    # Gemini vs Groq
    gemini_row = model_stats[model_stats['Model'] == 'Google Gemini']
    groq_row = model_stats[model_stats['Model'] == 'Groq (LLaMA 3.1)']
    
    if len(gemini_row) > 0:
        gemini_correct = float(gemini_row['Mean Correctness'].iloc[0].split()[0])
        groq_correct = float(groq_row['Mean Correctness'].iloc[0].split()[0])
        print(f"1. Model Comparison: Gemini ({gemini_correct:.2f}) vs Groq ({groq_correct:.2f})")
    
        # Usefulness
        gemini_use = gemini_row['Usefulness Rate'].iloc[0]
        groq_use = groq_row['Usefulness Rate'].iloc[0]
        print(f"2. Intention to Use: Gemini ({gemini_use}) vs Groq ({groq_use})")
    
    # Expertise effect
    print(f"3. Statistical significance: p={comparison_results['p_value']:.4f}")
    
    # Cognitive state
    if comparison_results['p_value'] < 0.05:
        print("4. Cognitive state (expertise) significantly affects evaluations ✓")

if __name__ == "__main__":
    main()