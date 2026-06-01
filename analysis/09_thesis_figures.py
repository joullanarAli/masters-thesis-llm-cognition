# analysis/08_thesis_figures.py
"""Create publication-quality figures for thesis."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from load_data import load_master_dataset

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

def create_model_comparison_figure(df):
    """Figure 1: Bar chart comparing Gemini vs Groq."""
    
    gemini = df[df['model_name'] == 'google_gemini']
    groq = df[df['model_name'] == 'groq']
    
    metrics = ['correctness', 'readability', 'reliability']
    gemini_means = [gemini[m].mean() for m in metrics]
    groq_means = [groq[m].mean() for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width/2, gemini_means, width, label='Google Gemini', color='#4285f4')
    bars2 = ax.bar(x + width/2, groq_means, width, label='Groq (LLaMA 3.1)', color='#f97316')
    
    ax.set_ylabel('Score (1-5)')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Correctness', 'Readability', 'Reliability'])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 5.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/figures/figure1_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Figure 1 saved: results/figures/figure1_model_comparison.png")

def create_usefulness_comparison_figure(df):
    """Figure 2: Intention to use comparison."""
    
    gemini = df[df['model_name'] == 'google_gemini']
    groq = df[df['model_name'] == 'groq']
    
    categories = ['Yes', 'With Modifications', 'No']
    gemini_counts = [
        (gemini['intention_to_use'] == 'yes').sum(),
        (gemini['intention_to_use'] == 'modified').sum(),
        (gemini['intention_to_use'] == 'no').sum()
    ]
    groq_counts = [
        (groq['intention_to_use'] == 'yes').sum(),
        (groq['intention_to_use'] == 'modified').sum(),
        (groq['intention_to_use'] == 'no').sum()
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, gemini_counts, width, label='Google Gemini', color='#4285f4')
    ax.bar(x + width/2, groq_counts, width, label='Groq (LLaMA 3.1)', color='#f97316')
    
    ax.set_ylabel('Number of Responses')
    ax.set_title('Developer Intention to Use AI-Generated Code', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('results/figures/figure2_usefulness_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Figure 2 saved: results/figures/figure2_usefulness_comparison.png")

def create_cognitive_state_impact_figure(df):
    """Figure 3: How expertise affects usefulness."""
    
    expertise_order = ['student_bachelor', 'junior', 'student_master', 'senior']
    labels = ['Bachelor', 'Junior', 'Master', 'Senior']
    
    usefulness_rates = []
    for expertise in expertise_order:
        exp_df = df[df['expertise_level'] == expertise]
        rate = (exp_df['intention_to_use'] == 'yes').mean() * 100
        usefulness_rates.append(rate)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.bar(labels, usefulness_rates, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Usefulness Rate (%)')
    ax.set_title('Impact of Developer Expertise on AI Code Adoption', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bar, rate in zip(bars, usefulness_rates):
        ax.annotate(f'{rate:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/figures/figure3_cognitive_state_impact.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Figure 3 saved: results/figures/figure3_cognitive_state_impact.png")

def create_bbn_accuracy_figure():
    """Figure 4: BBN accuracy comparison."""
    
    models = ['Gemini\n(ceiling effect)', 'Groq\n(main result)']
    accuracies = [82.3, 60.2]
    baselines = [80.8, 39.8]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width/2, accuracies, width, label='BBN with Cognitive State', color='#2ca02c')
    bars2 = ax.bar(x + width/2, baselines, width, label='Baseline (majority class)', color='#7f7f7f')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Cognitive State Prediction Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    
    # Add improvement annotations
    ax.annotate('+1.8%', xy=(0, 85), ha='center', fontsize=10, color='#2ca02c', fontweight='bold')
    ax.annotate('+51.1%', xy=(1, 65), ha='center', fontsize=10, color='#2ca02c', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/figures/figure4_bbn_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Figure 4 saved: results/figures/figure4_bbn_accuracy.png")

if __name__ == "__main__":
    print("Creating publication-ready figures for thesis...")
    print("="*60)
    
    df = load_master_dataset('../data/master_dataset.jsonl')
    
    create_model_comparison_figure(df)
    create_usefulness_comparison_figure(df)
    create_cognitive_state_impact_figure(df)
    create_bbn_accuracy_figure()
    
    print("\n✅ All figures saved to results/figures/")