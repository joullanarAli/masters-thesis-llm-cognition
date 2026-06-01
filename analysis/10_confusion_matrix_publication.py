# analysis/10_confusion_matrix_publication.py
"""Generate publication-ready confusion matrix visualization."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def plot_confusion_matrix():
    """Plot confusion matrix for Groq BBN results."""
    
    # From your BBN output
    cm = np.array([
        [136, 41, 2],   # Actual Low
        [21, 260, 0],   # Actual Medium
        [0, 142, 0]     # Actual High
    ])
    
    labels = ['Low', 'Medium', 'High']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=ax, cbar_kws={'label': 'Count'})
    
    # Calculate and add percentages
    row_sums = cm.sum(axis=1, keepdims=True)
    percentages = (cm / row_sums * 100).round(1)
    
    for i in range(3):
        for j in range(3):
            if cm[i, j] > 0:
                ax.text(j + 0.5, i + 0.7, f'({percentages[i, j]}%)',
                       ha='center', va='center', fontsize=9, color='gray')
    
    ax.set_xlabel('Predicted Usefulness', fontsize=12)
    ax.set_ylabel('Actual Usefulness', fontsize=12)
    ax.set_title('Confusion Matrix: Groq BBN Predictions', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/figures/figure5_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Figure 5 saved: results/figures/figure5_confusion_matrix.png")

if __name__ == "__main__":
    plot_confusion_matrix()