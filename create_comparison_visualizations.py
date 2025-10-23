#!/usr/bin/env python3
"""
Create visualizations comparing Contrastive Few-Shot method with GPT-4 baseline
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Load results
# results_dir = Path("/Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/contrastive_evaluation_20251010_121713_test1047")
results_dir = Path("/Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/contrastive_evaluation_20251010_135922_test10")
output_dir = results_dir / "visualizations"
output_dir.mkdir(exist_ok=True)

with open(results_dir / "evaluations" / "evaluation_summary.json") as f:
    results = json.load(f)

# GPT-4 baseline from paper
baseline_gpt4 = {
    "Overall": 50.91,
    "Lab": 51.68,
    "Physical": 77.50,
    "Date": 46.67,
    "Dosage": 37.50,
    "Risk": 33.75,
    "Severity": 27.50,
    "Diagnosis": 53.33
}

# Your method results
contrastive_results = results["contrastive_few_shot"]
your_method = {
    "Overall": contrastive_results["overall_accuracy"] * 100,
    "Lab": contrastive_results["by_category"]["lab"]["accuracy"] * 100,
    "Physical": contrastive_results["by_category"]["physical"]["accuracy"] * 100,
    "Date": contrastive_results["by_category"]["date"]["accuracy"] * 100,
    "Dosage": contrastive_results["by_category"]["dosage"]["accuracy"] * 100,
    "Risk": contrastive_results["by_category"]["risk"]["accuracy"] * 100,
    "Severity": contrastive_results["by_category"]["severity"]["accuracy"] * 100,
    "Diagnosis": contrastive_results["by_category"]["diagnosis"]["accuracy"] * 100
}

# Calculate improvements
improvements = {k: your_method[k] - baseline_gpt4[k] for k in baseline_gpt4.keys()}

print("="*80)
print("RESULTS COMPARISON")
print("="*80)
print(f"\n{'Category':<15} {'GPT-4 Baseline':<20} {'Your Method':<20} {'Improvement':<15}")
print("-"*80)
for category in baseline_gpt4.keys():
    print(f"{category:<15} {baseline_gpt4[category]:>6.2f}%{'':<13} {your_method[category]:>6.2f}%{'':<13} {improvements[category]:>+6.2f}%")
print("="*80)

# ============================================================================
# Figure 1: Overall Accuracy Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 6))

methods = ['GPT-4\nBaseline', 'Contrastive\nFew-Shot\n(Ours)']
accuracies = [baseline_gpt4["Overall"], your_method["Overall"]]
colors = ['#3498db', '#e74c3c']

bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{acc:.2f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add improvement annotation
improvement = your_method["Overall"] - baseline_gpt4["Overall"]
ax.annotate('', xy=(1, your_method["Overall"]), xytext=(0, baseline_gpt4["Overall"]),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(0.5, (baseline_gpt4["Overall"] + your_method["Overall"])/2,
        f'+{improvement:.2f}%\nimprovement',
        ha='center', va='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Overall Accuracy Comparison\nMedCalc-Bench (1047 test cases)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "overall_accuracy_comparison.png", bbox_inches='tight')
print(f"\n✓ Saved: {output_dir / 'overall_accuracy_comparison.png'}")

# ============================================================================
# Figure 2: Category-wise Accuracy Comparison (Grouped Bar Chart)
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 7))

categories = ['Lab', 'Physical', 'Date', 'Dosage', 'Risk', 'Severity', 'Diagnosis']
x = np.arange(len(categories))
width = 0.35

baseline_values = [baseline_gpt4[cat] for cat in categories]
your_values = [your_method[cat] for cat in categories]

bars1 = ax.bar(x - width/2, baseline_values, width, label='GPT-4 Baseline',
               color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, your_values, width, label='Contrastive Few-Shot (Ours)',
               color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('Category', fontsize=13, fontweight='bold')
ax.set_title('Category-wise Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(loc='upper right', fontsize=11)
ax.set_ylim(0, 105)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "category_wise_comparison.png", bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'category_wise_comparison.png'}")

# ============================================================================
# Figure 3: Improvement Heatmap
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 3))

improvement_values = [improvements[cat] for cat in ['Overall'] + categories]
improvement_labels = ['Overall'] + categories

# Create heatmap data
data = np.array([improvement_values])

# Custom colormap: red for negative, white for zero, green for positive
cmap = sns.diverging_palette(10, 130, as_cmap=True)

im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=-10, vmax=20)

# Set ticks
ax.set_xticks(np.arange(len(improvement_labels)))
ax.set_xticklabels(improvement_labels, fontsize=11)
ax.set_yticks([0])
ax.set_yticklabels(['Improvement (%)'], fontsize=12, fontweight='bold')

# Add text annotations
for i in range(len(improvement_labels)):
    text_color = 'white' if abs(improvement_values[i]) > 5 else 'black'
    ax.text(i, 0, f'{improvement_values[i]:+.1f}%',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color=text_color)

ax.set_title('Performance Improvement over GPT-4 Baseline', 
             fontsize=14, fontweight='bold', pad=15)

# Colorbar
cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.15, aspect=30)
cbar.set_label('Improvement (%)', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "improvement_heatmap.png", bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'improvement_heatmap.png'}")

# ============================================================================
# Figure 4: Side-by-side Comparison with Improvement Arrows
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

y_positions = np.arange(len(categories))
offset = 0.2

# Plot baseline
baseline_vals = [baseline_gpt4[cat] for cat in categories]
your_vals = [your_method[cat] for cat in categories]

# Scatter plots
ax.scatter([baseline_gpt4["Overall"]], [-1], s=200, color='#3498db', 
          marker='o', edgecolors='black', linewidths=2, label='GPT-4 Baseline', zorder=3)
ax.scatter([your_method["Overall"]], [-1], s=200, color='#e74c3c', 
          marker='s', edgecolors='black', linewidths=2, label='Contrastive Few-Shot (Ours)', zorder=3)

for i, cat in enumerate(categories):
    # Baseline
    ax.scatter([baseline_gpt4[cat]], [i], s=150, color='#3498db', 
              marker='o', edgecolors='black', linewidths=1.5, zorder=3)
    # Your method
    ax.scatter([your_method[cat]], [i], s=150, color='#e74c3c', 
              marker='s', edgecolors='black', linewidths=1.5, zorder=3)
    
    # Arrow showing improvement
    if improvements[cat] > 0:
        ax.annotate('', xy=(your_method[cat], i), xytext=(baseline_gpt4[cat], i),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2.5, alpha=0.7))
    else:
        ax.annotate('', xy=(your_method[cat], i), xytext=(baseline_gpt4[cat], i),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2.5, alpha=0.7))
    
    # Improvement text
    mid_x = (baseline_gpt4[cat] + your_method[cat]) / 2
    color = 'green' if improvements[cat] > 0 else 'red'
    ax.text(mid_x, i + 0.25, f'{improvements[cat]:+.1f}%', 
           ha='center', va='bottom', fontsize=9, fontweight='bold',
           color=color, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
           edgecolor=color, alpha=0.8))

# Overall line separator
ax.axhline(-0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
ax.text(-5, -0.7, 'Overall', ha='right', va='top', fontsize=11, fontweight='bold')

ax.set_yticks(list(range(-1, len(categories))))
ax.set_yticklabels(['Overall'] + categories, fontsize=11)
ax.set_xlabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Detailed Performance Comparison with Improvements', 
            fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(20, 100)
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "detailed_comparison_with_arrows.png", bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'detailed_comparison_with_arrows.png'}")

# ============================================================================
# Figure 5: Statistical Summary Table as Image
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

table_data = []
table_data.append(['Category', 'GPT-4 Baseline', 'Contrastive Few-Shot (Ours)', 'Improvement', 'Relative Gain'])
table_data.append(['', '(Accuracy %)', '(Accuracy %)', '(Absolute %)', '(Relative %)'])

for cat in ['Overall'] + categories:
    baseline_val = baseline_gpt4[cat]
    your_val = your_method[cat]
    improvement = improvements[cat]
    relative_gain = (improvement / baseline_val) * 100 if baseline_val > 0 else 0
    
    table_data.append([
        cat,
        f'{baseline_val:.2f}%',
        f'{your_val:.2f}%',
        f'{improvement:+.2f}%',
        f'{relative_gain:+.1f}%'
    ])

# Create table
table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.20, 0.20, 0.20, 0.20, 0.20])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for i in range(5):
    table[(0, i)].set_facecolor('#2c3e50')
    table[(0, i)].set_text_props(weight='bold', color='white')
    table[(1, i)].set_facecolor('#34495e')
    table[(1, i)].set_text_props(weight='bold', color='white', fontsize=9)

# Style data rows
for i in range(2, len(table_data)):
    for j in range(5):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')
        else:
            table[(i, j)].set_facecolor('white')
        
        # Highlight improvements
        if j == 3:  # Improvement column
            val = float(table_data[i][3].replace('%', '').replace('+', ''))
            if val > 0:
                table[(i, j)].set_text_props(color='green', weight='bold')
            elif val < 0:
                table[(i, j)].set_text_props(color='red', weight='bold')

# Highlight "Overall" row
for j in range(5):
    table[(2, j)].set_facecolor('#f39c12')
    table[(2, j)].set_text_props(weight='bold')

ax.set_title('Comprehensive Results Comparison\nMedCalc-Bench Test Set (1047 examples)', 
            fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(output_dir / "results_table.png", bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'results_table.png'}")

print(f"\n{'='*80}")
print(f"All visualizations saved to: {output_dir}/")
print(f"{'='*80}\n")

# Save summary statistics
summary = {
    "overall_improvement": improvements["Overall"],
    "average_category_improvement": np.mean([improvements[cat] for cat in categories]),
    "best_improvement_category": max(categories, key=lambda x: improvements[x]),
    "best_improvement_value": max([improvements[cat] for cat in categories]),
    "categories_with_improvement": sum(1 for cat in categories if improvements[cat] > 0),
    "total_categories": len(categories)
}

with open(output_dir / "summary_statistics.json", 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Saved: {output_dir / 'summary_statistics.json'}")

print(f"\n📊 Key Findings:")
print(f"   • Overall improvement: +{summary['overall_improvement']:.2f}%")
print(f"   • Average category improvement: +{summary['average_category_improvement']:.2f}%")
print(f"   • Best improvement in: {summary['best_improvement_category']} (+{summary['best_improvement_value']:.2f}%)")
print(f"   • Categories improved: {summary['categories_with_improvement']}/{summary['total_categories']}")

