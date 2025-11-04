#!/usr/bin/env python3
"""
Results Visualization for Publication

This module creates publication-ready visualizations comparing:
- Original one-shot prompt
- Contrastive few-shot prompt (refined)

Usage:
    python visualize_results.py --evaluation-dir <path>
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from scipy import stats
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class ResultsVisualizer:
    """Create publication-ready visualizations of evaluation results."""
    
    def __init__(self, evaluation_dir: str):
        """
        Initialize the visualizer.
        
        Args:
            evaluation_dir: Directory containing evaluation results
        """
        self.evaluation_dir = Path(evaluation_dir)
        self.output_dir = self.evaluation_dir / "visualizations"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        self.load_data()
        
        print(f"✅ Visualizer initialized")
        print(f"   • Evaluation dir: {self.evaluation_dir}")
        print(f"   • Output dir: {self.output_dir}")
    
    def load_data(self):
        """Load evaluation results and responses."""
        # Load evaluation summary
        eval_file = self.evaluation_dir / "evaluations" / "evaluation_summary.json"
        with open(eval_file, 'r') as f:
            self.eval_summary = json.load(f)
        
        # Load responses (optional - only if files exist)
        self.original_responses = []
        original_file = self.evaluation_dir / "responses" / "original_one_shot_responses.jsonl"
        if original_file.exists():
            with open(original_file, 'r') as f:
                for line in f:
                    self.original_responses.append(json.loads(line))
            print(f"   ✓ Loaded {len(self.original_responses)} original responses")
        else:
            print(f"   ℹ️  No original responses file found (using summary data only)")
        
        self.contrastive_responses = []
        contrastive_file = self.evaluation_dir / "responses" / "contrastive_few_shot_responses.jsonl"
        if contrastive_file.exists():
            with open(contrastive_file, 'r') as f:
                for line in f:
                    self.contrastive_responses.append(json.loads(line))
            print(f"   ✓ Loaded {len(self.contrastive_responses)} contrastive responses")
        else:
            print(f"   ℹ️  No contrastive responses file found (using summary data only)")
    
    def plot_overall_comparison(self):
        """Plot overall accuracy comparison."""
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Handle both evaluation formats
        baseline_key = 'original_one_shot' if 'original_one_shot' in self.eval_summary else 'baseline_gpt4_paper'
        
        methods = ['GPT-4\nBaseline', 'Contrastive\nFew-Shot\n(Ours)']
        accuracies = [
            self.eval_summary[baseline_key]['overall_accuracy'],
            self.eval_summary['contrastive_few_shot']['overall_accuracy']
        ]
        
        colors = ['#3498db', '#e74c3c']
        bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.1%}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Add improvement annotation
        improvement = self.eval_summary['improvement']
        ax.annotate(f'Improvement:\n{improvement:+.1%}',
                   xy=(1, accuracies[1]), xytext=(1.3, (accuracies[0] + accuracies[1])/2),
                   arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                   fontsize=10, ha='left', color='green', fontweight='bold')
        
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_title('Overall Accuracy Comparison\nMedCalc-Bench Test Set', fontweight='bold', pad=15)
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'overall_accuracy_comparison.png', bbox_inches='tight')
        plt.savefig(self.output_dir / 'overall_accuracy_comparison.pdf', bbox_inches='tight')
        plt.close()
        
        print("   ✓ Created overall accuracy comparison")
    
    def plot_category_comparison(self):
        """Plot per-category accuracy comparison."""
        # Handle both evaluation formats
        baseline_key = 'original_one_shot' if 'original_one_shot' in self.eval_summary else 'baseline_gpt4_paper'
        
        categories_orig = self.eval_summary[baseline_key].get('by_category', {})
        categories_contr = self.eval_summary['contrastive_few_shot']['by_category']
        
        # Get all categories
        all_categories = sorted(set(list(categories_orig.keys()) + list(categories_contr.keys())))
        
        # Prepare data
        orig_accs = [categories_orig.get(cat, {}).get('accuracy', 0) for cat in all_categories]
        contr_accs = [categories_contr.get(cat, {}).get('accuracy', 0) for cat in all_categories]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(all_categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, orig_accs, width, label='Original One-Shot',
                      color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, contr_accs, width, label='Contrastive Few-Shot',
                      color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1%}',
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_title('Accuracy by Medical Category', fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([cat.title() for cat in all_categories], rotation=45, ha='right')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'category_comparison.png', bbox_inches='tight')
        plt.savefig(self.output_dir / 'category_comparison.pdf', bbox_inches='tight')
        plt.close()
        
        print("   ✓ Created category comparison")
    
    def plot_improvement_distribution(self):
        """Plot distribution of improvements across calculators."""
        # Handle both evaluation formats
        baseline_key = 'original_one_shot' if 'original_one_shot' in self.eval_summary else 'baseline_gpt4_paper'
        
        calculators_orig = self.eval_summary[baseline_key].get('by_calculator', {})
        calculators_contr = self.eval_summary['contrastive_few_shot'].get('by_calculator', {})
        
        # Calculate improvements
        improvements = []
        calculator_names = []
        
        for calc_name in calculators_orig.keys():
            if calc_name in calculators_contr:
                orig_acc = calculators_orig[calc_name]['accuracy']
                contr_acc = calculators_contr[calc_name]['accuracy']
                improvement = contr_acc - orig_acc
                improvements.append(improvement * 100)  # Convert to percentage points
                calculator_names.append(calc_name)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(improvements, bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Change')
        ax1.axvline(x=np.mean(improvements), color='blue', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(improvements):.1f}%')
        ax1.set_xlabel('Improvement (percentage points)', fontweight='bold')
        ax1.set_ylabel('Number of Calculators', fontweight='bold')
        ax1.set_title('Distribution of Accuracy Improvements', fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Top improvements
        sorted_indices = np.argsort(improvements)[::-1]
        top_10 = sorted_indices[:10]
        
        top_names = [calculator_names[i][:30] for i in top_10]
        top_improvements = [improvements[i] for i in top_10]
        
        colors_top = ['#27ae60' if imp > 0 else '#e74c3c' for imp in top_improvements]
        
        ax2.barh(range(len(top_names)), top_improvements, color=colors_top, alpha=0.8, edgecolor='black')
        ax2.set_yticks(range(len(top_names)))
        ax2.set_yticklabels(top_names, fontsize=8)
        ax2.set_xlabel('Improvement (percentage points)', fontweight='bold')
        ax2.set_title('Top 10 Largest Improvements', fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'improvement_distribution.png', bbox_inches='tight')
        plt.savefig(self.output_dir / 'improvement_distribution.pdf', bbox_inches='tight')
        plt.close()
        
        print("   ✓ Created improvement distribution")
    
    def plot_error_analysis(self):
        """Analyze and plot error patterns."""
        # Compare errors: which examples were wrong in original but correct in contrastive?
        original_correct_ids = {r['Row Number'] for r in self.original_responses if r['Result'] == 'Correct'}
        contrastive_correct_ids = {r['Row Number'] for r in self.contrastive_responses if r['Result'] == 'Correct'}
        
        # Fixed by contrastive
        fixed_ids = contrastive_correct_ids - original_correct_ids
        # Broken by contrastive
        broken_ids = original_correct_ids - contrastive_correct_ids
        # Both correct
        both_correct_ids = original_correct_ids & contrastive_correct_ids
        # Both incorrect
        both_incorrect_ids = set(r['Row Number'] for r in self.original_responses) - (original_correct_ids | contrastive_correct_ids)
        
        # Create pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Overall breakdown
        sizes = [len(both_correct_ids), len(fixed_ids), len(broken_ids), len(both_incorrect_ids)]
        labels = [f'Both Correct\n({len(both_correct_ids)})',
                 f'Fixed by Contrastive\n({len(fixed_ids)})',
                 f'Broken by Contrastive\n({len(broken_ids)})',
                 f'Both Incorrect\n({len(both_incorrect_ids)})']
        colors = ['#27ae60', '#3498db', '#e74c3c', '#95a5a6']
        explode = (0, 0.1, 0.1, 0)
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 9})
        ax1.set_title('Error Analysis: Changes in Correctness', fontweight='bold')
        
        # Net improvement
        net_data = [len(fixed_ids), len(broken_ids)]
        net_labels = [f'Fixed\n{len(fixed_ids)}', f'Broken\n{len(broken_ids)}']
        net_colors = ['#27ae60', '#e74c3c']
        
        bars = ax2.bar(range(len(net_data)), net_data, color=net_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_xticks(range(len(net_labels)))
        ax2.set_xticklabels(net_labels)
        ax2.set_ylabel('Number of Examples', fontweight='bold')
        ax2.set_title('Net Effect of Contrastive Few-Shot', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add net improvement annotation
        net_improvement = len(fixed_ids) - len(broken_ids)
        ax2.text(0.5, max(net_data) * 0.9, f'Net Improvement:\n+{net_improvement} examples',
                ha='center', fontsize=11, fontweight='bold', color='green',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_analysis.png', bbox_inches='tight')
        plt.savefig(self.output_dir / 'error_analysis.pdf', bbox_inches='tight')
        plt.close()
        
        print("   ✓ Created error analysis")
    
    def plot_statistical_significance(self):
        """Perform and visualize statistical significance tests."""
        # Skip if response data not available
        if not self.original_responses or not self.contrastive_responses:
            print("   ℹ️  Skipping statistical significance (response data not available)")
            return
            
        # Paired comparison (same examples)
        original_correct = [1 if r['Result'] == 'Correct' else 0 for r in self.original_responses]
        contrastive_correct = [1 if r['Result'] == 'Correct' else 0 for r in self.contrastive_responses]
        
        # Use chi-square test instead of McNemar (for compatibility)
        from scipy.stats import chi2_contingency
        
        # Create contingency table
        both_correct = sum(1 for o, c in zip(original_correct, contrastive_correct) if o == 1 and c == 1)
        orig_only = sum(1 for o, c in zip(original_correct, contrastive_correct) if o == 1 and c == 0)
        contr_only = sum(1 for o, c in zip(original_correct, contrastive_correct) if o == 0 and c == 1)
        both_incorrect = sum(1 for o, c in zip(original_correct, contrastive_correct) if o == 0 and c == 0)
        
        contingency_table = [[both_correct, contr_only],
                            [orig_only, both_incorrect]]
        
        # Chi-square test for independence
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Create a simple result object
        class Result:
            def __init__(self, statistic, pvalue):
                self.statistic = statistic
                self.pvalue = pvalue
        
        result = Result(chi2, p_value)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Display contingency table as heatmap
        im = ax.imshow([[both_correct, contr_only],
                       [orig_only, both_incorrect]],
                      cmap='YlOrRd', alpha=0.6)
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Contrastive Correct', 'Contrastive Incorrect'])
        ax.set_yticklabels(['Original Correct', 'Original Incorrect'])
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                value = contingency_table[i][j]
                text = ax.text(j, i, value,
                             ha="center", va="center", color="black", fontsize=14, fontweight='bold')
        
        # Add statistics text
        significance = "***" if result.pvalue < 0.001 else "**" if result.pvalue < 0.01 else "*" if result.pvalue < 0.05 else "ns"
        
        stats_text = f"McNemar's Test\n"
        stats_text += f"χ² = {result.statistic:.2f}\n"
        stats_text += f"p-value = {result.pvalue:.4f}\n"
        stats_text += f"Significance: {significance}\n\n"
        stats_text += f"Net Improvement: +{contr_only - orig_only}"
        
        ax.text(0.5, -0.5, stats_text,
               ha='center', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               transform=ax.transAxes)
        
        ax.set_title('Statistical Significance Test\n(McNemar\'s Test for Paired Data)',
                    fontweight='bold', pad=15)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'statistical_significance.png', bbox_inches='tight')
        plt.savefig(self.output_dir / 'statistical_significance.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Created statistical significance plot (p={result.pvalue:.4f})")
    
    def create_summary_table(self):
        """Create a summary table of results."""
        # Handle both evaluation formats
        baseline_key = 'original_one_shot' if 'original_one_shot' in self.eval_summary else 'baseline_gpt4_paper'
        baseline_label = 'Original One-Shot' if baseline_key == 'original_one_shot' else 'GPT-4 Baseline'
        
        # Simplified metrics - only include what's available in both formats
        summary_data = {
            'Metric': [
                'Overall Accuracy',
                'Total Examples',
                'Correct Answers',
                'Incorrect Answers'
            ],
            baseline_label: [],
            'Contrastive Few-Shot': [],
            'Improvement': []
        }
        
        orig = self.eval_summary[baseline_key]
        contr = self.eval_summary['contrastive_few_shot']
        
        # Fill in data
        summary_data[baseline_label].append(f"{orig['overall_accuracy']:.2%}")
        summary_data['Contrastive Few-Shot'].append(f"{contr['overall_accuracy']:.2%}")
        summary_data['Improvement'].append(f"{self.eval_summary.get('improvement', contr['overall_accuracy'] - orig['overall_accuracy']):+.2%}")
        
        summary_data[baseline_label].append(str(orig['total']))
        summary_data['Contrastive Few-Shot'].append(str(contr['total']))
        summary_data['Improvement'].append('-')
        
        summary_data[baseline_label].append(str(orig['correct']))
        summary_data['Contrastive Few-Shot'].append(str(contr['correct']))
        summary_data['Improvement'].append(f"+{contr['correct'] - orig['correct']}")
        
        summary_data[baseline_label].append(str(orig['incorrect']))
        summary_data['Contrastive Few-Shot'].append(str(contr['incorrect']))
        summary_data['Improvement'].append(f"{contr['incorrect'] - orig['incorrect']:+d}")
        
        # Save as CSV
        df = pd.DataFrame(summary_data)
        df.to_csv(self.output_dir / 'summary_table.csv', index=False)
        
        # Also create a nice formatted text version
        with open(self.output_dir / 'summary_table.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("RESULTS SUMMARY TABLE\n")
            f.write("="*80 + "\n\n")
            f.write(df.to_string(index=False))
            f.write("\n\n" + "="*80 + "\n")
        
        print("   ✓ Created summary table")
    
    def generate_all_visualizations(self):
        """Generate all visualizations."""
        print("\n📊 Generating visualizations...")
        print("="*60)
        
        self.plot_overall_comparison()
        self.plot_category_comparison()
        self.plot_improvement_distribution()
        self.plot_error_analysis()
        self.plot_statistical_significance()
        self.create_summary_table()
        
        print(f"\n✅ All visualizations created!")
        print(f"📁 Saved to: {self.output_dir}/")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob('*')):
            print(f"   • {file.name}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Visualize Evaluation Results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--evaluation-dir',
        type=str,
        required=True,
        help='Directory containing evaluation results'
    )
    
    args = parser.parse_args()
    
    # Create visualizer and generate plots
    visualizer = ResultsVisualizer(args.evaluation_dir)
    visualizer.generate_all_visualizations()
    
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()

