#!/usr/bin/env python3
"""
Comprehensive Results Visualization

This script creates publication-ready visualizations comparing:
- Grid search results (all prompts and models)
- Test set results across models
- Improvement over baseline
- Per-category breakdown

Usage:
    python visualize_complete_results.py \
        --grid-search-dir <path> \
        --test-results-dir <path> \
        --output-dir <path>
"""

import json
import os
from pathlib import Path
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ResultsVisualizer:
    """Comprehensive results visualizer."""
    
    def __init__(self,
                 grid_search_dir: str,
                 test_results_dir: str,
                 output_dir: str = None):
        """
        Initialize visualizer.
        
        Args:
            grid_search_dir: Directory with grid search results
            test_results_dir: Directory with test evaluation results
            output_dir: Output directory for visualizations
        """
        self.grid_search_dir = Path(grid_search_dir)
        self.test_results_dir = Path(test_results_dir)
        
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(__file__).parent / "visualizations" / f"viz_{timestamp}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Visualizer initialized")
        print(f"   • Grid search: {self.grid_search_dir}")
        print(f"   • Test results: {self.test_results_dir}")
        print(f"   • Output: {self.output_dir}")
    
    def load_grid_search_summary(self) -> Dict:
        """Load grid search summary."""
        summary_file = self.grid_search_dir / "grid_search_summary.json"
        with open(summary_file, 'r') as f:
            return json.load(f)
    
    def load_test_summary(self) -> Dict:
        """Load test evaluation summary."""
        summary_file = self.test_results_dir / "test_evaluation_summary.json"
        with open(summary_file, 'r') as f:
            return json.load(f)
    
    def visualize_grid_search_heatmap(self, grid_summary: Dict):
        """Create heatmap of grid search results."""
        
        print("\n📊 Creating grid search heatmap...")
        
        # Extract results
        all_results = grid_summary['optimal_config']['all_results']
        
        # Create pivot table
        candidates = sorted(set(r['candidate_id'] for r in all_results))
        models = sorted(set(r['model'] for r in all_results), reverse=True)
        
        # Create matrix
        matrix = np.zeros((len(models), len(candidates)))
        for r in all_results:
            model_idx = models.index(r['model'])
            cand_idx = candidates.index(r['candidate_id'])
            matrix[model_idx, cand_idx] = r['accuracy'] * 100
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 6))
        
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # Set ticks
        ax.set_xticks(np.arange(len(candidates)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels([f"C{c}" for c in candidates])
        ax.set_yticklabels(models)
        
        # Add values
        for i in range(len(models)):
            for j in range(len(candidates)):
                text = ax.text(j, i, f'{matrix[i, j]:.1f}%',
                             ha="center", va="center", color="black", fontsize=9)
        
        # Mark optimal
        optimal_cand = grid_summary['optimal_config']['optimal_candidate_id']
        optimal_model = grid_summary['optimal_config']['optimal_model']
        optimal_model_idx = models.index(optimal_model)
        optimal_cand_idx = candidates.index(optimal_cand)
        
        ax.add_patch(plt.Rectangle((optimal_cand_idx - 0.5, optimal_model_idx - 0.5), 
                                   1, 1, fill=False, edgecolor='blue', linewidth=3))
        
        ax.set_xlabel('Candidate Prompt', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        ax.set_title('Grid Search Results: Validation Accuracy (%)\nBlue box = Optimal', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.colorbar(im, ax=ax, label='Accuracy (%)')
        plt.tight_layout()
        
        output_file = self.output_dir / "grid_search_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved to: {output_file}")
    
    def visualize_test_comparison(self, grid_summary: Dict, test_summary: Dict):
        """Create bar chart comparing test results."""
        
        print("\n📊 Creating test results comparison...")
        
        # Extract data
        baseline_acc = test_summary['baseline_gpt4_paper']['accuracy'] * 100
        evaluations = test_summary['evaluations']
        
        models = list(evaluations.keys())
        accuracies = [evaluations[m]['accuracy'] * 100 for m in models]
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(models) + 1)
        bars = ax.bar(x, [baseline_acc] + accuracies, 
                     color=['gray'] + ['#2ecc71', '#3498db', '#e74c3c'][:len(models)])
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, [baseline_acc] + accuracies)):
            height = bar.get_height()
            improvement = val - baseline_acc
            label = f'{val:.2f}%\n({improvement:+.2f}%)' if i > 0 else f'{val:.2f}%\n(baseline)'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va='bottom', fontsize=10)
        
        ax.set_xticks(x)
        ax.set_xticklabels(['GPT-4\n(Paper)'] + models, fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Test Set Performance Comparison\nOptimal Prompt vs. Baseline', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 100)
        ax.axhline(y=baseline_acc, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        ax.grid(axis='y', alpha=0.3)
        
        plt.legend()
        plt.tight_layout()
        
        output_file = self.output_dir / "test_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved to: {output_file}")
    
    def visualize_by_category(self, test_summary: Dict):
        """Create category-wise comparison."""
        
        print("\n📊 Creating category-wise comparison...")
        
        # Load detailed evaluations
        evaluations_dir = self.test_results_dir / "evaluations"
        
        # Find all evaluation files
        eval_files = list(evaluations_dir.glob("*_evaluation.json"))
        
        if not eval_files:
            print("   ⚠️  No detailed evaluation files found, skipping category visualization")
            return
        
        # Load first evaluation to get categories
        with open(eval_files[0], 'r') as f:
            first_eval = json.load(f)
        
        categories = sorted(first_eval['by_category'].keys())
        
        # Collect data for all models
        model_category_data = {}
        for eval_file in eval_files:
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
            
            model = eval_data['model']
            model_category_data[model] = {
                cat: eval_data['by_category'][cat]['accuracy'] * 100
                for cat in categories
            }
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(categories))
        width = 0.8 / len(model_category_data)
        
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        for i, (model, cat_data) in enumerate(model_category_data.items()):
            values = [cat_data[cat] for cat in categories]
            offset = (i - len(model_category_data)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=model, color=colors[i % len(colors)])
        
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Performance by Medical Category', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 100)
        ax.legend(title='Model', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.output_dir / "category_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved to: {output_file}")
    
    def visualize_improvement_breakdown(self, grid_summary: Dict, test_summary: Dict):
        """Create improvement breakdown visualization."""
        
        print("\n📊 Creating improvement breakdown...")
        
        baseline_acc = test_summary['baseline_gpt4_paper']['accuracy'] * 100
        optimal_model = grid_summary['optimal_config']['optimal_model']
        
        if optimal_model in test_summary['evaluations']:
            optimal_acc = test_summary['evaluations'][optimal_model]['accuracy'] * 100
        else:
            print("   ⚠️  Optimal model not found in test results")
            return
        
        improvement = optimal_acc - baseline_acc
        
        # Create figure with summary
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Summary
        summary_data = {
            'Baseline\n(GPT-4 Paper)': baseline_acc,
            'Our Method\n(Optimal)': optimal_acc
        }
        
        bars = ax1.bar(summary_data.keys(), summary_data.values(), 
                      color=['gray', '#2ecc71'], width=0.6)
        
        for bar, val in zip(bars, summary_data.values()):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Overall Performance\nImprovement: +{improvement:.2f}%', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3)
        
        # Right: Improvement components
        val_acc = grid_summary['optimal_config']['validation_accuracy'] * 100
        test_acc = optimal_acc
        
        components = {
            'Validation\nAccuracy': val_acc,
            'Test Set\nAccuracy': test_acc
        }
        
        bars = ax2.bar(components.keys(), components.values(),
                      color=['#3498db', '#2ecc71'], width=0.6)
        
        for bar, val in zip(bars, components.values()):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax2.axhline(y=baseline_acc, color='gray', linestyle='--', 
                   alpha=0.7, label='Baseline', linewidth=2)
        ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Validation vs Test Performance', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.output_dir / "improvement_breakdown.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved to: {output_file}")
    
    def visualize_model_efficiency(self, test_summary: Dict):
        """Create model efficiency visualization (accuracy vs cost)."""
        
        print("\n📊 Creating model efficiency comparison...")
        
        # Relative cost estimates (gpt-4o = 1.0)
        model_costs = {
            'gpt-4o': 1.0,
            'gpt-4o-mini': 0.15,
            'gpt-3.5-turbo': 0.05
        }
        
        evaluations = test_summary['evaluations']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model in evaluations.keys():
            if model in model_costs:
                acc = evaluations[model]['accuracy'] * 100
                cost = model_costs[model]
                
                # Plot point
                ax.scatter(cost, acc, s=300, alpha=0.6, label=model)
                ax.text(cost, acc, model, ha='center', va='center', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Relative Cost (GPT-4o = 1.0)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Model Efficiency: Accuracy vs. Cost', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(-0.1, 1.2)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        # Add baseline line
        baseline_acc = test_summary['baseline_gpt4_paper']['accuracy'] * 100
        ax.axhline(y=baseline_acc, color='gray', linestyle='--', 
                  alpha=0.7, label='Baseline (GPT-4 Paper)', linewidth=2)
        
        plt.tight_layout()
        
        output_file = self.output_dir / "model_efficiency.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved to: {output_file}")
    
    def create_summary_report(self, grid_summary: Dict, test_summary: Dict):
        """Create text summary report."""
        
        print("\n📄 Creating summary report...")
        
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE EVALUATION SUMMARY")
        report.append("="*80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report.append("\n" + "="*80)
        report.append("1. GRID SEARCH RESULTS")
        report.append("="*80)
        
        optimal_config = grid_summary['optimal_config']
        report.append(f"\nOptimal Configuration:")
        report.append(f"  • Candidate ID: {optimal_config['optimal_candidate_id']}")
        report.append(f"  • Candidate Name: {optimal_config['optimal_candidate_name']}")
        report.append(f"  • Model: {optimal_config['optimal_model']}")
        report.append(f"  • Validation Accuracy: {optimal_config['validation_accuracy']:.4f} ({optimal_config['validation_accuracy']*100:.2f}%)")
        
        report.append(f"\nAll Grid Search Results:")
        for r in sorted(optimal_config['all_results'], key=lambda x: x['accuracy'], reverse=True):
            report.append(f"  • C{r['candidate_id']} + {r['model']}: {r['accuracy']*100:.2f}%")
        
        report.append("\n" + "="*80)
        report.append("2. TEST SET RESULTS")
        report.append("="*80)
        
        baseline = test_summary['baseline_gpt4_paper']
        report.append(f"\nBaseline (MedCalc Paper):")
        report.append(f"  • GPT-4 One-Shot: {baseline['accuracy']*100:.2f}% ({baseline['correct']}/{baseline['total']})")
        
        report.append(f"\nOur Results:")
        for model, eval_data in test_summary['evaluations'].items():
            improvement = (eval_data['accuracy'] - baseline['accuracy']) * 100
            report.append(f"  • {model}: {eval_data['accuracy']*100:.2f}% ({eval_data['correct']}/{eval_data['total']}) [{improvement:+.2f}% vs baseline]")
        
        report.append("\n" + "="*80)
        report.append("3. KEY FINDINGS")
        report.append("="*80)
        
        optimal_model = optimal_config['optimal_model']
        if optimal_model in test_summary['evaluations']:
            optimal_test_acc = test_summary['evaluations'][optimal_model]['accuracy']
            improvement = (optimal_test_acc - baseline['accuracy']) * 100
            
            report.append(f"\n✓ Best configuration achieved {optimal_test_acc*100:.2f}% accuracy")
            report.append(f"✓ Improvement of {improvement:+.2f}% over baseline")
            report.append(f"✓ Used {optimal_config['optimal_candidate_name']} prompt strategy")
            report.append(f"✓ Optimal model: {optimal_model}")
        
        report_text = '\n'.join(report)
        
        output_file = self.output_dir / "summary_report.txt"
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print(f"   ✓ Saved to: {output_file}")
        
        # Also print to console
        print("\n" + report_text)
    
    def run_complete_visualization(self):
        """Run all visualizations."""
        
        print("="*80)
        print("COMPREHENSIVE RESULTS VISUALIZATION")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Load data
        print("📋 Loading data...")
        grid_summary = self.load_grid_search_summary()
        test_summary = self.load_test_summary()
        print("   ✓ Data loaded")
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        
        self.visualize_grid_search_heatmap(grid_summary)
        self.visualize_test_comparison(grid_summary, test_summary)
        self.visualize_by_category(test_summary)
        self.visualize_improvement_breakdown(grid_summary, test_summary)
        self.visualize_model_efficiency(test_summary)
        self.create_summary_report(grid_summary, test_summary)
        
        print(f"\n{'='*80}")
        print("VISUALIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"\n📁 All visualizations saved to: {self.output_dir}/")
        
        return {"output_dir": self.output_dir}


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Results Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--grid-search-dir',
        type=str,
        required=True,
        help='Directory containing grid search results'
    )
    
    parser.add_argument(
        '--test-results-dir',
        type=str,
        required=True,
        help='Directory containing test evaluation results'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for visualizations (default: auto-generated)'
    )
    
    args = parser.parse_args()
    
    visualizer = ResultsVisualizer(
        grid_search_dir=args.grid_search_dir,
        test_results_dir=args.test_results_dir,
        output_dir=args.output_dir
    )
    
    results = visualizer.run_complete_visualization()
    
    print(f"\n✅ Visualization completed successfully!")
    print(f"📁 Outputs: {results['output_dir']}")


if __name__ == "__main__":
    main()

