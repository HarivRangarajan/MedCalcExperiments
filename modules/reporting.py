"""Report generation utilities for evaluation pipelines."""

import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple

from ..utils.evaluation_utils import clean_prompt_name


def generate_report(accuracy_results: Dict[str, Dict[str, float]], 
                   judge_results: Dict[str, Any] = None,
                   dataset_name: str = "Dataset") -> str:
    """Create detailed text report of evaluation results.
    
    Args:
        accuracy_results: Dictionary of accuracy evaluation results
        judge_results: Optional dictionary of LLM judge results
        dataset_name: Name of the dataset being evaluated
        
    Returns:
        str: Formatted report content
    """
    from scipy import stats
    
    report = f"""
{dataset_name} Prompt Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

OVERVIEW
--------
This report compares the performance of original prompts against 
enhanced prompts on {dataset_name} tasks.

EVALUATION METRICS
-----------------
1. Numerical Accuracy: Exact match with ground truth answers (with tolerance)
2. LLM-as-a-Judge: Qualitative evaluation of response quality

RESULTS SUMMARY
--------------
"""
    
    # Overall accuracy results
    report += "\nNUMERICAL ACCURACY RESULTS:\n"
    report += "-" * 30 + "\n"
    
    for prompt_type, results in accuracy_results.items():
        clean_name = clean_prompt_name(prompt_type)
        accuracy = results['overall_accuracy']
        correct = results['correct']
        total = results['total']
        
        report += f"{clean_name:30}: {accuracy:6.1%} ({correct:3d}/{total:3d})\n"
    
    # Best performing approaches
    best_original = max((k, v) for k, v in accuracy_results.items() if k.startswith('original_'))
    best_enhanced = max((k, v) for k, v in accuracy_results.items() if k.startswith('enhanced_'))
    
    report += f"\nBEST PERFORMERS:\n"
    report += f"Original: {clean_prompt_name(best_original[0])} - {best_original[1]['overall_accuracy']:.1%}\n"
    report += f"Enhanced: {clean_prompt_name(best_enhanced[0])} - {best_enhanced[1]['overall_accuracy']:.1%}\n"
    
    # Category breakdown
    all_categories = set()
    for results in accuracy_results.values():
        all_categories.update(results['category_accuracies'].keys())
    
    if all_categories:
        report += f"\nCATEGORY-WISE PERFORMANCE:\n"
        report += "-" * 30 + "\n"
        
        for category in sorted(all_categories):
            report += f"\n{category}:\n"
            for prompt_type, results in accuracy_results.items():
                acc = results['category_accuracies'].get(category, 0)
                clean_name = clean_prompt_name(prompt_type)
                report += f"  {clean_name:25}: {acc:6.1%}\n"
    
    # LLM Judge results
    if judge_results:
        report += f"\nLLM-AS-A-JUDGE RESULTS:\n"
        report += "-" * 30 + "\n"
        
        for prompt_type, evaluations in judge_results.items():
            if evaluations:
                pass_rate = sum(1 for e in evaluations if e["llm_judge_label"] == 1) / len(evaluations)
                clean_name = clean_prompt_name(prompt_type)
                report += f"{clean_name:30}: {pass_rate:6.1%} pass rate ({len(evaluations)} evaluated)\n"
    
    # Statistical analysis
    original_accs = [v['overall_accuracy'] for k, v in accuracy_results.items() if k.startswith('original_')]
    enhanced_accs = [v['overall_accuracy'] for k, v in accuracy_results.items() if k.startswith('enhanced_')]
    
    if original_accs and enhanced_accs:
        report += f"\nSTATISTICAL ANALYSIS:\n"
        report += "-" * 30 + "\n"
        report += f"Original approaches - Mean: {np.mean(original_accs):.1%}, Std: {np.std(original_accs):.1%}\n"
        report += f"Enhanced approaches - Mean: {np.mean(enhanced_accs):.1%}, Std: {np.std(enhanced_accs):.1%}\n"
        
        if len(original_accs) > 1 and len(enhanced_accs) > 1:
            t_stat, p_value = stats.ttest_ind(original_accs, enhanced_accs)
            report += f"T-test results: t={t_stat:.3f}, p={p_value:.3f}\n"
            if p_value < 0.05:
                report += "Result: Statistically significant difference (p < 0.05)\n"
            else:
                report += "Result: No statistically significant difference\n"
    
    # Conclusions
    report += f"\nCONCLUSIONS:\n"
    report += "-" * 30 + "\n"
    
    improvement = best_enhanced[1]['overall_accuracy'] - best_original[1]['overall_accuracy']
    if improvement > 0:
        report += f"• Enhanced prompts showed improvement of {improvement:.1%}\n"
    else:
        report += f"• Original prompts performed better by {-improvement:.1%}\n"
    
    best = max(accuracy_results.items(), key=lambda x: x[1]['overall_accuracy'])
    report += f"• Best overall approach: {clean_prompt_name(best[0])}\n"
    
    return report


def create_summary_data(accuracy_results: Dict[str, Dict[str, float]], 
                       judge_results: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create summary data for JSON export.
    
    Args:
        accuracy_results: Dictionary of accuracy evaluation results
        judge_results: Optional dictionary of LLM judge results
        
    Returns:
        dict: Summary data dictionary
    """
    summary = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "accuracy_results": accuracy_results,
        "best_performers": {
            "overall": max(accuracy_results.items(), key=lambda x: x[1]['overall_accuracy']),
            "original": max((k, v) for k, v in accuracy_results.items() if k.startswith('original_')),
            "enhanced": max((k, v) for k, v in accuracy_results.items() if k.startswith('enhanced_'))
        }
    }
    
    if judge_results:
        summary["judge_results"] = judge_results
    
    return summary

