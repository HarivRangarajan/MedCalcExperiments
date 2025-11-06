"""Base evaluation pipeline with common functionality for all datasets."""

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from ..utils.evaluation_utils import clean_prompt_name


class BaseEvaluationPipeline(ABC):
    """Abstract base class for dataset evaluation pipelines."""
    
    # Configuration constants
    DEFAULT_LLM_JUDGE_SAMPLE_SIZE = 20
    DEFAULT_RESPONSE_GENERATION_MAX = 5
    DEFAULT_SAMPLE_SIZE = 20
    DEFAULT_BUDGET_LIMIT = 10.0
    DEFAULT_MODEL = "gpt-4o"
    
    def __init__(self, api_key: str, output_dir: str = None,
                 sample_size: int = None,
                 max_responses: int = None,
                 llm_judge_sample_size: int = None,
                 budget_limit: float = None,
                 model: str = None):
        """Initialize the evaluation pipeline.
        
        Args:
            api_key: OpenAI API key
            output_dir: Output directory for results
            sample_size: Number of examples to evaluate
            max_responses: Max responses to generate per technique
            llm_judge_sample_size: Max responses to evaluate with LLM judge
            budget_limit: Budget limit in USD
            model: Model to use for evaluations
        """
        self.api_key = api_key
        
        # Set configuration parameters with defaults
        self.sample_size = sample_size if sample_size is not None else self.DEFAULT_SAMPLE_SIZE
        self.max_responses = max_responses if max_responses is not None else self.DEFAULT_RESPONSE_GENERATION_MAX
        self.llm_judge_sample_size = llm_judge_sample_size if llm_judge_sample_size is not None else self.DEFAULT_LLM_JUDGE_SAMPLE_SIZE
        self.budget_limit = budget_limit if budget_limit is not None else self.DEFAULT_BUDGET_LIMIT
        self.model = model if model is not None else self.DEFAULT_MODEL
        
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "outputs" / f"{self.dataset_name}_evaluation_{timestamp}"
        
        self.output_dir = Path(output_dir)
        self._setup_output_directory()
        
        # Initialize LLM Judge if available
        self.llm_judge = self._initialize_llm_judge()
        
        print(f"✅ Pipeline initialized with output directory: {self.output_dir}")
    
    def _setup_output_directory(self) -> None:
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        subdirs = ["data", "prompts", "responses", "evaluations", "visualizations", "reports", "judge_prompts"]
        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(exist_ok=True)
    
    def _initialize_llm_judge(self):
        """Initialize LLM Judge if available. Override in subclasses if needed."""
        try:
            from modules.custom_llm_judge import CustomLLMJudge as LLMJudge
            judge = LLMJudge(api_key=self.api_key, model=self.model)
            print(f"✅ Custom LLM Judge initialized with model: {self.model}")
            return judge
        except (ImportError, Exception) as e:
            print(f"⚠️  LLM Judge not available: {e}")
            return None
    
    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Name of the dataset. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def load_data(self, sample_size: int = None):
        """Load and prepare dataset. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def create_context(self):
        """Create PromptContext for tasks. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def generate_responses(self, data, enhanced_prompts: Dict[str, Any], 
                          original_prompts: Dict[str, str], max_examples: int = None):
        """Generate responses using prompts. Must be implemented by subclasses."""
        pass
    
    def evaluate_accuracy(self, responses: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
        """Evaluate numerical accuracy. Override in subclasses for custom evaluation."""
        print("\n🎯 Evaluating Numerical Accuracy")
        print("="*60)
        
        accuracy_results = {}
        
        for prompt_type, response_list in responses.items():
            print(f"\n   📊 Evaluating {prompt_type}...")
            
            correct = 0
            total = 0
            category_stats = {}
            
            for response_data in response_list:
                try:
                    # Extract numerical answer from response
                    from ..utils.evaluation_utils import extract_numerical_answer, evaluate_with_tolerance
                    predicted_answer = extract_numerical_answer(response_data['response'])
                    ground_truth = float(response_data['ground_truth_answer'])
                    
                    # Evaluate with tolerance
                    is_correct = evaluate_with_tolerance(predicted_answer, ground_truth)
                    
                    if is_correct:
                        correct += 1
                    total += 1
                    
                    # Track by category if available
                    if 'category' in response_data:
                        category = response_data['category']
                        if category not in category_stats:
                            category_stats[category] = {'correct': 0, 'total': 0}
                        category_stats[category]['total'] += 1
                        if is_correct:
                            category_stats[category]['correct'] += 1
                    
                except Exception as e:
                    print(f"      ⚠️  Error evaluating response: {str(e)}")
                    total += 1
                    continue
            
            # Calculate accuracy
            overall_accuracy = correct / total if total > 0 else 0
            
            # Calculate category accuracies
            category_accuracies = {}
            for cat, stats_dict in category_stats.items():
                category_accuracies[cat] = stats_dict['correct'] / stats_dict['total'] if stats_dict['total'] > 0 else 0
            
            accuracy_results[prompt_type] = {
                'overall_accuracy': overall_accuracy,
                'correct': correct,
                'total': total,
                'category_accuracies': category_accuracies
            }
            
            print(f"      ✅ Overall accuracy: {overall_accuracy:.1%} ({correct}/{total})")
            for cat, acc in category_accuracies.items():
                stats_dict = category_stats[cat]
                print(f"         • {cat}: {acc:.1%} ({stats_dict['correct']}/{stats_dict['total']})")
        
        # Save accuracy results
        accuracy_file = self.output_dir / "evaluations" / "accuracy_results.json"
        with open(accuracy_file, 'w') as f:
            json.dump(accuracy_results, f, indent=2)
        
        return accuracy_results
    
    def evaluate_with_llm_judge(self, responses: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Evaluate response quality using LLM-as-a-judge."""
        import random
        
        if not self.llm_judge:
            print("\n⚠️  Skipping LLM-as-a-judge evaluation (not available)")
            return {}
        
        print("\n⚖️ LLM-as-a-Judge Evaluation")
        print("="*60)
        
        judge_system_prompt = self.llm_judge.system_prompt
        
        # Save the judge system prompt
        judge_prompt_file = self.output_dir / "judge_prompts" / "llm_judge_system_prompt.txt"
        with open(judge_prompt_file, 'w') as f:
            f.write(judge_system_prompt)
        
        judge_results = {}
        sample_interactions = []
        
        for prompt_type, response_list in responses.items():
            print(f"\n   🔍 Judging {prompt_type}...")
            
            evaluations = []
            sample_size = min(self.llm_judge_sample_size, len(response_list))
            sampled_responses = random.sample(response_list, sample_size)
            
            for i, response_data in enumerate(sampled_responses):
                try:
                    label, reason = self.llm_judge.evaluate_medical_response(
                        patient_note=response_data.get('patient_note', ''),
                        question=response_data.get('question', ''),
                        ai_response=response_data['response'],
                        ground_truth_answer=response_data['ground_truth_answer'],
                        ground_truth_explanation=response_data.get('ground_truth_explanation', '')
                    )
                    
                    evaluation_entry = {
                        **response_data,
                        "llm_judge_label": label if label is not None else 0,
                        "llm_judge_reason": reason if reason is not None else "Evaluation failed",
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    evaluations.append(evaluation_entry)
                    
                    if len(sample_interactions) < 5:
                        sample_interactions.append({
                            "technique": prompt_type,
                            "ai_response": response_data['response'][:500] + "...",
                            "judge_label": label,
                            "judge_reason": reason,
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    if (i + 1) % 10 == 0:
                        print(f"      ✅ Evaluated {i + 1}/{len(sampled_responses)}")
                    
                except Exception as e:
                    print(f"      ❌ Judge evaluation error: {str(e)}")
                    evaluation_entry = {
                        **response_data,
                        "llm_judge_label": 0,
                        "llm_judge_reason": f"Evaluation failed: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }
                    evaluations.append(evaluation_entry)
                    continue
            
            judge_results[prompt_type] = evaluations
            
            if evaluations:
                pass_rate = sum(1 for e in evaluations if e["llm_judge_label"] == 1) / len(evaluations)
                print(f"      ✅ Judge pass rate: {pass_rate:.1%} ({len(evaluations)} evaluated)")
        
        # Save judge results
        judge_file = self.output_dir / "evaluations" / "llm_judge_results.json"
        with open(judge_file, 'w') as f:
            json.dump(judge_results, f, indent=2)
        
        return judge_results
    
    def create_visualizations(self, accuracy_results: Dict[str, Dict[str, float]], 
                             judge_results: Dict[str, Any] = None) -> None:
        """Create comprehensive visualizations of results."""
        print("\n📊 Creating Visualizations")
        print("="*60)
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        self._plot_overall_accuracy(accuracy_results)
        self._plot_category_accuracy(accuracy_results)
        self._plot_prompt_comparison(accuracy_results)
        
        if judge_results:
            self._plot_judge_results(judge_results)
        
        self._plot_statistical_tests(accuracy_results)
        
        print("✅ All visualizations saved to visualizations/ directory")
    
    def _plot_overall_accuracy(self, accuracy_results: Dict[str, Dict[str, float]]) -> None:
        """Plot overall accuracy comparison."""
        prompt_types = list(accuracy_results.keys())
        accuracies = [accuracy_results[pt]['overall_accuracy'] for pt in prompt_types]
        display_names = [clean_prompt_name(pt) for pt in prompt_types]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(display_names, accuracies, alpha=0.8)
        
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Overall Accuracy Comparison: Original vs Enhanced Prompts', 
                 fontsize=16, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Prompt Type', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, max(accuracies) * 1.2 if accuracies else 1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / "visualizations" / "overall_accuracy.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_category_accuracy(self, accuracy_results: Dict[str, Dict[str, float]]) -> None:
        """Plot category-wise accuracy comparison."""
        all_categories = set()
        for results in accuracy_results.values():
            all_categories.update(results['category_accuracies'].keys())
        
        if not all_categories:
            return
        
        categories = sorted(list(all_categories))
        prompt_types = list(accuracy_results.keys())
        
        heatmap_data = []
        for prompt_type in prompt_types:
            row = []
            for category in categories:
                acc = accuracy_results[prompt_type]['category_accuracies'].get(category, 0)
                row.append(acc)
            heatmap_data.append(row)
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(heatmap_data, 
                   xticklabels=categories,
                   yticklabels=[clean_prompt_name(pt) for pt in prompt_types],
                   annot=True, 
                   fmt='.2f',
                   cmap='RdYlBu_r',
                   cbar_kws={'label': 'Accuracy'})
        
        plt.title('Category-wise Accuracy Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Categories', fontsize=12)
        plt.ylabel('Prompt Types', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / "visualizations" / "category_accuracy_heatmap.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prompt_comparison(self, accuracy_results: Dict[str, Dict[str, float]]) -> None:
        """Plot comparison between original and enhanced prompts."""
        original_results = {k: v for k, v in accuracy_results.items() if k.startswith('original_')}
        enhanced_results = {k: v for k, v in accuracy_results.items() if k.startswith('enhanced_')}
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        orig_names = [k.replace('original_', '') for k in original_results.keys()]
        orig_accs = [v['overall_accuracy'] for v in original_results.values()]
        
        bars1 = ax1.bar(orig_names, orig_accs, alpha=0.8, color='skyblue')
        ax1.set_title('Original Prompts', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_ylim(0, 1)
        
        for bar, acc in zip(bars1, orig_accs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
        
        enh_names = [k.replace('enhanced_', '').replace('_', ' ').title() for k in enhanced_results.keys()]
        enh_accs = [v['overall_accuracy'] for v in enhanced_results.values()]
        
        bars2 = ax2.bar(enh_names, enh_accs, alpha=0.8, color='lightcoral')
        ax2.set_title('Enhanced Prompts', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_ylim(0, 1)
        
        for bar, acc in zip(bars2, enh_accs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "visualizations" / "prompt_type_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_judge_results(self, judge_results: Dict[str, Any]) -> None:
        """Plot LLM-as-a-judge evaluation results."""
        pass_rates = {}
        for prompt_type, evaluations in judge_results.items():
            if evaluations:
                pass_rate = sum(1 for e in evaluations if e["llm_judge_label"] == 1) / len(evaluations)
                pass_rates[prompt_type] = pass_rate
        
        if not pass_rates:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        prompt_names = [clean_prompt_name(pt) for pt in pass_rates.keys()]
        rates = [pass_rates[pt] for pt in pass_rates.keys()]
        
        bars = ax.bar(prompt_names, rates, alpha=0.8, color='skyblue')
        
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Prompt Types', fontsize=12)
        ax.set_ylabel('Pass Rate', fontsize=12)
        ax.set_title('LLM-as-a-Judge Pass Rates by Prompt Type', fontsize=16, fontweight='bold')
        ax.set_ylim(0, max(rates) * 1.2 if rates else 1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / "visualizations" / "llm_judge_results.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_tests(self, accuracy_results: Dict[str, Dict[str, float]]) -> None:
        """Plot statistical significance tests."""
        original_accs = [v['overall_accuracy'] for k, v in accuracy_results.items() if k.startswith('original_')]
        enhanced_accs = [v['overall_accuracy'] for k, v in accuracy_results.items() if k.startswith('enhanced_')]
        
        if len(original_accs) > 1 and len(enhanced_accs) > 1:
            t_stat, p_value = stats.ttest_ind(original_accs, enhanced_accs)
            
            plt.figure(figsize=(10, 6))
            
            data = [original_accs, enhanced_accs]
            labels = ['Original', 'Enhanced']
            
            plt.boxplot(data, labels=labels, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
            
            plt.ylabel('Accuracy', fontsize=12)
            plt.title(f'Statistical Comparison\nt-statistic: {t_stat:.3f}, p-value: {p_value:.3f}', 
                     fontsize=14, fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
            
            if p_value < 0.05:
                plt.text(0.5, max(max(original_accs), max(enhanced_accs)) * 1.1,
                        f'Statistically Significant (p < 0.05)', 
                        ha='center', va='center', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "visualizations" / "statistical_comparison.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_report(self, accuracy_results: Dict[str, Dict[str, float]], 
                       judge_results: Dict[str, Any] = None) -> None:
        """Generate comprehensive evaluation report."""
        print("\n📄 Generating Report")
        print("="*60)
        
        from .reporting import generate_report, create_summary_data
        
        report_content = generate_report(accuracy_results, judge_results, self.dataset_name)
        
        report_file = self.output_dir / "reports" / "evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        summary_data = create_summary_data(accuracy_results, judge_results)
        summary_file = self.output_dir / "reports" / "evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"✅ Report saved to: {report_file}")
        print(f"✅ Summary saved to: {summary_file}")
    
    @abstractmethod
    def run_complete_evaluation(self, **kwargs):
        """Run the complete evaluation pipeline. Must be implemented by subclasses."""
        pass

