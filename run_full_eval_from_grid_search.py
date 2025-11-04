#!/usr/bin/env python3
"""
Run Full Test Evaluation Using Grid Search Results

This script takes the optimal prompts identified during grid search validation
and runs a full-scale evaluation on the complete test set (1047 examples).

Usage:
    python run_full_eval_from_grid_search.py \
        --grid-search-dir test_pipeline/3_grid_search \
        --training-results-dir /path/to/training/results \
        --output-dir full_eval_from_grid_search \
        --test-all-models
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from full_test_evaluation import FullTestEvaluator


class GridSearchBasedEvaluator:
    """Evaluator that uses grid search results to select optimal prompts."""
    
    def __init__(
        self,
        api_key: str,
        grid_search_dir: Path,
        training_results_dir: Path,
        output_dir: Path,
        test_all_models: bool = True,
        num_positive: int = 1,
        num_negative: int = 1
    ):
        self.api_key = api_key
        self.grid_search_dir = Path(grid_search_dir)
        self.training_results_dir = Path(training_results_dir)
        self.output_dir = Path(output_dir)
        self.test_all_models = test_all_models
        self.num_positive = num_positive
        self.num_negative = num_negative
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in ["results", "responses", "evaluations", "visualizations"]:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        # Load grid search results
        self.grid_search_config = self._load_grid_search_config()
        self.optimal_per_model = self.grid_search_config['optimal_per_model']
        self.optimal_prompts = self._load_optimal_prompts()
        
        print("="*80)
        print("FULL EVALUATION FROM GRID SEARCH RESULTS")
        print("="*80)
        print(f"\n📊 Loaded Grid Search Results:")
        print(f"   • Grid search dir: {self.grid_search_dir}")
        print(f"   • Validation size: {self.grid_search_config.get('validation_size', 'N/A')}")
        
        print(f"\n🎯 Optimal Prompts Per Model:")
        for model, config in self.optimal_per_model.items():
            print(f"   • {model}:")
            print(f"      - Candidate: {config['candidate_id']} ({config['candidate_name']})")
            print(f"      - Val Accuracy: {config['validation_accuracy']:.2%}")
        
        print(f"\n📁 Output directory: {self.output_dir}")
    
    def _load_grid_search_config(self) -> Dict:
        """Load grid search optimal configuration."""
        config_file = self.grid_search_dir / "optimal" / "optimal_config.json"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Grid search config not found at {config_file}")
        
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def _load_optimal_prompts(self) -> Dict[str, str]:
        """Load optimal prompt text for each model."""
        optimal_prompts = {}
        
        for model in self.optimal_per_model.keys():
            model_safe_name = model.replace('-', '_')
            prompt_file = self.grid_search_dir / "optimal" / f"optimal_prompt_{model_safe_name}.txt"
            
            if not prompt_file.exists():
                raise FileNotFoundError(f"Optimal prompt for {model} not found at {prompt_file}")
            
            with open(prompt_file, 'r') as f:
                optimal_prompts[model] = f.read()
        
        return optimal_prompts
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run full evaluation using optimal prompts from grid search."""
        
        print("\n" + "="*80)
        print("STARTING FULL TEST SET EVALUATION")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Determine which models to evaluate
        if self.test_all_models:
            models_to_test = list(self.optimal_prompts.keys())
            print(f"🔬 Evaluating all {len(models_to_test)} models")
        else:
            # Test only the best performing model from validation
            best_model = max(
                self.optimal_per_model.items(),
                key=lambda x: x[1]['validation_accuracy']
            )[0]
            models_to_test = [best_model]
            print(f"🔬 Evaluating only best model: {best_model}")
        
        print(f"\n📋 Models to evaluate:")
        for model in models_to_test:
            config = self.optimal_per_model[model]
            print(f"   • {model}: Candidate {config['candidate_id']} @ {config['validation_accuracy']:.2%} val acc")
        
        # Use the FullTestEvaluator but pass it our grid search directory
        evaluator = FullTestEvaluator(
            api_key=self.api_key,
            grid_search_dir=self.grid_search_dir,
            training_results_dir=self.training_results_dir,
            output_dir=self.output_dir,
            test_all_models=self.test_all_models
        )
        
        # Run the evaluation
        results = evaluator.run_full_evaluation()
        
        # Add grid search metadata to results
        results['grid_search_source'] = {
            'grid_search_dir': str(self.grid_search_dir),
            'validation_size': self.grid_search_config.get('validation_size', 'N/A'),
            'optimal_per_model': self.optimal_per_model
        }
        
        # Save enhanced results
        enhanced_results_file = self.output_dir / "evaluation_with_grid_search_metadata.json"
        with open(enhanced_results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Saved enhanced results to: {enhanced_results_file}")
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of the evaluation results."""
        
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\n📊 Grid Search Validation → Full Test Results:")
        print(f"\n{'Model':<20} {'Candidate':<12} {'Val Acc':<10} {'Test Acc':<10} {'Δ':<10}")
        print("-" * 62)
        
        for model, details in results['evaluation_details'].items():
            val_acc = details['validation_accuracy']
            test_acc = details['test_accuracy']
            delta = test_acc - val_acc
            candidate = f"#{details['optimal_prompt_candidate']}"
            
            delta_str = f"{delta:+.2%}"
            print(f"{model:<20} {candidate:<12} {val_acc:<10.2%} {test_acc:<10.2%} {delta_str:<10}")
        
        # Compare to baseline
        baseline_acc = 533 / 1047  # From paper
        print(f"\n📈 Comparison to Baseline (GPT-4 from paper: {baseline_acc:.2%}):")
        for model, details in results['evaluation_details'].items():
            improvement = details['test_accuracy'] - baseline_acc
            improvement_str = f"{improvement:+.2%}"
            print(f"   • {model}: {details['test_accuracy']:.2%} ({improvement_str} vs baseline)")
        
        print(f"\n💾 All results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run full evaluation using grid search results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  # Use test_pipeline grid search results to evaluate on full test set
  python run_full_eval_from_grid_search.py \\
    --grid-search-dir test_pipeline/3_grid_search \\
    --training-results-dir outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \\
    --output-dir full_eval_from_test_grid_search \\
    --test-all-models
  
  # Or use a different grid search directory
  python run_full_eval_from_grid_search.py \\
    --grid-search-dir full_optimization_run/3_grid_search \\
    --training-results-dir outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \\
    --output-dir full_eval_from_full_grid_search
        """
    )
    
    parser.add_argument(
        '--grid-search-dir',
        type=str,
        required=True,
        help='Directory containing grid search results (e.g., test_pipeline/3_grid_search)'
    )
    
    parser.add_argument(
        '--training-results-dir',
        type=str,
        required=True,
        help='Directory containing training evaluation results'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for full evaluation results'
    )
    
    parser.add_argument(
        '--test-all-models',
        action='store_true',
        help='Test all models (default: only best model from validation)'
    )
    
    parser.add_argument(
        '--num-positive',
        type=int,
        default=1,
        help='Number of positive contrastive examples (default: 1)'
    )
    
    parser.add_argument(
        '--num-negative',
        type=int,
        default=1,
        help='Number of negative contrastive examples (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Create evaluator
    evaluator = GridSearchBasedEvaluator(
        api_key=api_key,
        grid_search_dir=Path(args.grid_search_dir),
        training_results_dir=Path(args.training_results_dir),
        output_dir=Path(args.output_dir),
        test_all_models=args.test_all_models,
        num_positive=args.num_positive,
        num_negative=args.num_negative
    )
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Print summary
    evaluator.print_summary(results)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\n✅ Successfully evaluated {len(results['evaluations'])} model(s)")
    print(f"📁 Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()

