#!/usr/bin/env python3
"""
Complete Prompt Optimization and Evaluation Pipeline

This master script orchestrates the entire optimization pipeline:
1. Iterative prompt refinement (10 iterations per type) -> 5 candidate prompts
2. Create validation split (100 examples)
3. Grid search on validation set (5 candidates x 3 models)
4. Full test evaluation with optimal (prompt, model) pair
5. Comprehensive visualization and comparison

Usage:
    python run_complete_optimization_pipeline.py \
        --training-results-dir <path> \
        --num-refinement-iterations 10 \
        --num-candidates 5 \
        --validation-size 100 \
        --test-all-models
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import argparse
import json


def run_command(cmd, description, check=True):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=check, capture_output=False, text=True)
        print(f"\n✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with error code {e.returncode}")
        if check:
            sys.exit(1)
        return False


def find_latest_dir(base_dir: Path, pattern: str) -> Path:
    """Find the most recently created directory matching a pattern."""
    dirs = sorted(base_dir.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    if dirs:
        return dirs[0]
    raise FileNotFoundError(f"No directory found matching pattern: {pattern}")


def main():
    parser = argparse.ArgumentParser(
        description="Complete Prompt Optimization and Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  # Full pipeline with 10 refinement iterations, 5 candidates, test all models
  python run_complete_optimization_pipeline.py \\
    --training-results-dir outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \\
    --num-refinement-iterations 10 \\
    --num-candidates 5 \\
    --validation-size 100 \\
    --test-all-models
  
  # Quick test with fewer iterations
  python run_complete_optimization_pipeline.py \\
    --training-results-dir outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \\
    --num-refinement-iterations 3 \\
    --num-candidates 3 \\
    --validation-size 50
        """
    )
    
    parser.add_argument(
        '--training-results-dir',
        type=str,
        required=True,
        help='Directory containing training evaluation results (contrastive edits output)'
    )
    
    parser.add_argument(
        '--num-refinement-iterations',
        type=int,
        default=10,
        help='Number of refinement iterations per prompt type (default: 10)'
    )
    
    parser.add_argument(
        '--num-candidates',
        type=int,
        default=5,
        help='Number of candidate prompts to create (default: 5)'
    )
    
    parser.add_argument(
        '--validation-size',
        type=int,
        default=100,
        help='Number of validation examples (default: 100)'
    )
    
    parser.add_argument(
        '--test-all-models',
        action='store_true',
        help='Test all 3 models on test set (default: only optimal model)'
    )
    
    parser.add_argument(
        '--skip-refinement',
        action='store_true',
        help='Skip refinement if already done'
    )
    
    parser.add_argument(
        '--refined-prompts-dir',
        type=str,
        default=None,
        help='Directory with refined prompts (required if --skip-refinement)'
    )
    
    parser.add_argument(
        '--skip-validation-split',
        action='store_true',
        help='Skip validation split if already created'
    )
    
    parser.add_argument(
        '--validation-split-dir',
        type=str,
        default=None,
        help='Directory with validation split (required if --skip-validation-split)'
    )
    
    parser.add_argument(
        '--skip-grid-search',
        action='store_true',
        help='Skip grid search if already done'
    )
    
    parser.add_argument(
        '--grid-search-dir',
        type=str,
        default=None,
        help='Directory with grid search results (required if --skip-grid-search)'
    )
    
    parser.add_argument(
        '--output-base-dir',
        type=str,
        default=None,
        help='Base directory for all outputs (default: ./optimization_results_TIMESTAMP)'
    )
    
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("\nPlease set it with:")
        print('  export OPENAI_API_KEY="your-api-key-here"')
        sys.exit(1)
    
    # Create base output directory
    if args.output_base_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_base_dir = Path(__file__).parent / "optimization_results" / f"run_{timestamp}"
    else:
        output_base_dir = Path(args.output_base_dir)
    
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("COMPLETE PROMPT OPTIMIZATION AND EVALUATION PIPELINE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📋 Configuration:")
    print(f"   • Training results: {args.training_results_dir}")
    print(f"   • Refinement iterations: {args.num_refinement_iterations}")
    print(f"   • Candidate prompts: {args.num_candidates}")
    print(f"   • Validation size: {args.validation_size}")
    print(f"   • Test all models: {args.test_all_models}")
    print(f"   • Output base dir: {output_base_dir}")
    print("="*80)
    
    # Track directories for each stage
    stage_dirs = {}
    
    # =========================================================================
    # STAGE 1: ITERATIVE PROMPT REFINEMENT
    # =========================================================================
    if not args.skip_refinement:
        print("\n" + "🔧 STAGE 1: ITERATIVE PROMPT REFINEMENT".center(80))
        print("="*80)
        
        refinement_output_dir = output_base_dir / "1_refined_prompts"
        
        cmd = [
            sys.executable,
            'prompt_refinement_pipeline.py',
            '--results-dir', args.training_results_dir,
            '--batch-size', '17',
            '--max-iterations', str(args.num_refinement_iterations),
            '--num-candidates', str(args.num_candidates),
            '--output-dir', str(refinement_output_dir)
        ]
        
        if not run_command(cmd, "Prompt Refinement"):
            print("\n❌ Pipeline failed at refinement stage")
            sys.exit(1)
        
        stage_dirs['refined_prompts'] = refinement_output_dir
    else:
        if not args.refined_prompts_dir:
            print("\n❌ Error: --refined-prompts-dir required when using --skip-refinement")
            sys.exit(1)
        stage_dirs['refined_prompts'] = Path(args.refined_prompts_dir)
        print(f"\n⏭️  Skipping refinement, using: {stage_dirs['refined_prompts']}")
    
    # =========================================================================
    # STAGE 2: CREATE VALIDATION SPLIT
    # =========================================================================
    if not args.skip_validation_split:
        print("\n" + "📊 STAGE 2: CREATE VALIDATION SPLIT".center(80))
        print("="*80)
        
        validation_output_dir = output_base_dir / "2_validation_split"
        
        cmd = [
            sys.executable,
            'create_validation_split.py',
            '--training-results-dir', args.training_results_dir,
            '--validation-size', str(args.validation_size),
            '--output-dir', str(validation_output_dir),
            '--seed', '42'
        ]
        
        if not run_command(cmd, "Validation Split Creation"):
            print("\n❌ Pipeline failed at validation split stage")
            sys.exit(1)
        
        stage_dirs['validation_split'] = validation_output_dir
    else:
        if not args.validation_split_dir:
            print("\n❌ Error: --validation-split-dir required when using --skip-validation-split")
            sys.exit(1)
        stage_dirs['validation_split'] = Path(args.validation_split_dir)
        print(f"\n⏭️  Skipping validation split, using: {stage_dirs['validation_split']}")
    
    # =========================================================================
    # STAGE 3: GRID SEARCH ON VALIDATION SET
    # =========================================================================
    if not args.skip_grid_search:
        print("\n" + "🔍 STAGE 3: GRID SEARCH ON VALIDATION SET".center(80))
        print("="*80)
        
        grid_search_output_dir = output_base_dir / "3_grid_search"
        
        cmd = [
            sys.executable,
            'grid_search_validation.py',
            '--refined-prompts-dir', str(stage_dirs['refined_prompts']),
            '--validation-split-dir', str(stage_dirs['validation_split']),
            '--training-results-dir', args.training_results_dir,
            '--output-dir', str(grid_search_output_dir),
            '--batch-size', '5',
            '--num-positive', '1',
            '--num-negative', '1'
        ]
        
        if not run_command(cmd, "Grid Search on Validation Set"):
            print("\n❌ Pipeline failed at grid search stage")
            sys.exit(1)
        
        stage_dirs['grid_search'] = grid_search_output_dir
    else:
        if not args.grid_search_dir:
            print("\n❌ Error: --grid-search-dir required when using --skip-grid-search")
            sys.exit(1)
        stage_dirs['grid_search'] = Path(args.grid_search_dir)
        print(f"\n⏭️  Skipping grid search, using: {stage_dirs['grid_search']}")
    
    # =========================================================================
    # STAGE 4: FULL TEST SET EVALUATION
    # =========================================================================
    print("\n" + "🎯 STAGE 4: FULL TEST SET EVALUATION".center(80))
    print("="*80)
    
    test_eval_output_dir = output_base_dir / "4_test_evaluation"
    
    cmd = [
        sys.executable,
        'full_test_evaluation.py',
        '--grid-search-dir', str(stage_dirs['grid_search']),
        '--training-results-dir', args.training_results_dir,
        '--output-dir', str(test_eval_output_dir),
        '--batch-size', '10',
        '--save-frequency', '50'
    ]
    
    if args.test_all_models:
        cmd.append('--test-all-models')
    
    if not run_command(cmd, "Full Test Set Evaluation"):
        print("\n❌ Pipeline failed at test evaluation stage")
        sys.exit(1)
    
    stage_dirs['test_evaluation'] = test_eval_output_dir
    
    # =========================================================================
    # STAGE 5: COMPREHENSIVE VISUALIZATION
    # =========================================================================
    print("\n" + "📈 STAGE 5: COMPREHENSIVE VISUALIZATION".center(80))
    print("="*80)
    
    viz_output_dir = output_base_dir / "5_visualizations"
    
    cmd = [
        sys.executable,
        'visualize_complete_results.py',
        '--grid-search-dir', str(stage_dirs['grid_search']),
        '--test-results-dir', str(stage_dirs['test_evaluation']),
        '--output-dir', str(viz_output_dir)
    ]
    
    if not run_command(cmd, "Comprehensive Visualization", check=False):
        print("\n⚠️  Warning: Visualization failed, but pipeline completed")
    else:
        stage_dirs['visualizations'] = viz_output_dir
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    
    # Load and display key results
    try:
        grid_search_summary_file = stage_dirs['grid_search'] / "grid_search_summary.json"
        with open(grid_search_summary_file, 'r') as f:
            grid_summary = json.load(f)
        
        test_eval_summary_file = stage_dirs['test_evaluation'] / "test_evaluation_summary.json"
        with open(test_eval_summary_file, 'r') as f:
            test_summary = json.load(f)
        
        optimal_config = grid_summary['optimal_config']
        baseline_acc = test_summary['baseline_gpt4_paper']['accuracy']
        optimal_model = optimal_config['optimal_model']
        
        if optimal_model in test_summary['evaluations']:
            optimal_acc = test_summary['evaluations'][optimal_model]['accuracy']
            improvement = optimal_acc - baseline_acc
            
            print(f"\n🎉 KEY RESULTS:")
            print(f"   • Optimal Prompt: Candidate {optimal_config['optimal_candidate_id']} ({optimal_config['optimal_candidate_name']})")
            print(f"   • Optimal Model: {optimal_model}")
            print(f"   • Validation Accuracy: {optimal_config['validation_accuracy']:.2%}")
            print(f"   • Test Set Accuracy: {optimal_acc:.2%}")
            print(f"   • Baseline (Paper): {baseline_acc:.2%}")
            print(f"   • Improvement: {improvement:+.2%}")
    except Exception as e:
        print(f"\n⚠️  Could not load final results: {e}")
    
    print(f"\n📁 Output Directory Structure:")
    print(f"   {output_base_dir}/")
    for stage_name, stage_dir in stage_dirs.items():
        print(f"   ├── {stage_dir.name}/")
    
    print(f"\n📄 Key Files:")
    print(f"   • Candidate Prompts: {stage_dirs['refined_prompts']}/final/candidates/")
    print(f"   • Optimal Config: {stage_dirs['grid_search']}/optimal/optimal_config.json")
    print(f"   • Optimal Prompt: {stage_dirs['grid_search']}/optimal/optimal_prompt.txt")
    print(f"   • Test Results: {stage_dirs['test_evaluation']}/test_evaluation_summary.json")
    if 'visualizations' in stage_dirs:
        print(f"   • Visualizations: {stage_dirs['visualizations']}/")
    
    # Save pipeline summary
    pipeline_summary = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "training_results_dir": args.training_results_dir,
            "num_refinement_iterations": args.num_refinement_iterations,
            "num_candidates": args.num_candidates,
            "validation_size": args.validation_size,
            "test_all_models": args.test_all_models
        },
        "stage_directories": {k: str(v) for k, v in stage_dirs.items()},
        "output_base_dir": str(output_base_dir)
    }
    
    summary_file = output_base_dir / "pipeline_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(pipeline_summary, f, indent=2)
    
    print(f"\n💾 Pipeline summary saved to: {summary_file}")
    
    print(f"\n{'='*80}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print(f"\n✅ All phases completed successfully!")
    print(f"🎉 Results ready for publication!")


if __name__ == "__main__":
    main()

