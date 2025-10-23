#!/usr/bin/env python3
"""
Complete Prompt Refinement and Evaluation Pipeline

This script orchestrates the entire pipeline:
1. Iterative prompt refinement using feedback
2. Contrastive few-shot evaluation on test set
3. Publication-ready visualizations

Usage:
    python run_complete_pipeline.py \
        --training-results-dir outputs/medcalc_contrastive_edits_evaluation_TIMESTAMP \
        --batch-size 17 \
        --skip-refinement  # Optional: skip refinement if already done
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import subprocess


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with error code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Complete Prompt Refinement and Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python run_complete_pipeline.py \\
    --training-results-dir outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \\
    --batch-size 17
        """
    )
    
    parser.add_argument(
        '--training-results-dir',
        type=str,
        required=True,
        help='Directory containing initial training evaluation results'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=17,
        help='Batch size for iterative refinement (default: 17)'
    )
    
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=None,
        help='Maximum refinement iterations (default: None = use all examples)'
    )
    
    parser.add_argument(
        '--skip-refinement',
        action='store_true',
        help='Skip refinement step (if already done)'
    )
    
    parser.add_argument(
        '--refined-prompts-dir',
        type=str,
        default=None,
        help='Directory with refined prompts (required if --skip-refinement)'
    )
    
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation step (if already done)'
    )
    
    parser.add_argument(
        '--evaluation-dir',
        type=str,
        default=None,
        help='Directory with evaluation results (required if --skip-evaluation)'
    )
    
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    print("="*80)
    print("COMPLETE PROMPT REFINEMENT AND EVALUATION PIPELINE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training results: {args.training_results_dir}")
    print("="*80)
    
    # Step 1: Prompt Refinement
    if not args.skip_refinement:
        print("\n" + "🔧 PHASE 1: ITERATIVE PROMPT REFINEMENT".center(80))
        print("="*80)
        
        cmd = [
            sys.executable,
            'prompt_refinement_pipeline.py',
            '--results-dir', args.training_results_dir,
            '--batch-size', str(args.batch_size)
        ]
        
        if args.max_iterations:
            cmd.extend(['--max-iterations', str(args.max_iterations)])
        
        if not run_command(cmd, "Prompt Refinement"):
            print("\n❌ Pipeline failed at refinement stage")
            sys.exit(1)
        
        # Find the output directory
        outputs_dir = Path(__file__).parent.parent / "outputs"
        refined_dirs = sorted(outputs_dir.glob("refined_prompts_*"), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if refined_dirs:
            refined_prompts_dir = refined_dirs[0]
            print(f"\n📁 Refined prompts directory: {refined_prompts_dir}")
        else:
            print("\n❌ Could not find refined prompts directory")
            sys.exit(1)
    else:
        if not args.refined_prompts_dir:
            print("\n❌ Error: --refined-prompts-dir required when using --skip-refinement")
            sys.exit(1)
        refined_prompts_dir = Path(args.refined_prompts_dir)
        print(f"\n⏭️  Skipping refinement, using: {refined_prompts_dir}")
    
    # Step 2: Contrastive Few-Shot Evaluation
    if not args.skip_evaluation:
        print("\n" + "📊 PHASE 2: CONTRASTIVE FEW-SHOT EVALUATION".center(80))
        print("="*80)
        
        cmd = [
            sys.executable,
            'contrastive_few_shot_evaluation.py',
            '--refined-prompts-dir', str(refined_prompts_dir),
            '--training-results-dir', args.training_results_dir
        ]
        
        if not run_command(cmd, "Contrastive Few-Shot Evaluation"):
            print("\n❌ Pipeline failed at evaluation stage")
            sys.exit(1)
        
        # Find the evaluation output directory
        outputs_dir = Path(__file__).parent.parent / "outputs"
        eval_dirs = sorted(outputs_dir.glob("contrastive_evaluation_*"), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if eval_dirs:
            evaluation_dir = eval_dirs[0]
            print(f"\n📁 Evaluation directory: {evaluation_dir}")
        else:
            print("\n❌ Could not find evaluation directory")
            sys.exit(1)
    else:
        if not args.evaluation_dir:
            print("\n❌ Error: --evaluation-dir required when using --skip-evaluation")
            sys.exit(1)
        evaluation_dir = Path(args.evaluation_dir)
        print(f"\n⏭️  Skipping evaluation, using: {evaluation_dir}")
    
    # Step 3: Visualization
    print("\n" + "📈 PHASE 3: GENERATING VISUALIZATIONS".center(80))
    print("="*80)
    
    cmd = [
        sys.executable,
        'visualize_results.py',
        '--evaluation-dir', str(evaluation_dir)
    ]
    
    if not run_command(cmd, "Visualization Generation"):
        print("\n⚠️  Warning: Visualization failed, but main pipeline completed")
    
    # Final Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\n✅ All phases completed successfully!")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📁 Output Directories:")
    print(f"   • Refined Prompts: {refined_prompts_dir}")
    print(f"   • Evaluation Results: {evaluation_dir}")
    print(f"   • Visualizations: {evaluation_dir / 'visualizations'}")
    print(f"\n🎉 Results are ready for publication!")


if __name__ == "__main__":
    main()

