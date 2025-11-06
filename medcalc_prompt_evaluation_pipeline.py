#!/usr/bin/env python3
"""
MedCalc-Bench Prompt Evaluation Pipeline (Modular)

This script evaluates different prompt engineering techniques on the MedCalc-Bench dataset,
comparing:
1. Original MedCalc prompts (Direct, Zero-shot CoT, One-shot CoT)
2. PromptEngineer generated prompts (Chain of Thought, Chain of Thoughtlessness, Chain of Draft)

Evaluation includes:
- Built-in accuracy metrics (numerical answer comparison)
- LLM-as-a-judge evaluation for response quality
- Comprehensive visualizations and statistical analysis

Usage:
  python medcalc_prompt_evaluation_pipeline.py --sample-size 300 --output-dir results_experiment1
"""

import sys
import argparse
from pathlib import Path

# Import from modular structure
from utils.api_utils import load_api_key
from modules import MedCalcEvaluationPipeline


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MedCalc-Bench Prompt Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python medcalc_prompt_evaluation_pipeline.py --sample-size 20 --max-responses 5
  python medcalc_prompt_evaluation_pipeline.py --sample-size 10 --max-responses 3 --budget-limit 5.0
  python medcalc_prompt_evaluation_pipeline.py --sample-size 50 --llm-judge-sample-size 30 --output-dir my_experiment
        """
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=20,
        help='Number of MedCalc examples to evaluate (default: 20)'
    )
    
    parser.add_argument(
        '--max-responses',
        type=int,
        default=5,
        help='Maximum responses to generate per technique (default: 5)'
    )
    
    parser.add_argument(
        '--llm-judge-sample-size',
        type=int,
        default=20,
        dest='llm_judge_sample_size',
        help='Maximum responses to evaluate per technique with LLM judge (default: 20)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: auto-generated with timestamp)'
    )
    
    parser.add_argument(
        '--skip-judge',
        action='store_true',
        help='Skip LLM-as-a-judge evaluation to save costs'
    )
    
    parser.add_argument(
        '--budget-limit',
        type=float,
        default=10.0,
        help='Maximum budget in USD (default: $10.00)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Load API key
    OPENAI_API_KEY = load_api_key()
    
    # Check API key
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your-api-key-here":
        print("❌ API key not configured properly")
        print("   Please set your OpenAI API key environment variable")
        sys.exit(1)
    
    print("✅ API key configured")
    
    # Initialize and run pipeline
    pipeline = MedCalcEvaluationPipeline(
        api_key=OPENAI_API_KEY,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        max_responses=args.max_responses,
        llm_judge_sample_size=args.llm_judge_sample_size,
        budget_limit=args.budget_limit
    )
    
    # Temporarily disable judge if requested
    if args.skip_judge:
        pipeline.llm_judge = None
        print("⚠️  LLM-as-a-judge evaluation disabled per user request")
    
    try:
        results = pipeline.run_complete_evaluation()
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📁 Results saved to: {results['output_directory']}")
        
    except KeyboardInterrupt:
        print("\n\n⚡ Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
