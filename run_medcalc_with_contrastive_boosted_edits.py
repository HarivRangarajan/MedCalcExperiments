#!/usr/bin/env python3
"""
Simple Runner Script for MedCalc-Bench Evaluation with Contrastive Boosted Edits

This script provides an easy way to run the MedCalc evaluation pipeline
with sensible defaults and clear output.
"""

import sys
import os
from pathlib import Path

# Add the main script to the path
sys.path.insert(0, str(Path(__file__).parent))

from medcalc_with_contrastive_boosted_edits import MedCalcContrastiveEvaluationPipeline

def main():
    print("🏥 MedCalc-Bench with Contrastive Boosted Edits Evaluation")
    print("="*60)
    
    # Check if we're in the right directory
    if not Path("MedCalc-Bench").exists():
        print("❌ Error: MedCalc-Bench directory not found!")
        print("   Please run this script from the medcalc-evaluation directory")
        return
    
    # Try to get API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ OpenAI API key not found!")
        print("   Please set the OPENAI_API_KEY environment variable")
        return
    
    print("✅ API key found")
    
    # Configure sample size (adjust as needed for your use case)
    SAMPLE_SIZE = 10  # Start small for testing, increase to 500 for full evaluation
    MODEL = "OpenAI/gpt-4o"
    
    print(f"\n📋 Configuration:")
    print(f"   • Sample size: {SAMPLE_SIZE} examples from train data")
    print(f"   • Model: {MODEL}")
    print(f"   • Prompt types: Original, Chain of Thought, Chain of Draft")
    
    # Initialize pipeline with configuration
    pipeline = MedCalcContrastiveEvaluationPipeline(
        api_key=api_key,
        sample_size=SAMPLE_SIZE,
        model=MODEL
    )
    
    print(f"\n🚀 Starting evaluation...")
    
    try:
        results = pipeline.run_complete_evaluation()
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"\n📊 Quick Results Summary:")
        
        # Show quick summary
        evaluation_results = results["evaluation_results"]
        
        print(f"\n   Prompt Type Performance:")
        for prompt_type, result in evaluation_results.items():
            acc = result['accuracy']
            correct = result['correct']
            total = result['total']
            print(f"     • {prompt_type:20}: {acc:6.1%} ({correct}/{total} correct)")
        
        # Best performer
        best = max(evaluation_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n   🏆 Best Performer: {best[0]} ({best[1]['accuracy']:.1%})")
        
        # Show file locations
        print(f"\n📁 Output Files:")
        print(f"   • All results: {results['output_directory']}/")
        print(f"   • Correct responses: {results['output_directory']}/correct/")
        print(f"   • Incorrect responses: {results['output_directory']}/incorrect/")
        print(f"   • Enhanced prompts: {results['output_directory']}/prompts/")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 