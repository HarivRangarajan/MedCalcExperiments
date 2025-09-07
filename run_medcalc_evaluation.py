#!/usr/bin/env python3
"""
Simple Runner Script for MedCalc-Bench Evaluation

This script provides an easy way to run the MedCalc evaluation pipeline
with sensible defaults and clear output.
"""

import sys
import os
from pathlib import Path

# Add the main script to the path
sys.path.insert(0, str(Path(__file__).parent))

from medcalc_prompt_evaluation_pipeline import MedCalcEvaluationPipeline

def main():
    print("🏥 MedCalc-Bench Prompt Evaluation Pipeline")
    print("="*60)
    
    # Check if we're in the right directory
    if not Path("MedCalc-Bench").exists():
        print("❌ Error: MedCalc-Bench directory not found!")
        print("   Please run this script from the medcalc-evaluation directory")
        return
    
    # Try to get API key from environment or config
    api_key = None
    
    # Try from parent wound care config
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "mohs-llm-as-a-judge"))
        from configs.config import OPENAI_API_KEY
        api_key = OPENAI_API_KEY
    except ImportError:
        pass
    
    # Try from environment
    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key or api_key == "your-api-key-here":
        print("❌ OpenAI API key not found!")
        print("   Please either:")
        print("   1. Set OPENAI_API_KEY environment variable")
        print("   2. Configure it in mohs-llm-as-a-judge/configs/config.py")
        return
    
    print("✅ API key found")
    
    # Configure within budget limits
    SAMPLE_SIZE = 20  # Adjust as needed
    MAX_RESPONSES_PER_TECHNIQUE = 5  # To manage costs during testing
    BUDGET_LIMIT = 10.0  # Maximum cost in USD
    
    print(f"\n⚙️  Configuration:")
    print(f"   • Sample size: {SAMPLE_SIZE} examples")
    print(f"   • Max responses per technique: {MAX_RESPONSES_PER_TECHNIQUE}")
    print(f"   • Budget limit: ${BUDGET_LIMIT}")
    print(f"   • LLM-as-a-judge: Enabled (limited sample for cost control)")
    
    # Initialize pipeline
    pipeline = MedCalcEvaluationPipeline(api_key=api_key)
    
    print(f"\n🚀 Starting evaluation...")
    
    try:
        results = pipeline.run_complete_evaluation(
            sample_size=SAMPLE_SIZE,
            max_responses=MAX_RESPONSES_PER_TECHNIQUE,
            budget_limit=BUDGET_LIMIT
        )
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"\n Quick Results Summary:")
        
        # Show quick summary
        accuracy_results = results["accuracy_results"]
        
        print(f"\n   Original MedCalc Prompts:")
        for prompt_type, result in accuracy_results.items():
            if prompt_type.startswith('original_'):
                name = prompt_type.replace('original_', '').replace('_', ' ').title()
                acc = result['overall_accuracy']
                print(f"     • {name:20}: {acc:6.1%}")
        
        print(f"\n   PromptEngineer Enhanced Prompts:")
        for prompt_type, result in accuracy_results.items():
            if prompt_type.startswith('enhanced_'):
                name = prompt_type.replace('enhanced_', '').replace('_', ' ').title()
                acc = result['overall_accuracy']
                print(f"     • {name:20}: {acc:6.1%}")
        
        # Best performer
        best = max(accuracy_results.items(), key=lambda x: x[1]['overall_accuracy'])
        best_name = best[0].replace('original_', 'Original ').replace('enhanced_', 'Enhanced ').replace('_', ' ').title()
        print(f"\n   🏆 Best Performer: {best_name} ({best[1]['overall_accuracy']:.1%})")
        
        print(f"\n📁 Detailed results saved to: {results['output_directory']}/")
        print(f"   • Visualizations: {results['output_directory']}/visualizations/")
        print(f"   • Reports: {results['output_directory']}/reports/")
        print(f"   • Data: {results['output_directory']}/data/")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 