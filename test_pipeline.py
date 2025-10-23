#!/usr/bin/env python3
"""
Quick Test Script for Pipeline Verification

This script runs a minimal test of the entire pipeline to verify everything works.
Uses limited iterations and small batch size for speed.
"""

import os
import sys
from pathlib import Path
import subprocess

def run_command(cmd, description):
    """Run a command and check if it succeeds."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"✅ {description} - PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Exit code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    print("="*60)
    print("PIPELINE QUICK TEST")
    print("="*60)
    print("This test runs minimal versions of all pipeline components")
    print("to verify everything is working correctly.\n")
    
    # Check prerequisites
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not set")
        sys.exit(1)
    print("✅ OPENAI_API_KEY is set")
    
    # Check training results directory exists
    training_dir = "/Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434"
    if not Path(training_dir).exists():
        print(f"❌ Training results directory not found: {training_dir}")
        sys.exit(1)
    print(f"✅ Training results directory exists")
    
    # Test 1: Prompt Refinement (limited to 2 iterations)
    print("\n" + "="*60)
    print("TEST 1: Prompt Refinement (2 iterations, batch_size=10)")
    print("="*60)
    
    refinement_cmd = [
        sys.executable,
        'prompt_refinement_pipeline.py',
        '--results-dir', training_dir,
        '--batch-size', '10',
        '--max-iterations', '2',
        '--output-dir', 'outputs/test_refined_prompts'
    ]
    
    if not run_command(refinement_cmd, "Prompt Refinement"):
        print("\n❌ Test failed at refinement stage")
        return False
    
    # Check if unified prompt was created
    unified_prompt_file = Path('outputs/test_refined_prompts/final/unified_prompt.txt')
    if unified_prompt_file.exists():
        print(f"✅ Unified prompt created: {unified_prompt_file}")
        with open(unified_prompt_file, 'r') as f:
            prompt_content = f.read()
        print(f"   Prompt length: {len(prompt_content)} characters")
    else:
        print(f"❌ Unified prompt not found at {unified_prompt_file}")
        return False
    
    print("\n" + "="*60)
    print("TEST 2: Contrastive Evaluation (SKIPPED - would take hours)")
    print("="*60)
    print("⏭️  Skipping full evaluation test (1047 examples)")
    print("   Full evaluation test should be run separately when ready")
    print("   Estimated time: 3-5 hours")
    print("   Estimated cost: $40-60")
    
    # Test 3: Check imports and basic functionality
    print("\n" + "="*60)
    print("TEST 3: Import Verification")
    print("="*60)
    
    try:
        print("Importing contrastive_few_shot_evaluation...")
        from contrastive_few_shot_evaluation import ContrastiveFewShotEvaluator
        print("✅ contrastive_few_shot_evaluation imports successfully")
        
        print("Importing visualize_results...")
        from visualize_results import ResultsVisualizer
        print("✅ visualize_results imports successfully")
        
        print("Importing OpenAI...")
        from openai import OpenAI
        print("✅ OpenAI imports successfully")
        
        print("Importing MedCalc evaluate...")
        sys.path.insert(0, str(Path(__file__).parent / "MedCalc-Bench" / "evaluation"))
        from evaluate import check_correctness
        print("✅ MedCalc evaluate imports successfully")
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Final summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("✅ Prompt refinement: PASSED (2 iterations completed)")
    print("✅ Unified prompt: CREATED")
    print("✅ All imports: WORKING")
    print("⏭️  Full evaluation: READY (run manually when desired)")
    print("\n🎉 All tests passed! Pipeline is ready for full run.")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\nTo run the full pipeline:")
    print(f"\n  python run_complete_pipeline.py \\")
    print(f"    --training-results-dir {training_dir} \\")
    print(f"    --batch-size 17")
    print("\nOR use the test refined prompts directly:")
    print(f"\n  python contrastive_few_shot_evaluation.py \\")
    print(f"    --refined-prompts-dir outputs/test_refined_prompts \\")
    print(f"    --training-results-dir {training_dir}")
    print("\nTest outputs saved to: outputs/test_refined_prompts/")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

