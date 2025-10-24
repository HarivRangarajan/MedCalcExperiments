#!/usr/bin/env python3
"""
Setup Verification Script

Verifies that all prerequisites are met before running the optimization pipeline.

Usage:
    python verify_setup.py [--training-results-dir <path>]
"""

import sys
import os
from pathlib import Path
import argparse


def check_api_key():
    """Check if OpenAI API key is set."""
    print("\n1. Checking OpenAI API key...")
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("   ❌ OPENAI_API_KEY not set")
        print("   → Set it with: export OPENAI_API_KEY='your-key-here'")
        return False
    
    if len(api_key) < 20:
        print("   ⚠️  API key seems too short, might be invalid")
        return False
    
    print(f"   ✅ API key found ({len(api_key)} chars)")
    
    # Test API connection
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        # Try a minimal API call
        client.models.list()
        print("   ✅ API key is valid and working")
        return True
    except ImportError:
        print("   ⚠️  OpenAI package not installed")
        print("   → Install with: pip install openai")
        return False
    except Exception as e:
        print(f"   ❌ API key test failed: {e}")
        return False


def check_dependencies():
    """Check if required Python packages are installed."""
    print("\n2. Checking dependencies...")
    
    required = {
        'openai': 'OpenAI API client',
        'pandas': 'Data manipulation',
        'matplotlib': 'Visualization',
        'seaborn': 'Statistical visualization',
        'tqdm': 'Progress bars',
        'numpy': 'Numerical computing'
    }
    
    all_ok = True
    for package, description in required.items():
        try:
            __import__(package)
            print(f"   ✅ {package} - {description}")
        except ImportError:
            print(f"   ❌ {package} - {description} (NOT INSTALLED)")
            all_ok = False
    
    if not all_ok:
        print("\n   → Install all dependencies with: pip install -r requirements.txt")
    
    return all_ok


def check_medcalc_bench():
    """Check if MedCalc-Bench is properly set up."""
    print("\n3. Checking MedCalc-Bench...")
    
    base_dir = Path(__file__).parent
    
    # Check evaluation module
    eval_file = base_dir / "MedCalc-Bench" / "evaluation" / "evaluate.py"
    if not eval_file.exists():
        print(f"   ❌ Evaluation module not found: {eval_file}")
        return False
    print("   ✅ Evaluation module found")
    
    # Check test data
    test_data = base_dir / "MedCalc-Bench" / "dataset" / "test_data.csv"
    if not test_data.exists():
        print(f"   ❌ Test data not found: {test_data}")
        return False
    
    # Count test examples
    import pandas as pd
    df = pd.read_csv(test_data)
    print(f"   ✅ Test data found ({len(df)} examples)")
    
    # Check one-shot examples
    oneshot_file = base_dir / "MedCalc-Bench" / "evaluation" / "one_shot_finalized_explanation.json"
    if not oneshot_file.exists():
        print(f"   ⚠️  One-shot examples not found: {oneshot_file}")
        print("   → Some features may not work")
    else:
        import json
        with open(oneshot_file, 'r') as f:
            oneshot = json.load(f)
        print(f"   ✅ One-shot examples found ({len(oneshot)} examples)")
    
    return True


def check_training_results(training_dir: str = None):
    """Check if training results directory exists and has required files."""
    print("\n4. Checking training results...")
    
    if training_dir is None:
        # Use default
        training_dir = "/Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434"
        print(f"   Using default: {training_dir}")
    
    training_path = Path(training_dir)
    
    if not training_path.exists():
        print(f"   ❌ Training results directory not found: {training_path}")
        print("   → Specify with: --training-results-dir <path>")
        return False
    
    print(f"   ✅ Training directory found")
    
    # Check for correct/incorrect subdirectories
    correct_dir = training_path / "correct"
    incorrect_dir = training_path / "incorrect"
    
    if not correct_dir.exists():
        print(f"   ❌ 'correct' subdirectory not found")
        return False
    
    if not incorrect_dir.exists():
        print(f"   ❌ 'incorrect' subdirectory not found")
        return False
    
    # Count examples
    correct_files = list(correct_dir.glob("*.jsonl"))
    incorrect_files = list(incorrect_dir.glob("*.jsonl"))
    
    if not correct_files:
        print(f"   ❌ No correct examples found")
        return False
    
    if not incorrect_files:
        print(f"   ❌ No incorrect examples found")
        return False
    
    # Count total examples
    import json
    correct_count = 0
    for file in correct_files:
        with open(file, 'r') as f:
            correct_count += sum(1 for _ in f)
    
    incorrect_count = 0
    for file in incorrect_files:
        with open(file, 'r') as f:
            incorrect_count += sum(1 for _ in f)
    
    print(f"   ✅ Correct examples: {correct_count}")
    print(f"   ✅ Incorrect examples: {incorrect_count}")
    print(f"   ✅ Total: {correct_count + incorrect_count}")
    
    # Check for prompts
    prompts_file = training_path / "prompts" / "enhanced_prompts.json"
    if not prompts_file.exists():
        print(f"   ⚠️  Enhanced prompts not found: {prompts_file}")
    else:
        with open(prompts_file, 'r') as f:
            prompts = json.load(f)
        print(f"   ✅ Enhanced prompts found ({len(prompts)} types)")
    
    return True


def check_scripts():
    """Check if all pipeline scripts are present."""
    print("\n5. Checking pipeline scripts...")
    
    base_dir = Path(__file__).parent
    
    required_scripts = [
        'prompt_refinement_pipeline.py',
        'create_validation_split.py',
        'grid_search_validation.py',
        'full_test_evaluation.py',
        'visualize_complete_results.py',
        'run_complete_optimization_pipeline.py'
    ]
    
    all_ok = True
    for script in required_scripts:
        script_path = base_dir / script
        if not script_path.exists():
            print(f"   ❌ {script} not found")
            all_ok = False
        else:
            print(f"   ✅ {script}")
    
    return all_ok


def check_disk_space():
    """Check available disk space."""
    print("\n6. Checking disk space...")
    
    import shutil
    
    try:
        base_dir = Path(__file__).parent
        stat = shutil.disk_usage(base_dir)
        
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)
        
        print(f"   Total: {total_gb:.1f} GB")
        print(f"   Free: {free_gb:.1f} GB")
        
        if free_gb < 1:
            print(f"   ⚠️  Low disk space (< 1 GB)")
            print(f"   → Pipeline outputs typically use ~500 MB")
            return False
        elif free_gb < 5:
            print(f"   ⚠️  Limited disk space")
        else:
            print(f"   ✅ Sufficient disk space")
        
        return True
    except Exception as e:
        print(f"   ⚠️  Could not check disk space: {e}")
        return True  # Don't fail on this


def print_summary(results: dict):
    """Print summary of checks."""
    print("\n" + "="*80)
    print("SETUP VERIFICATION SUMMARY")
    print("="*80)
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {check}")
    
    print("="*80)
    
    if all_passed:
        print("\n🎉 All checks passed! You're ready to run the pipeline.")
        print("\nQuick start:")
        print("  ./quick_start.sh")
        print("\nOr full command:")
        print("  python run_complete_optimization_pipeline.py \\")
        print("      --training-results-dir <path> \\")
        print("      --num-refinement-iterations 10 \\")
        print("      --num-candidates 5 \\")
        print("      --validation-size 100 \\")
        print("      --test-all-models")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above before running the pipeline.")
        return 1
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Verify setup for optimization pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--training-results-dir',
        type=str,
        default=None,
        help='Training results directory to verify (optional)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("OPTIMIZATION PIPELINE SETUP VERIFICATION")
    print("="*80)
    
    results = {
        'API Key': check_api_key(),
        'Dependencies': check_dependencies(),
        'MedCalc-Bench': check_medcalc_bench(),
        'Training Results': check_training_results(args.training_results_dir),
        'Pipeline Scripts': check_scripts(),
        'Disk Space': check_disk_space()
    }
    
    return print_summary(results)


if __name__ == "__main__":
    sys.exit(main())

