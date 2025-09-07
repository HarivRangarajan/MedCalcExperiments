#!/usr/bin/env python3
"""
Quick setup test for MedCalc evaluation system
"""

import sys
import os
from pathlib import Path

def test_setup():
    print("🔧 Testing MedCalc Evaluation Setup")
    print("="*50)
    
    # Test 1: Check MedCalc-Bench data
    print("1. Checking MedCalc-Bench data...")
    test_data_path = Path("MedCalc-Bench/dataset/test_data.csv")
    if test_data_path.exists():
        print("   ✅ test_data.csv found")
        
        # Quick data check
        try:
            import pandas as pd
            df = pd.read_csv(test_data_path)
            print(f"   ✅ Data loaded: {len(df)} examples")
            print(f"   ✅ Categories: {list(df['Category'].unique())}")
        except Exception as e:
            print(f"   ⚠️  Data loading issue: {e}")
    else:
        print("   ❌ test_data.csv not found")
        return False
    
    # Test 2: Check dependencies
    print("\n2. Checking dependencies...")
    required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'openai']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   To install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
    
    # Test 3: Check API key access
    print("\n3. Checking API key access...")
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print("   ✅ API key found in environment")
    else:
        print("   ⚠️  API key not found in environment")
    
    # Test 4: Check shared components
    print("\n4. Checking shared components...")
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "promptengineer"))
        from promptengineer import PromptPipeline
        print("   ✅ PromptEngineer library accessible")
    except ImportError as e:
        print(f"   ⚠️  PromptEngineer import issue: {e}")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "mohs-llm-as-a-judge"))
        from modules.llm_judge import LLMJudge
        print("   ✅ LLM Judge accessible")
    except ImportError as e:
        print(f"   ⚠️  LLM Judge import issue: {e}")
    
    # Test 5: Quick data sample
    print("\n5. Quick data sampling test...")
    try:
        import pandas as pd
        df = pd.read_csv(test_data_path)
        sample = df.sample(n=5, random_state=42)
        
        print(f"   ✅ Successfully sampled 5 examples")
        print(f"   📊 Sample categories: {list(sample['Category'].unique())}")
        print(f"   📊 Sample calculators: {list(sample['Calculator Name'].unique())}")
        
    except Exception as e:
        print(f"   ❌ Sampling failed: {e}")
        return False
    
    print("\n🎉 Setup verification complete!")
    
    if missing_packages:
        print(f"\n⚠️  Install missing packages before running evaluation:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✅ System ready for evaluation!")
    print("\nNext steps:")
    print("1. Run: python run_medcalc_evaluation.py")
    print("2. Or for custom options: python medcalc_prompt_evaluation_pipeline.py --help")
    
    return True

if __name__ == "__main__":
    success = test_setup()
    if not success:
        sys.exit(1) 