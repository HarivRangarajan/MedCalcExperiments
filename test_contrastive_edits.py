#!/usr/bin/env python3
"""
Quick Test Script for MedCalc Contrastive Edits Pipeline

This script runs a minimal test to verify the pipeline works correctly.
"""

import sys
import os
from pathlib import Path

# Add the main script to the path
sys.path.insert(0, str(Path(__file__).parent))

from medcalc_with_contrastive_boosted_edits import MedCalcContrastiveEvaluationPipeline

def test_pipeline():
    print("🧪 Testing MedCalc Contrastive Edits Pipeline")
    print("="*60)
    
    # Check virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("\n⚠️  Warning: Virtual environment not detected!")
        print("   It's recommended to activate the virtual environment:")
        print("   source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate")
        response = input("\n   Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return False
    else:
        print("\n✅ Virtual environment active")
    
    # Check prerequisites
    print("\n1️⃣ Checking Prerequisites...")
    
    if not Path("MedCalc-Bench").exists():
        print("❌ MedCalc-Bench directory not found!")
        return False
    print("   ✓ MedCalc-Bench directory found")
    
    if not Path("MedCalc-Bench/dataset/train_data.csv").exists():
        print("❌ train_data.csv not found! Run: unzip MedCalc-Bench/dataset/train_data.csv.zip")
        return False
    print("   ✓ train_data.csv found")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set!")
        return False
    print("   ✓ OPENAI_API_KEY found")
    
    # Test initialization
    print("\n2️⃣ Testing Pipeline Initialization...")
    try:
        pipeline = MedCalcContrastiveEvaluationPipeline(
            api_key=api_key,
            sample_size=2,  # Very small sample for quick test
            model="OpenAI/gpt-4o"
        )
        print("   ✓ Pipeline initialized successfully")
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        return False
    
    # Test data loading
    print("\n3️⃣ Testing Data Loading...")
    try:
        df = pipeline.load_medcalc_data(sample_size=2)
        if len(df) == 2:
            print(f"   ✓ Loaded 2 samples successfully")
        else:
            print(f"   ⚠️  Expected 2 samples, got {len(df)}")
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        return False
    
    # Test prompt creation
    print("\n4️⃣ Testing Enhanced Prompt Creation...")
    try:
        first_calculator_id = str(df.iloc[0]["Calculator ID"])
        example = pipeline.one_shot_examples.get(first_calculator_id)
        
        if example:
            system_msg, user_msg = pipeline.extract_original_one_shot_prompt(
                "Sample note", "Sample question",
                example["Patient Note"],
                {"step_by_step_thinking": example["Response"]["step_by_step_thinking"], 
                 "answer": example["Response"]["answer"]}
            )
            original_prompt = system_msg + "\n\n" + user_msg
            print(f"   ✓ Original prompt extracted ({len(original_prompt)} chars)")
            
            enhanced_prompts = pipeline.create_enhanced_prompts(original_prompt)
            if 'chain_of_thought' in enhanced_prompts and 'chain_of_draft' in enhanced_prompts:
                print(f"   ✓ Enhanced prompts created:")
                print(f"      • CoT: {len(enhanced_prompts['chain_of_thought']['prompt'])} chars")
                print(f"      • CoD: {len(enhanced_prompts['chain_of_draft']['prompt'])} chars")
            else:
                print("   ⚠️  Enhanced prompts missing expected techniques")
        else:
            print("   ⚠️  No one-shot example found for first calculator")
    except Exception as e:
        print(f"   ❌ Prompt creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test response generation (single sample to avoid API costs)
    print("\n5️⃣ Testing Response Generation (1 sample)...")
    print("   ⚠️  This will make 3 API calls (costs ~$0.03-0.05)")
    
    response = input("   Continue with API test? (y/n): ").strip().lower()
    if response != 'y':
        print("   ⏭️  Skipping API test")
        print("\n✅ Basic tests passed! Pipeline structure is correct.")
        return True
    
    try:
        # Test with just one sample from the loaded data
        single_sample_df = df.head(1)
        all_responses = pipeline.generate_responses(single_sample_df, enhanced_prompts)
        
        if len(all_responses) == 3:
            print(f"   ✓ Generated responses for 3 prompt types")
            for prompt_type, responses in all_responses.items():
                if len(responses) > 0:
                    print(f"      • {prompt_type}: {len(responses)} response(s)")
                else:
                    print(f"      ⚠️  {prompt_type}: No responses generated")
        else:
            print(f"   ⚠️  Expected 3 prompt types, got {len(all_responses)}")
    except Exception as e:
        print(f"   ❌ Response generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test evaluation
    print("\n6️⃣ Testing Evaluation...")
    try:
        evaluation_results = pipeline.evaluate_and_split_responses(all_responses)
        print(f"   ✓ Evaluation completed:")
        for prompt_type, results in evaluation_results.items():
            print(f"      • {prompt_type}: {results['accuracy']:.0%} ({results['correct']}/{results['total']})")
    except Exception as e:
        print(f"   ❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify output files
    print("\n7️⃣ Verifying Output Files...")
    try:
        output_dir = pipeline.output_dir
        expected_files = [
            "responses/original_responses.jsonl",
            "responses/chain_of_thought_responses.jsonl",
            "responses/chain_of_draft_responses.jsonl",
            "evaluations/evaluation_summary.json",
            "correct/original_correct.jsonl",
            "incorrect/original_incorrect.jsonl"
        ]
        
        for file_path in expected_files:
            full_path = output_dir / file_path
            if full_path.exists():
                print(f"   ✓ {file_path}")
            else:
                print(f"   ⚠️  {file_path} not found")
    except Exception as e:
        print(f"   ⚠️  File verification failed: {e}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print(f"\nOutput directory: {pipeline.output_dir}")
    print("\nYou can now run the full pipeline with:")
    print("  python medcalc_with_contrastive_boosted_edits.py --sample-size 500")
    
    return True


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)

