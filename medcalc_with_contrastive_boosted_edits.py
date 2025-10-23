#!/usr/bin/env python3
"""
MedCalc-Bench Contrastive Boosted Edits Evaluation Pipeline

Usage:
  python medcalc_with_contrastive_boosted_edits.py --sample-size 500 --output-dir results_experiment1
"""

import pandas as pd
import numpy as np
import json
import sys
import os
import argparse
import random
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path
import re
import warnings
import openai
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "promptengineer"))

# Import shared components
from promptengineer import PromptPipeline
from promptengineer.techniques.base import PromptContext

# MedCalc evaluation imports
sys.path.insert(0, str(Path(__file__).parent / "MedCalc-Bench" / "evaluation"))

# Try importing with fallback for OpenAI-only usage
try:
    # Suppress torch import warnings for OpenAI-only usage
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        from evaluate import check_correctness
    
    # Custom LLMInference for OpenAI only (avoiding torch dependency)
    LLMInference = None
    try:
        from llm_inference import LLMInference
    except ImportError:
        print("ℹ️  Using OpenAI-only inference (torch not available)")
        
except ImportError as e:
    print(f"⚠️  MedCalc evaluation imports failed: {e}")
    sys.exit(1)


class MedCalcContrastiveEvaluationPipeline:
    """Complete evaluation pipeline for MedCalc-Bench with prompt engineering comparison."""
    
    # Configuration constants
    DEFAULT_SAMPLE_SIZE = 500  # Default number of examples to sample from dataset
    DEFAULT_MODEL = "OpenAI/gpt-4o"  # Default model for evaluations
    
    def __init__(self, api_key: str, output_dir: str = None, 
                 sample_size: int = None,
                 model: str = None):
        """Initialize the evaluation pipeline."""
        self.api_key = api_key
        openai.api_key = api_key
        
        # Set configuration parameters with defaults
        self.sample_size = sample_size if sample_size is not None else self.DEFAULT_SAMPLE_SIZE
        self.model = model if model is not None else self.DEFAULT_MODEL
        
        if output_dir is None:
            # Create output directory in centralized outputs folder (relative to project root)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            project_root = Path(__file__).parent.parent
            output_dir = project_root / "outputs" / f"medcalc_contrastive_edits_evaluation_{timestamp}"
        self.output_dir = Path(output_dir)
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ["data", "prompts", "responses", "evaluations", "correct", "incorrect"]:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        # Initialize components
        self.prompt_pipeline = PromptPipeline(api_key=api_key, output_dir=str(self.output_dir))
        
        # Initialize LLM Inference (use custom OpenAI wrapper if LLMInference not available)
        if LLMInference is not None:
            self.llm = LLMInference(llm_name=self.model)
        else:
            # Use custom OpenAI-only wrapper
            self.llm = self._create_openai_wrapper()
        
        # Load MedCalc one-shot examples
        self.one_shot_examples = self._load_medcalc_one_shot_examples()
        
        print(f"✅ Pipeline initialized with output directory: {self.output_dir}")
    
    def _create_openai_wrapper(self):
        """Create a simple OpenAI wrapper that mimics LLMInference interface."""
        class SimpleOpenAIWrapper:
            def __init__(self, model_name, api_key):
                self.model = model_name.split('/')[-1] if '/' in model_name else model_name
                # Use new OpenAI client (v1.0+)
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
            
            def answer(self, messages):
                """Generate response using OpenAI API."""
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                ans = response.choices[0].message.content
                ans = re.sub(r"\s+", " ", ans)
                return ans
        
        return SimpleOpenAIWrapper(self.model, self.api_key)
    
    def _load_medcalc_one_shot_examples(self) -> Dict[str, Any]:
        """Load MedCalc's original one-shot examples for calculator-specific prompting."""
        try:
            one_shot_file = Path(__file__).parent / "MedCalc-Bench" / "evaluation" / "one_shot_finalized_explanation.json"
            if one_shot_file.exists():
                with open(one_shot_file, 'r') as f:
                    examples = json.load(f)
                print(f"✅ Loaded {len(examples)} calculator-specific one-shot examples")
                return examples
            else:
                print(f"⚠️  One-shot examples file not found")
                return {}
        except Exception as e:
            print(f"⚠️  Error loading one-shot examples: {e}")
            return {}
    
    def load_medcalc_data(self, sample_size: int = None) -> pd.DataFrame:
        """Load and sample MedCalc-Bench train data."""
        if sample_size is None:
            sample_size = self.sample_size
        
        print(f"\n📋 STEP 1: Loading MedCalc-Bench Data (Sample: {sample_size})")
        print("="*60)
        
        # Load train data
        train_data_path = Path(__file__).parent / "MedCalc-Bench" / "dataset" / "train_data.csv"
        df = pd.read_csv(train_data_path)
        
        print(f"✅ Loaded {len(df)} total examples from MedCalc-Bench train data")
        
        # Basic statistics
        print(f"   • Categories: {df['Category'].unique()}")
        print(f"   • Calculator types: {len(df['Calculator Name'].unique())} unique calculators")
        print(f"   • Output types: {df['Output Type'].unique()}")
        
        # Random sampling
        if sample_size < len(df):
            sampled_df = df.sample(n=sample_size, random_state=42)
            print(f"   • Randomly sampled {sample_size} examples")
        else:
            sampled_df = df
            print(f"   • Using all {len(df)} examples (requested sample size >= total)")
        
        # Save sampled data
        sample_file = self.output_dir / "data" / "sampled_medcalc_data.csv"
        sampled_df.to_csv(sample_file, index=False)
        
        # Category distribution in sample
        category_dist = sampled_df['Category'].value_counts()
        print(f"\n   Sample distribution by category:")
        for cat, count in category_dist.items():
            print(f"      • {cat}: {count} ({count/len(sampled_df)*100:.1f}%)")
        
        return sampled_df
    
    def extract_original_one_shot_prompt(self, note: str, question: str, example_note: str, example_output: Dict) -> Tuple[str, str]:
        """
        Extract the exact one-shot prompt from MedCalc's run.py.
        Returns (system_msg, user_msg) tuple.
        """
        system_msg = 'You are a helpful assistant for calculating a score for a given patient note. Please think step-by-step to solve the question and then generate the required score. Your output should only contain a JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}}.'
        system_msg += f'Here is an example patient note:\n\n{example_note}'
        system_msg += f'\n\nHere is an example task:\n\n{question}'
        system_msg += f'\n\nPlease directly output the JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(value which is the answer to the question)}}:\n\n{json.dumps(example_output)}'
        user_temp = f'Here is the patient note:\n\n{note}\n\nHere is the task:\n\n{question}\n\nPlease directly output the JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}}:'
        return system_msg, user_temp
    
    def create_enhanced_prompts(self, original_one_shot_prompt: str) -> Dict[str, str]:
        """
        Create enhanced versions of the one-shot prompt using PromptEngineer.
        Returns dict with 'chain_of_thought' and 'chain_of_draft' enhanced prompts.
        """
        print(f"\n🚀 STEP 2: Creating Enhanced Prompts")
        print("="*60)
        
        # Create a PromptContext from the original one-shot prompt
        context = PromptContext(
            task_description=original_one_shot_prompt,
            domain="Medical calculation and clinical reasoning"
        )
        
        # Generate enhanced prompts using specified techniques
        techniques = ["chain_of_thought", "chain_of_draft"]
        enhanced_prompts = self.prompt_pipeline.generate_enhanced_prompts(context, techniques)
        
        print(f"✅ Enhanced prompts generated:")
        for technique, prompt_data in enhanced_prompts.items():
            print(f"   • {technique}: {len(prompt_data['prompt']):,} characters")
        
        # Save enhanced prompts
        prompts_file = self.output_dir / "prompts" / "enhanced_prompts.json"
        with open(prompts_file, 'w') as f:
            json.dump(enhanced_prompts, f, indent=2)
        
        return enhanced_prompts
    
    def extract_answer(self, answer: str, calid: int):
        """Extract answer and explanation from LLM response."""
        extracted_answer = re.findall(r'[Aa]nswer":\s*(.*?)\}', answer)
        matches = re.findall(r'"step_by_step_thinking":\s*"([^"]+)"\s*,\s*"[Aa]nswer"', answer)
        
        if matches:
            explanation = matches[-1]    
        else:
            explanation = "No Explanation"
        
        if len(extracted_answer) == 0:
            extracted_answer = "Not Found"
        else:
            extracted_answer = extracted_answer[-1].strip().strip('"')
            if extracted_answer == "str(short_and_direct_answer_of_the_question)" or extracted_answer == "str(value which is the answer to the question)" or extracted_answer == "X.XX":
                extracted_answer = "Not Found"
        
        # Handle different calculator output types (simplified version)
        if calid in [13, 68]:
            # Date type
            match = re.search(r"^(0?[1-9]|1[0-2])\/(0?[1-9]|[12][0-9]|3[01])\/(\d{4})", extracted_answer)
            if match:
                month = int(match.group(1))
                day = int(match.group(2))
                year = match.group(3)
                answer = f"{month:02}/{day:02}/{year}"
            else:
                answer = "N/A"
        elif calid in [69]:
            # Tuple type
            match = re.search(r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?", extracted_answer)
            if match:
                weeks = match.group(1)
                days = match.group(3)
                answer = f"({weeks}, {days})"
            else:
                answer = "N/A"
        elif calid in [4, 15, 16, 17, 18, 20, 21, 25, 27, 28, 29, 32, 33, 36, 43, 45, 48, 51, 69]:
            # Integer type
            match = re.search(r"(\d+) out of", extracted_answer)
            if match:
                answer = match.group(1)
            else:
                match = re.findall(r"(-?\d+(\.\d+)?)", extracted_answer)
                if len(match) > 0:
                    answer = match[-1][0]
                else:
                    answer = "N/A"
        else:
            # Decimal type
            match = re.findall(r"(-?\d+(\.\d+)?)", extracted_answer)
            if len(match) > 0:
                answer = eval(match[-1][0])
                answer = str(answer)
            else:
                answer = "N/A"
        
        return answer, explanation
    
    def generate_responses(self, df: pd.DataFrame, enhanced_prompts: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """
        Generate responses for all prompt types (original + enhanced).
        """
        print(f"\n🔧 STEP 3: Generating Responses")
        print("="*60)
        
        all_responses = {
            "original": [],
            "chain_of_thought": [],
            "chain_of_draft": []
        }
        
        # Generate responses for each prompt type
        for prompt_type in all_responses.keys():
            print(f"\n   Generating responses for: {prompt_type}")
            
            for index in tqdm(range(len(df)), desc=f"Processing {prompt_type}"):
                row = df.iloc[index]
                
                patient_note = row["Patient Note"]
                question = row["Question"]
                calculator_id = str(row["Calculator ID"])
                note_id = str(row["Note ID"])
                
                # Get one-shot example for this calculator
                example = self.one_shot_examples.get(calculator_id)
                if example is None:
                    print(f"   ⚠️  No one-shot example for calculator {calculator_id}, skipping...")
                    continue
                
                try:
                    # Create messages based on prompt type
                    if prompt_type == "original":
                        # Use original one-shot prompt
                        system_msg, user_msg = self.extract_original_one_shot_prompt(
                            patient_note, question, example["Patient Note"],
                            {"step_by_step_thinking": example["Response"]["step_by_step_thinking"], 
                             "answer": example["Response"]["answer"]}
                        )
                    else:
                        # Use enhanced prompt - combine with one-shot example
                        enhanced_prompt = enhanced_prompts[prompt_type]["prompt"]
                        # Inject the one-shot example into the enhanced prompt
                        system_msg = enhanced_prompt + f'\n\nHere is an example patient note:\n\n{example["Patient Note"]}'
                        system_msg += f'\n\nHere is an example task:\n\n{question}'
                        system_msg += f'\n\nHere is the expected output:\n\n{json.dumps({"step_by_step_thinking": example["Response"]["step_by_step_thinking"], "answer": example["Response"]["answer"]})}'
                        user_msg = f'Here is the patient note:\n\n{patient_note}\n\nHere is the task:\n\n{question}\n\nPlease directly output the JSON dict.'
                    
                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ]
                    
                    # Generate answer
                    answer = self.llm.answer(messages)
                    
                    # Extract answer and explanation
                    answer_value, explanation = self.extract_answer(answer, int(calculator_id))
                    
                    # Check correctness
                    correctness = check_correctness(
                        answer_value, 
                        row["Ground Truth Answer"], 
                        calculator_id, 
                        row["Upper Limit"], 
                        row["Lower Limit"]
                    )
                    
                    status = "Correct" if correctness else "Incorrect"
                    
                    # Store result
                    result = {
                        "Row Number": int(row["Row Number"]),
                        "Calculator Name": row["Calculator Name"],
                        "Calculator ID": calculator_id,
                        "Category": row["Category"],
                        "Note ID": note_id,
                        "Patient Note": patient_note,
                        "Question": question,
                        "LLM Answer": answer_value,
                        "LLM Explanation": explanation,
                        "Ground Truth Answer": row["Ground Truth Answer"],
                        "Ground Truth Explanation": row["Ground Truth Explanation"],
                        "Result": status,
                        "Prompt Type": prompt_type
                    }
                    
                    all_responses[prompt_type].append(result)
                    
                except Exception as e:
                    print(f"   ⚠️  Error processing row {index} with {prompt_type}: {e}")
                    # Store error result
                    result = {
                        "Row Number": int(row["Row Number"]),
                        "Calculator Name": row["Calculator Name"],
                        "Calculator ID": calculator_id,
                        "Category": row["Category"],
                        "Note ID": note_id,
                        "Patient Note": patient_note,
                        "Question": question,
                        "LLM Answer": str(e),
                        "LLM Explanation": str(e),
                        "Ground Truth Answer": row["Ground Truth Answer"],
                        "Ground Truth Explanation": row["Ground Truth Explanation"],
                        "Result": "Incorrect",
                        "Prompt Type": prompt_type
                    }
                    all_responses[prompt_type].append(result)
            
            # Save responses for this prompt type
            responses_file = self.output_dir / "responses" / f"{prompt_type}_responses.jsonl"
            with open(responses_file, 'w') as f:
                for result in all_responses[prompt_type]:
                    f.write(json.dumps(result) + "\n")
        
        return all_responses
    
    def evaluate_and_split_responses(self, all_responses: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Evaluate accuracy for each prompt type and split into correct/incorrect files.
        """
        print(f"\n📊 STEP 4: Evaluating and Splitting Responses")
        print("="*60)
        
        evaluation_results = {}
        
        for prompt_type, responses in all_responses.items():
            print(f"\n   Evaluating: {prompt_type}")
            
            # Calculate accuracy
            total = len(responses)
            correct_count = sum(1 for r in responses if r["Result"] == "Correct")
            accuracy = correct_count / total if total > 0 else 0
            
            print(f"      Accuracy: {accuracy:.2%} ({correct_count}/{total})")
            
            # Split into correct and incorrect
            correct_responses = [r for r in responses if r["Result"] == "Correct"]
            incorrect_responses = [r for r in responses if r["Result"] == "Incorrect"]
            
            # Save correct responses
            correct_file = self.output_dir / "correct" / f"{prompt_type}_correct.jsonl"
            with open(correct_file, 'w') as f:
                for result in correct_responses:
                    f.write(json.dumps(result) + "\n")
            
            # Save incorrect responses
            incorrect_file = self.output_dir / "incorrect" / f"{prompt_type}_incorrect.jsonl"
            with open(incorrect_file, 'w') as f:
                for result in incorrect_responses:
                    f.write(json.dumps(result) + "\n")
            
            print(f"      Saved {len(correct_responses)} correct responses to: {correct_file.name}")
            print(f"      Saved {len(incorrect_responses)} incorrect responses to: {incorrect_file.name}")
            
            evaluation_results[prompt_type] = {
                "accuracy": accuracy,
                "total": total,
                "correct": correct_count,
                "incorrect": len(incorrect_responses)
            }
        
        # Save evaluation summary
        summary_file = self.output_dir / "evaluations" / "evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        return evaluation_results
    
    def run_complete_evaluation(self) -> Dict[str, Any]:
        """Run the complete evaluation pipeline."""
        print("="*100)
        print("MEDCALC-BENCH WITH CONTRASTIVE BOOSTED EDITS EVALUATION PIPELINE")
        print("="*100)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Sample size: {self.sample_size}")
        print(f"Model: {self.model}")
        
        # Step 1: Load data
        df = self.load_medcalc_data()
        
        # Step 2: Create enhanced prompts based on original one-shot
        # Get a sample one-shot prompt to create the base context
        first_calculator_id = str(df.iloc[0]["Calculator ID"])
        example = self.one_shot_examples.get(first_calculator_id)
        
        if example:
            system_msg, user_msg = self.extract_original_one_shot_prompt(
                "Sample patient note", "Sample question",
                example["Patient Note"],
                {"step_by_step_thinking": example["Response"]["step_by_step_thinking"], 
                 "answer": example["Response"]["answer"]}
            )
            original_prompt = system_msg + "\n\n" + user_msg
        else:
            original_prompt = "You are a helpful assistant for calculating a score for a given patient note."
        
        enhanced_prompts = self.create_enhanced_prompts(original_prompt)
        
        # Step 3: Generate responses
        all_responses = self.generate_responses(df, enhanced_prompts)
        
        # Step 4: Evaluate and split responses
        evaluation_results = self.evaluate_and_split_responses(all_responses)
        
        # Final summary
        print("\n" + "="*100)
        print("EVALUATION COMPLETE")
        print("="*100)
        
        print(f"\nKey Results:")
        for prompt_type, results in evaluation_results.items():
            print(f"   • {prompt_type}: {results['accuracy']:.2%} accuracy ({results['correct']}/{results['total']})")
        
        print(f"\nAll results saved to: {self.output_dir}/")
        
        return {
            "data": df,
            "enhanced_prompts": enhanced_prompts,
            "all_responses": all_responses,
            "evaluation_results": evaluation_results,
            "output_directory": self.output_dir
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MedCalc-Bench Contrastive Boosted Edits Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python medcalc_with_contrastive_boosted_edits.py --sample-size 500
  python medcalc_with_contrastive_boosted_edits.py --sample-size 100 --output-dir ./my_results
        """
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=500,
        help='Number of MedCalc train examples to evaluate (default: 500)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: auto-generated with timestamp)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='OpenAI/gpt-4o',
        help='Model to use for inference (default: OpenAI/gpt-4o)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = MedCalcContrastiveEvaluationPipeline(
        api_key=api_key,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        model=args.model
    )
    
    # Run evaluation
    results = pipeline.run_complete_evaluation()
    
    print("\n✅ Pipeline execution completed successfully!")
    print(f"Results saved to: {results['output_directory']}")


if __name__ == "__main__":
    main()