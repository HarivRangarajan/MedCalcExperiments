#!/usr/bin/env python3
"""
Contrastive Few-Shot Evaluation Pipeline

This module evaluates prompts using contrastive few-shot examples on the full
MedCalc test set. It compares:
1. Original one-shot prompt
2. Unified refined prompt with contrastive few-shot examples

Usage:
    python contrastive_few_shot_evaluation.py \
        --refined-prompts-dir <path> \
        --training-results-dir <path>
"""

import pandas as pd
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import argparse
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
import re
import random
import asyncio
import warnings
warnings.filterwarnings('ignore')

# Add MedCalc evaluation imports
sys.path.insert(0, str(Path(__file__).parent / "MedCalc-Bench" / "evaluation"))
try:
    from evaluate import check_correctness
except ImportError as e:
    print(f"⚠️  MedCalc evaluation imports failed: {e}")
    sys.exit(1)


class ContrastiveFewShotEvaluator:
    """Evaluator using contrastive few-shot examples."""
    
    def __init__(self,
                 api_key: str,
                 refined_prompts_dir: str,
                 training_results_dir: str,
                 output_dir: str = None,
                 num_test_examples: int = None,
                 num_positive: int = 1,
                 num_negative: int = 1,
                 batch_size: int = 10,
                 save_frequency: int = 50):
        """
        Initialize the evaluator.
        
        Args:
            api_key: OpenAI API key
            refined_prompts_dir: Directory containing refined prompts
            training_results_dir: Directory with training results (for contrastive examples)
            output_dir: Output directory for evaluation results
            num_test_examples: Number of test examples to evaluate (None = all)
            num_positive: Number of positive contrastive examples (default: 1)
            num_negative: Number of negative contrastive examples (default: 1)
            batch_size: Batch size for OpenAI API calls (default: 10)
            save_frequency: Save progress every N examples (default: 50)
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.refined_prompts_dir = Path(refined_prompts_dir)
        self.training_results_dir = Path(training_results_dir)
        self.num_test_examples = num_test_examples
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.batch_size = batch_size
        self.save_frequency = save_frequency
        
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            suffix = f"_test{num_test_examples}" if num_test_examples else ""
            output_dir = Path(__file__).parent.parent / "outputs" / f"contrastive_evaluation_{timestamp}{suffix}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in ["responses", "evaluations", "logs", "visualizations"]:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        # Load one-shot examples
        self.one_shot_examples = self._load_one_shot_examples()
        
        # Load contrastive examples
        self.contrastive_examples = self._load_contrastive_examples()
        
        print(f"✅ Contrastive evaluator initialized")
        print(f"   • Refined prompts: {self.refined_prompts_dir}")
        print(f"   • Training results: {self.training_results_dir}")
        print(f"   • Output dir: {self.output_dir}")
    
    def _load_one_shot_examples(self) -> Dict[str, Any]:
        """Load MedCalc's original one-shot examples."""
        try:
            one_shot_file = Path(__file__).parent / "MedCalc-Bench" / "evaluation" / "one_shot_finalized_explanation.json"
            if one_shot_file.exists():
                with open(one_shot_file, 'r') as f:
                    examples = json.load(f)
                print(f"   ✓ Loaded {len(examples)} one-shot examples")
                return examples
            else:
                print(f"   ⚠️  One-shot examples file not found")
                return {}
        except Exception as e:
            print(f"   ⚠️  Error loading one-shot examples: {e}")
            return {}
    
    def _load_contrastive_examples(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Load contrastive examples (correct and incorrect) organized by calculator ID."""
        contrastive = {}
        
        # Load correct examples
        correct_files = list((self.training_results_dir / "correct").glob("*.jsonl"))
        for file in correct_files:
            with open(file, 'r') as f:
                for line in f:
                    example = json.loads(line)
                    calc_id = example['Calculator ID']
                    
                    if calc_id not in contrastive:
                        contrastive[calc_id] = {'correct': [], 'incorrect': []}
                    
                    contrastive[calc_id]['correct'].append(example)
        
        # Load incorrect examples
        incorrect_files = list((self.training_results_dir / "incorrect").glob("*.jsonl"))
        for file in incorrect_files:
            with open(file, 'r') as f:
                for line in f:
                    example = json.loads(line)
                    calc_id = example['Calculator ID']
                    
                    if calc_id not in contrastive:
                        contrastive[calc_id] = {'correct': [], 'incorrect': []}
                    
                    contrastive[calc_id]['incorrect'].append(example)
        
        num_calcs_with_correct = sum(1 for v in contrastive.values() if v['correct'])
        num_calcs_with_incorrect = sum(1 for v in contrastive.values() if v['incorrect'])
        
        print(f"   ✓ Loaded contrastive examples:")
        print(f"      - {num_calcs_with_correct} calculators with correct examples")
        print(f"      - {num_calcs_with_incorrect} calculators with incorrect examples")
        
        return contrastive
    
    def load_unified_prompt(self) -> str:
        """Load the unified refined prompt."""
        unified_file = self.refined_prompts_dir / "final" / "unified_prompt.txt"
        
        if not unified_file.exists():
            raise FileNotFoundError(f"Unified prompt not found at {unified_file}")
        
        with open(unified_file, 'r') as f:
            prompt = f.read()
        
        print(f"   ✓ Loaded unified prompt ({len(prompt)} characters)")
        return prompt
    
    def load_test_data(self) -> pd.DataFrame:
        """Load MedCalc test data."""
        test_data_path = Path(__file__).parent / "MedCalc-Bench" / "dataset" / "test_data.csv"
        df = pd.read_csv(test_data_path)
        
        if self.num_test_examples is not None:
            df = df.head(self.num_test_examples)
            print(f"   ✓ Loaded {len(df)} test examples (limited to first {self.num_test_examples})")
        else:
            print(f"   ✓ Loaded {len(df)} test examples")
        
        return df
    
    def get_contrastive_examples(self, 
                                 calculator_id: str,
                                 num_positive: int = None,
                                 num_negative: int = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Get contrastive examples for a calculator ID.
        
        Returns:
            Tuple of (positive_examples, negative_examples)
            Returns empty lists if calculator_id not found
        """
        # Use instance defaults if not specified
        if num_positive is None:
            num_positive = self.num_positive
        if num_negative is None:
            num_negative = self.num_negative
            
        if calculator_id not in self.contrastive_examples:
            return [], []
        
        calc_examples = self.contrastive_examples[calculator_id]
        
        # Sample positive examples
        available_positive = calc_examples['correct']
        if len(available_positive) >= num_positive:
            positive = random.sample(available_positive, num_positive)
        else:
            positive = available_positive
        
        # Sample negative examples
        available_negative = calc_examples['incorrect']
        if len(available_negative) >= num_negative:
            negative = random.sample(available_negative, num_negative)
        else:
            negative = available_negative
        
        return positive, negative
    
    def create_original_one_shot_prompt(self, 
                                       note: str, 
                                       question: str, 
                                       calculator_id: str) -> Tuple[str, str]:
        """Create the original MedCalc one-shot prompt."""
        example = self.one_shot_examples.get(calculator_id)
        
        if example is None:
            # Fallback to zero-shot if no example available
            system_msg = 'You are a helpful assistant for calculating a score for a given patient note. Please think step-by-step to solve the question and then generate the required score. Your output should only contain a JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}}.'
            user_msg = f'Here is the patient note:\n\n{note}\n\nHere is the task:\n\n{question}\n\nPlease directly output the JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}}:'
        else:
            system_msg = 'You are a helpful assistant for calculating a score for a given patient note. Please think step-by-step to solve the question and then generate the required score. Your output should only contain a JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}}.'
            system_msg += f'Here is an example patient note:\n\n{example["Patient Note"]}'
            system_msg += f'\n\nHere is an example task:\n\n{question}'
            system_msg += f'\n\nPlease directly output the JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(value which is the answer to the question)}}:\n\n{json.dumps({"step_by_step_thinking": example["Response"]["step_by_step_thinking"], "answer": example["Response"]["answer"]})}'
            user_msg = f'Here is the patient note:\n\n{note}\n\nHere is the task:\n\n{question}\n\nPlease directly output the JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}}:'
        
        return system_msg, user_msg
    
    def create_contrastive_few_shot_prompt(self,
                                          note: str,
                                          question: str,
                                          calculator_id: str,
                                          unified_prompt: str) -> Tuple[str, str]:
        """Create a contrastive few-shot prompt with positive and negative examples."""
        
        # Get one-shot example
        one_shot_example = self.one_shot_examples.get(calculator_id)
        
        # Get contrastive examples
        positive_examples, negative_examples = self.get_contrastive_examples(calculator_id)
        
        # Build system message with unified prompt
        system_msg = unified_prompt
        
        # Add one-shot example if available
        if one_shot_example:
            system_msg += f'\n\n**Example (Correct Approach):**\n'
            system_msg += f'Patient Note: {one_shot_example["Patient Note"][:500]}...\n'
            system_msg += f'Task: {question}\n'
            system_msg += f'Response: {json.dumps({"step_by_step_thinking": one_shot_example["Response"]["step_by_step_thinking"], "answer": one_shot_example["Response"]["answer"]})}'
        
        # Add positive demonstrations
        if positive_examples:
            system_msg += f'\n\n**Additional Correct Examples:**\n'
            for i, ex in enumerate(positive_examples, 1):
                system_msg += f'\nExample {i} (CORRECT):\n'
                system_msg += f'Patient Note: {ex["Patient Note"][:300]}...\n'
                system_msg += f'Task: {ex["Question"][:200]}...\n'
                system_msg += f'LLM Answer: {ex["LLM Answer"]}\n'
                system_msg += f'Ground Truth: {ex["Ground Truth Answer"]}\n'
        
        # Add negative demonstrations (what NOT to do)
        if negative_examples:
            system_msg += f'\n\n**Examples to AVOID (Common Mistakes):**\n'
            for i, ex in enumerate(negative_examples, 1):
                system_msg += f'\nExample {i} (INCORRECT - Learn from this mistake):\n'
                system_msg += f'Patient Note: {ex["Patient Note"][:300]}...\n'
                system_msg += f'Task: {ex["Question"][:200]}...\n'
                system_msg += f'Incorrect LLM Answer: {ex["LLM Answer"]}\n'
                system_msg += f'Correct Answer Should Be: {ex["Ground Truth Answer"]}\n'
        
        # User message with current task
        user_msg = f'Here is the patient note:\n\n{note}\n\nHere is the task:\n\n{question}\n\nPlease directly output the JSON dict with your step-by-step thinking and final answer.'
        
        return system_msg, user_msg
    
    def extract_answer(self, answer: str, calid: int) -> Tuple[str, str]:
        """Extract answer and explanation from LLM response (same as original)."""
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
            if extracted_answer in ["str(short_and_direct_answer_of_the_question)", 
                                   "str(value which is the answer to the question)", "X.XX"]:
                extracted_answer = "Not Found"
        
        # Handle different calculator output types (simplified)
        if calid in [13, 68]:
            match = re.search(r"^(0?[1-9]|1[0-2])\/(0?[1-9]|[12][0-9]|3[01])\/(\d{4})", extracted_answer)
            if match:
                month = int(match.group(1))
                day = int(match.group(2))
                year = match.group(3)
                answer = f"{month:02}/{day:02}/{year}"
            else:
                answer = "N/A"
        elif calid in [69]:
            match = re.search(r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?", extracted_answer)
            if match:
                weeks = match.group(1)
                days = match.group(3)
                answer = f"({weeks}, {days})"
            else:
                answer = "N/A"
        elif calid in [4, 15, 16, 17, 18, 20, 21, 25, 27, 28, 29, 32, 33, 36, 43, 45, 48, 51, 69]:
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
            match = re.findall(r"(-?\d+(\.\d+)?)", extracted_answer)
            if len(match) > 0:
                answer = eval(match[-1][0])
                answer = str(answer)
            else:
                answer = "N/A"
        
        return answer, explanation
    
    async def _process_single_example_async(self,
                                            row: pd.Series,
                                            prompt_type: str,
                                            unified_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Process a single example asynchronously."""
        patient_note = row["Patient Note"]
        question = row["Question"]
        calculator_id = str(row["Calculator ID"])
        note_id = str(row["Note ID"])
        
        try:
            # Create prompt based on type
            if prompt_type == "original_one_shot":
                system_msg, user_msg = self.create_original_one_shot_prompt(
                    patient_note, question, calculator_id
                )
            elif prompt_type == "contrastive_few_shot":
                system_msg, user_msg = self.create_contrastive_few_shot_prompt(
                    patient_note, question, calculator_id, unified_prompt
                )
            else:
                raise ValueError(f"Unknown prompt type: {prompt_type}")
            
            # Generate response asynchronously
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            
            response = await self.async_client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            
            answer = response.choices[0].message.content
            answer = re.sub(r"\s+", " ", answer)
            
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
            
            # Return result
            return {
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
                "Prompt Type": prompt_type,
                "Error": None
            }
            
        except Exception as e:
            return {
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
                "Prompt Type": prompt_type,
                "Error": str(e)
            }
    
    async def _process_batch_async(self, 
                                   batch_df: pd.DataFrame, 
                                   prompt_type: str,
                                   unified_prompt: Optional[str] = None) -> List[Dict]:
        """Process a batch of examples concurrently using asyncio.gather."""
        tasks = []
        for idx in range(len(batch_df)):
            row = batch_df.iloc[idx]
            tasks.append(self._process_single_example_async(row, prompt_type, unified_prompt))
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        return list(results)
    
    def generate_and_evaluate(self,
                            df: pd.DataFrame,
                            prompt_type: str,
                            unified_prompt: Optional[str] = None) -> List[Dict]:
        """Generate responses and evaluate for all test examples with concurrent batched API calls and periodic saving."""
        
        print(f"\n🔧 Generating responses for: {prompt_type}")
        print("="*60)
        print(f"   • API batch size: {self.batch_size} (concurrent API calls)")
        print(f"   • Save frequency: every {self.save_frequency} examples")
        print(f"   • Positive examples: {self.num_positive}")
        print(f"   • Negative examples: {self.num_negative}")
        
        results = []
        responses_file = self.output_dir / "responses" / f"{prompt_type}_responses.jsonl"
        
        # Clear file if it exists (fresh start)
        if responses_file.exists():
            responses_file.unlink()
        
        # Process examples in concurrent batches
        total_processed = 0
        for batch_start in tqdm(range(0, len(df), self.batch_size), 
                                desc=f"Processing {prompt_type}"):
            batch_end = min(batch_start + self.batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]
            
            # Process batch concurrently using async
            batch_results = asyncio.run(self._process_batch_async(batch_df, prompt_type, unified_prompt))
            
            # Add batch results to overall results
            results.extend(batch_results)
            total_processed += len(batch_results)
            
            # Log any errors
            for i, result in enumerate(batch_results):
                if result.get("Error"):
                    print(f"\n   ⚠️  Error in batch at index {batch_start + i}: {result['Error']}")
            
            # Periodic save at save_frequency intervals
            if total_processed % self.save_frequency == 0 or total_processed == len(df):
                # Calculate how many results to save
                results_since_last_save = total_processed % self.save_frequency
                if results_since_last_save == 0:
                    results_since_last_save = self.save_frequency
                
                # Save the most recent results
                results_to_save = results[-results_since_last_save:]
                append_mode = total_processed > results_since_last_save
                
                self._save_batch_results(results_to_save, responses_file, append=append_mode)
                print(f"\n   💾 Saved progress at {total_processed} examples")
        
        print(f"   ✓ Completed {len(results)} responses, saved to {responses_file.name}")
        
        return results
    
    def _save_batch_results(self, batch_results: List[Dict], file_path: Path, append: bool = True):
        """Save a batch of results to file."""
        mode = 'a' if append else 'w'
        with open(file_path, mode) as f:
            for result in batch_results:
                f.write(json.dumps(result) + "\n")
    
    def evaluate_results(self, results: List[Dict], prompt_type: str) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        total = len(results)
        correct = sum(1 for r in results if r["Result"] == "Correct")
        accuracy = correct / total if total > 0 else 0
        
        # By category
        by_category = {}
        for result in results:
            category = result["Category"]
            if category not in by_category:
                by_category[category] = {"total": 0, "correct": 0}
            by_category[category]["total"] += 1
            if result["Result"] == "Correct":
                by_category[category]["correct"] += 1
        
        for cat in by_category:
            by_category[cat]["accuracy"] = by_category[cat]["correct"] / by_category[cat]["total"]
        
        # By calculator
        by_calculator = {}
        for result in results:
            calc_name = result["Calculator Name"]
            if calc_name not in by_calculator:
                by_calculator[calc_name] = {"total": 0, "correct": 0}
            by_calculator[calc_name]["total"] += 1
            if result["Result"] == "Correct":
                by_calculator[calc_name]["correct"] += 1
        
        for calc in by_calculator:
            by_calculator[calc]["accuracy"] = by_calculator[calc]["correct"] / by_calculator[calc]["total"]
        
        evaluation = {
            "prompt_type": prompt_type,
            "overall_accuracy": accuracy,
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "by_category": by_category,
            "by_calculator": by_calculator
        }
        
        return evaluation
    
    def run_complete_evaluation(self) -> Dict[str, Any]:
        """Run the complete contrastive evaluation pipeline."""
        
        print("="*80)
        print("CONTRASTIVE FEW-SHOT EVALUATION PIPELINE")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Load test data
        print("📋 Loading test data...")
        df = self.load_test_data()
        
        # Load unified prompt
        print("\n📄 Loading unified prompt...")
        unified_prompt = self.load_unified_prompt()
        
        # Use paper's reported GPT-4 accuracy as baseline
        print("\n📊 Baseline (from paper):")
        print("   • GPT-4 One-Shot Accuracy: 50.91%")
        print("   • Correct: 533/1047")
        print("   • Source: MedCalc-Bench paper")
        
        baseline_eval = {
            "prompt_type": "gpt4_one_shot_paper",
            "overall_accuracy": 0.5091,
            "total": 1047,
            "correct": 533,
            "incorrect": 514,
            "source": "MedCalc-Bench paper"
        }
        
        # Evaluate contrastive few-shot (our refined prompt)
        print("\n" + "="*80)
        print("EVALUATING: Contrastive Few-Shot Prompt (Refined)")
        print("="*80)
        contrastive_results = self.generate_and_evaluate(df, "contrastive_few_shot", unified_prompt)
        contrastive_eval = self.evaluate_results(contrastive_results, "contrastive_few_shot")
        
        print(f"\n📊 Contrastive Few-Shot Results:")
        print(f"   • Overall Accuracy: {contrastive_eval['overall_accuracy']:.2%}")
        print(f"   • Correct: {contrastive_eval['correct']}/{contrastive_eval['total']}")
        
        # Calculate improvement over paper baseline
        improvement = contrastive_eval['overall_accuracy'] - baseline_eval['overall_accuracy']
        print(f"\n📈 Improvement over paper baseline: {improvement:+.2%}")
        
        # Save evaluations
        eval_summary = {
            "timestamp": datetime.now().isoformat(),
            "test_set_size": len(df),
            "configuration": {
                "num_positive_examples": self.num_positive,
                "num_negative_examples": self.num_negative,
                "api_batch_size": self.batch_size,
                "save_frequency": self.save_frequency
            },
            "baseline_gpt4_paper": baseline_eval,
            "contrastive_few_shot": contrastive_eval,
            "improvement": improvement
        }
        
        eval_file = self.output_dir / "evaluations" / "evaluation_summary.json"
        with open(eval_file, 'w') as f:
            json.dump(eval_summary, f, indent=2)
        
        print(f"\n💾 Saved evaluation summary to: {eval_file}")
        
        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"\n📁 All outputs saved to: {self.output_dir}/")
        
        return {
            "baseline_eval": baseline_eval,
            "contrastive_results": contrastive_results,
            "contrastive_eval": contrastive_eval,
            "improvement": improvement,
            "output_dir": self.output_dir
        }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Contrastive Few-Shot Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--refined-prompts-dir',
        type=str,
        required=True,
        help='Directory containing refined prompts'
    )
    
    parser.add_argument(
        '--training-results-dir',
        type=str,
        required=True,
        help='Directory containing training evaluation results'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for evaluation results (default: auto-generated)'
    )
    
    parser.add_argument(
        '--num-test-examples',
        type=int,
        default=None,
        help='Number of test examples to evaluate (default: None = all 1047 examples)'
    )
    
    parser.add_argument(
        '--num-positive',
        type=int,
        default=1,
        help='Number of positive contrastive examples (default: 1)'
    )
    
    parser.add_argument(
        '--num-negative',
        type=int,
        default=1,
        help='Number of negative contrastive examples (default: 1)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Batch size for API calls (default: 10)'
    )
    
    parser.add_argument(
        '--save-frequency',
        type=int,
        default=50,
        help='Save progress every N examples (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Initialize and run evaluator
    evaluator = ContrastiveFewShotEvaluator(
        api_key=api_key,
        refined_prompts_dir=args.refined_prompts_dir,
        training_results_dir=args.training_results_dir,
        output_dir=args.output_dir,
        num_test_examples=args.num_test_examples,
        num_positive=args.num_positive,
        num_negative=args.num_negative,
        batch_size=args.batch_size,
        save_frequency=args.save_frequency
    )
    
    results = evaluator.run_complete_evaluation()
    
    print(f"\n✅ Evaluation completed successfully!")
    print(f"📁 Outputs: {results['output_dir']}")
    print(f"📈 Final improvement: {results['improvement']:+.2%}")


if __name__ == "__main__":
    main()

