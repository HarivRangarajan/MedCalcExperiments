#!/usr/bin/env python3
"""
Full Test Set Evaluation with Optimal Configuration

This script runs evaluation on the full test set (1047 examples) using the
optimal (prompt, model) pair identified by grid search.

Usage:
    python full_test_evaluation.py \
        --grid-search-dir <path> \
        --training-results-dir <path> \
        --test-all-models  # Optional: test all 3 models for comparison
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import argparse
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
import asyncio
import re
import pandas as pd

# Add MedCalc evaluation imports
sys.path.insert(0, str(Path(__file__).parent / "MedCalc-Bench" / "evaluation"))
try:
    from evaluate import check_correctness
except ImportError as e:
    print(f"⚠️  MedCalc evaluation imports failed: {e}")
    sys.exit(1)


class FullTestEvaluator:
    """Full test set evaluator using optimal configuration."""
    
    def __init__(self,
                 api_key: str,
                 grid_search_dir: str,
                 training_results_dir: str,
                 output_dir: str = None,
                 test_all_models: bool = False,
                 batch_size: int = 10,
                 save_frequency: int = 50):
        """
        Initialize full test evaluator.
        
        Args:
            api_key: OpenAI API key
            grid_search_dir: Directory with grid search results
            training_results_dir: Directory with training results
            output_dir: Output directory for results
            test_all_models: If True, test all 3 models for comparison
            batch_size: Batch size for concurrent API calls
            save_frequency: Save progress every N examples
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.grid_search_dir = Path(grid_search_dir)
        self.training_results_dir = Path(training_results_dir)
        self.test_all_models = test_all_models
        self.batch_size = batch_size
        self.save_frequency = save_frequency
        
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(__file__).parent / "full_test_results" / f"test_eval_{timestamp}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in ["results", "responses", "evaluations", "logs"]:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        # Load optimal configuration (per model)
        self.optimal_config = self._load_optimal_config()
        
        # Load optimal prompts (one per model)
        self.optimal_prompts = self._load_optimal_prompts()
        
        # Load test data
        self.test_data = self._load_test_data()
        
        # Load one-shot examples
        self.one_shot_examples = self._load_one_shot_examples()
        
        # Load contrastive examples
        self.contrastive_examples = self._load_contrastive_examples()
        
        # Default to 1 positive + 1 negative (matching contrastive_few_shot_evaluation.py)
        self.num_positive = 1
        self.num_negative = 1
        
        print(f"✅ Full test evaluator initialized")
        print(f"   • Test examples: {len(self.test_data)}")
        print(f"   • Positive examples per query: {self.num_positive}")
        print(f"   • Negative examples per query: {self.num_negative}")
        print(f"   • Test all models: {self.test_all_models}")
        print(f"\n   • Optimal prompts loaded:")
        for model, config in self.optimal_config['optimal_per_model'].items():
            print(f"      - {model}: Candidate {config['candidate_id']} ({config['candidate_name']}) @ {config['validation_accuracy']:.2%}")
        print(f"   • Output dir: {self.output_dir}")
    
    def _load_optimal_config(self) -> Dict:
        """Load optimal configuration from grid search."""
        config_file = self.grid_search_dir / "optimal" / "optimal_config.json"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Optimal config not found at {config_file}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        return config
    
    def _load_optimal_prompts(self) -> Dict[str, str]:
        """Load optimal prompt for each model."""
        optimal_prompts = {}
        
        for model in self.optimal_config['optimal_per_model'].keys():
            model_safe_name = model.replace('-', '_')
            prompt_file = self.grid_search_dir / "optimal" / f"optimal_prompt_{model_safe_name}.txt"
            
            if not prompt_file.exists():
                raise FileNotFoundError(f"Optimal prompt for {model} not found at {prompt_file}")
            
            with open(prompt_file, 'r') as f:
                optimal_prompts[model] = f.read()
        
        return optimal_prompts
    
    def _load_test_data(self) -> pd.DataFrame:
        """Load full MedCalc test data."""
        test_data_path = Path(__file__).parent / "MedCalc-Bench" / "dataset" / "test_data.csv"
        df = pd.read_csv(test_data_path)
        print(f"   ✓ Loaded {len(df)} test examples")
        return df
    
    def _load_one_shot_examples(self) -> Dict[str, Any]:
        """Load MedCalc's original one-shot examples."""
        try:
            one_shot_file = Path(__file__).parent / "MedCalc-Bench" / "evaluation" / "one_shot_finalized_explanation.json"
            if one_shot_file.exists():
                with open(one_shot_file, 'r') as f:
                    examples = json.load(f)
                return examples
            else:
                return {}
        except Exception:
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
        
        return contrastive
    
    def get_contrastive_examples(self, calculator_id: str) -> Tuple[List[Dict], List[Dict]]:
        """Get contrastive examples for a calculator ID."""
        import random
        
        if calculator_id not in self.contrastive_examples:
            return [], []
        
        calc_examples = self.contrastive_examples[calculator_id]
        
        # Sample positive examples
        available_positive = calc_examples['correct']
        if len(available_positive) >= self.num_positive:
            positive = random.sample(available_positive, self.num_positive)
        else:
            positive = available_positive
        
        # Sample negative examples
        available_negative = calc_examples['incorrect']
        if len(available_negative) >= self.num_negative:
            negative = random.sample(available_negative, self.num_negative)
        else:
            negative = available_negative
        
        return positive, negative
    
    def create_prompt_for_example(self, 
                                  row: pd.Series,
                                  prompt_text: str) -> Tuple[str, str]:
        """
        Create system and user messages with contrastive few-shot setup.
        Matches the exact setup from contrastive_few_shot_evaluation.py.
        """
        patient_note = row["Patient Note"]
        question = row["Question"]
        calculator_id = str(row["Calculator ID"])
        
        # Get one-shot example
        one_shot_example = self.one_shot_examples.get(calculator_id)
        
        # Get contrastive examples
        positive_examples, negative_examples = self.get_contrastive_examples(calculator_id)
        
        # Build system message with prompt
        system_msg = prompt_text
        
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
        user_msg = f'Here is the patient note:\n\n{patient_note}\n\nHere is the task:\n\n{question}\n\nPlease directly output the JSON dict with your step-by-step thinking and final answer.'
        
        return system_msg, user_msg
    
    def extract_answer(self, answer: str, calid: int) -> Tuple[str, str]:
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
            if extracted_answer in ["str(short_and_direct_answer_of_the_question)", 
                                   "str(value which is the answer to the question)", "X.XX"]:
                extracted_answer = "Not Found"
        
        # Handle different calculator output types
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
                try:
                    answer = eval(match[-1][0])
                    answer = str(answer)
                except:
                    answer = match[-1][0]
            else:
                answer = "N/A"
        
        return answer, explanation
    
    async def evaluate_single_example(self,
                                     row: pd.Series,
                                     prompt_text: str,
                                     model: str) -> Dict[str, Any]:
        """Evaluate a single example."""
        try:
            # Create prompt
            system_msg, user_msg = self.create_prompt_for_example(row, prompt_text)
            
            # Generate response
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.0  # Deterministic
            )
            
            answer = response.choices[0].message.content
            answer = re.sub(r"\s+", " ", answer)
            
            # Extract answer
            answer_value, explanation = self.extract_answer(answer, int(row["Calculator ID"]))
            
            # Check correctness (wrap in try-except to handle eval errors in check_correctness)
            try:
                correctness = check_correctness(
                    answer_value,
                    row["Ground Truth Answer"],
                    row["Calculator ID"],
                    row["Upper Limit"],
                    row["Lower Limit"]
                )
            except:
                # If check_correctness fails (e.g., eval error), mark as incorrect
                correctness = False
            
            return {
                "Row Number": int(row["Row Number"]),
                "Calculator Name": row["Calculator Name"],
                "Calculator ID": str(row["Calculator ID"]),
                "Category": row["Category"],
                "Note ID": str(row["Note ID"]),
                "Patient Note": row["Patient Note"],
                "Question": row["Question"],
                "LLM Answer": answer_value,
                "LLM Explanation": explanation,
                "Ground Truth Answer": row["Ground Truth Answer"],
                "Ground Truth Explanation": row["Ground Truth Explanation"],
                "Result": "Correct" if correctness else "Incorrect",
                "Model": model,
                "Error": None
            }
            
        except Exception as e:
            return {
                "Row Number": int(row["Row Number"]),
                "Calculator Name": row["Calculator Name"],
                "Calculator ID": str(row["Calculator ID"]),
                "Category": row["Category"],
                "Note ID": str(row["Note ID"]),
                "Patient Note": row["Patient Note"],
                "Question": row["Question"],
                "LLM Answer": str(e),
                "LLM Explanation": str(e),
                "Ground Truth Answer": row["Ground Truth Answer"],
                "Ground Truth Explanation": row["Ground Truth Explanation"],
                "Result": "Incorrect",
                "Model": model,
                "Error": str(e)
            }
    
    async def evaluate_model(self, 
                           model: str, 
                           prompt_text: str) -> List[Dict]:
        """Evaluate all test examples with a specific model."""
        
        print(f"\n🔧 Evaluating model: {model}")
        print("="*60)
        
        results = []
        responses_file = self.output_dir / "responses" / f"{model}_responses.jsonl"
        
        # Clear file if it exists
        if responses_file.exists():
            responses_file.unlink()
        
        # Process in batches
        total_processed = 0
        for batch_start in tqdm(range(0, len(self.test_data), self.batch_size),
                               desc=f"Processing {model}"):
            batch_end = min(batch_start + self.batch_size, len(self.test_data))
            batch_df = self.test_data.iloc[batch_start:batch_end]
            
            # Process batch concurrently
            tasks = []
            for idx in range(len(batch_df)):
                row = batch_df.iloc[idx]
                tasks.append(self.evaluate_single_example(row, prompt_text, model))
            
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            total_processed += len(batch_results)
            
            # Periodic save
            if total_processed % self.save_frequency == 0 or total_processed == len(self.test_data):
                results_since_last_save = total_processed % self.save_frequency
                if results_since_last_save == 0:
                    results_since_last_save = self.save_frequency
                
                results_to_save = results[-results_since_last_save:]
                append_mode = total_processed > results_since_last_save
                
                mode = 'a' if append_mode else 'w'
                with open(responses_file, mode) as f:
                    for result in results_to_save:
                        f.write(json.dumps(result) + "\n")
                
                print(f"\n   💾 Saved progress at {total_processed} examples")
        
        print(f"   ✓ Completed {len(results)} responses")
        
        # Calculate metrics
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
        
        evaluation = {
            "model": model,
            "overall_accuracy": accuracy,
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "by_category": by_category
        }
        
        # Save evaluation
        eval_file = self.output_dir / "evaluations" / f"{model}_evaluation.json"
        with open(eval_file, 'w') as f:
            json.dump(evaluation, f, indent=2)
        
        print(f"\n   📊 {model} Results:")
        print(f"      • Overall Accuracy: {accuracy:.2%}")
        print(f"      • Correct: {correct}/{total}")
        
        return results
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run full test set evaluation with each model using its optimal prompt."""
        
        print("="*80)
        print("FULL TEST SET EVALUATION")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"🎯 Using optimal prompts from grid search:")
        for model, config in self.optimal_config['optimal_per_model'].items():
            print(f"   • {model}:")
            print(f"      - Prompt: Candidate {config['candidate_id']} ({config['candidate_name']})")
            print(f"      - Val Accuracy: {config['validation_accuracy']:.2%}")
        
        all_evaluations = {}
        
        # Determine which models to test
        if self.test_all_models:
            models_to_test = list(self.optimal_prompts.keys())
            print(f"\n📋 Testing all {len(models_to_test)} models (each with its optimal prompt)")
        else:
            # Just test the best performing model on validation
            best_model = max(
                self.optimal_config['optimal_per_model'].items(),
                key=lambda x: x[1]['validation_accuracy']
            )[0]
            models_to_test = [best_model]
            print(f"\n📋 Testing only best performing model: {models_to_test[0]}")
        
        # Evaluate each model with its optimal prompt
        for model in models_to_test:
            print(f"\n{'='*80}")
            print(f"EVALUATING: {model.upper()}")
            print(f"{'='*80}")
            print(f"Using: Candidate {self.optimal_config['optimal_per_model'][model]['candidate_id']} - {self.optimal_config['optimal_per_model'][model]['candidate_name']}")
            
            # Use this model's optimal prompt
            model_prompt = self.optimal_prompts[model]
            results = asyncio.run(self.evaluate_model(model, model_prompt))
            
            # Calculate evaluation metrics
            total = len(results)
            correct = sum(1 for r in results if r["Result"] == "Correct")
            accuracy = correct / total if total > 0 else 0
            
            all_evaluations[model] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total
            }
        
        # Compare with baseline (MedCalc paper GPT-4)
        baseline_accuracy = 0.5091  # From paper
        
        print(f"\n{'='*80}")
        print("FINAL RESULTS")
        print(f"{'='*80}")
        
        print(f"\n📊 Baseline (MedCalc Paper):")
        print(f"   • GPT-4 One-Shot: {baseline_accuracy:.2%} (533/1047)")
        
        print(f"\n📊 Our Results:")
        for model, eval_data in all_evaluations.items():
            improvement = eval_data['accuracy'] - baseline_accuracy
            print(f"   • {model}: {eval_data['accuracy']:.2%} ({eval_data['correct']}/{eval_data['total']}) [{improvement:+.2%} vs baseline]")
        
        # Save final summary with per-model optimal prompts
        summary = {
            "timestamp": datetime.now().isoformat(),
            "grid_search_dir": str(self.grid_search_dir),
            "optimal_per_model": self.optimal_config['optimal_per_model'],
            "test_set_size": len(self.test_data),
            "baseline_gpt4_paper": {
                "accuracy": baseline_accuracy,
                "correct": 533,
                "total": 1047
            },
            "evaluations": all_evaluations,
            "evaluation_details": {
                model: {
                    "optimal_prompt_candidate": self.optimal_config['optimal_per_model'][model]['candidate_id'],
                    "optimal_prompt_name": self.optimal_config['optimal_per_model'][model]['candidate_name'],
                    "validation_accuracy": self.optimal_config['optimal_per_model'][model]['validation_accuracy'],
                    "test_accuracy": all_evaluations[model]['accuracy'],
                    "test_correct": all_evaluations[model]['correct'],
                    "test_total": all_evaluations[model]['total']
                }
                for model in all_evaluations.keys()
            },
            "output_dir": str(self.output_dir)
        }
        
        summary_file = self.output_dir / "test_evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Saved summary to: {summary_file}")
        
        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"\n📁 All outputs saved to: {self.output_dir}/")
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Full Test Set Evaluation with Optimal Configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--grid-search-dir',
        type=str,
        required=True,
        help='Directory containing grid search results with optimal config'
    )
    
    parser.add_argument(
        '--training-results-dir',
        type=str,
        required=True,
        help='Directory containing training results'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: auto-generated)'
    )
    
    parser.add_argument(
        '--test-all-models',
        action='store_true',
        help='Test all 3 models (gpt-4o, gpt-4o-mini, gpt-3.5-turbo) for comparison'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Batch size for concurrent API calls (default: 10)'
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
    evaluator = FullTestEvaluator(
        api_key=api_key,
        grid_search_dir=args.grid_search_dir,
        training_results_dir=args.training_results_dir,
        output_dir=args.output_dir,
        test_all_models=args.test_all_models,
        batch_size=args.batch_size,
        save_frequency=args.save_frequency
    )
    
    results = evaluator.run_full_evaluation()
    
    print(f"\n✅ Full test evaluation completed successfully!")
    print(f"📁 Outputs: {results['output_dir']}")


if __name__ == "__main__":
    main()

