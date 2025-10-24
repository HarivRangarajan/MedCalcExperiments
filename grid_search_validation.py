#!/usr/bin/env python3
"""
Grid Search on Validation Set

This script performs grid search over (candidate_prompt x model) combinations
on the validation set to find the optimal configuration.

Models tested:
- gpt-4o (highest capability)
- gpt-4o-mini (good balance)
- gpt-3.5-turbo (cost-effective)

Usage:
    python grid_search_validation.py \
        --refined-prompts-dir <path> \
        --validation-split-dir <path> \
        --training-results-dir <path>
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


class GridSearchEvaluator:
    """Grid search evaluator for prompt and model combinations."""
    
    # Models to evaluate (in order of capability/cost)
    MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo"
    ]
    
    def __init__(self,
                 api_key: str,
                 refined_prompts_dir: str,
                 validation_split_dir: str,
                 training_results_dir: str,
                 output_dir: str = None,
                 batch_size: int = 5,
                 num_positive: int = 1,
                 num_negative: int = 1):
        """
        Initialize grid search evaluator.
        
        Args:
            api_key: OpenAI API key
            refined_prompts_dir: Directory with candidate prompts
            validation_split_dir: Directory with validation split
            training_results_dir: Directory with training results (for contrastive examples)
            output_dir: Output directory for results
            batch_size: Batch size for concurrent API calls
            num_positive: Number of positive contrastive examples (default: 1)
            num_negative: Number of negative contrastive examples (default: 1)
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.refined_prompts_dir = Path(refined_prompts_dir)
        self.validation_split_dir = Path(validation_split_dir)
        self.training_results_dir = Path(training_results_dir)
        self.batch_size = batch_size
        self.num_positive = num_positive
        self.num_negative = num_negative
        
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(__file__).parent / "grid_search_results" / f"grid_search_{timestamp}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in ["results", "optimal", "logs"]:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        # Load candidate prompts
        self.candidate_prompts = self._load_candidate_prompts()
        
        # Load validation examples
        self.validation_examples = self._load_validation_examples()
        
        # Load one-shot examples
        self.one_shot_examples = self._load_one_shot_examples()
        
        # Load contrastive examples
        self.contrastive_examples = self._load_contrastive_examples()
        
        print(f"✅ Grid search evaluator initialized")
        print(f"   • Candidate prompts: {len(self.candidate_prompts)}")
        print(f"   • Models: {len(self.MODELS)}")
        print(f"   • Validation examples: {len(self.validation_examples)}")
        print(f"   • Positive examples per query: {self.num_positive}")
        print(f"   • Negative examples per query: {self.num_negative}")
        print(f"   • Total configurations: {len(self.candidate_prompts) * len(self.MODELS)}")
        print(f"   • Output dir: {self.output_dir}")
    
    def _load_candidate_prompts(self) -> List[Dict]:
        """Load candidate prompts."""
        candidates_file = self.refined_prompts_dir / "final" / "candidate_prompts.json"
        
        if not candidates_file.exists():
            raise FileNotFoundError(f"Candidate prompts not found at {candidates_file}")
        
        with open(candidates_file, 'r') as f:
            candidates = json.load(f)
        
        print(f"   ✓ Loaded {len(candidates)} candidate prompts")
        return candidates
    
    def _load_validation_examples(self) -> List[Dict]:
        """Load validation examples."""
        val_file = self.validation_split_dir / "validation_examples.jsonl"
        
        if not val_file.exists():
            raise FileNotFoundError(f"Validation examples not found at {val_file}")
        
        examples = []
        with open(val_file, 'r') as f:
            for line in f:
                examples.append(json.loads(line))
        
        print(f"   ✓ Loaded {len(examples)} validation examples")
        return examples
    
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
    
    def get_contrastive_examples(self, 
                                 calculator_id: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Get contrastive examples for a calculator ID.
        
        Returns:
            Tuple of (positive_examples, negative_examples)
        """
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
                                  example: Dict,
                                  candidate_prompt: str) -> Tuple[str, str]:
        """
        Create system and user messages with contrastive few-shot setup.
        Matches the exact setup from contrastive_few_shot_evaluation.py.
        """
        patient_note = example["Patient Note"]
        question = example["Question"]
        calculator_id = str(example["Calculator ID"])
        
        # Get one-shot example
        one_shot_example = self.one_shot_examples.get(calculator_id)
        
        # Get contrastive examples
        positive_examples, negative_examples = self.get_contrastive_examples(calculator_id)
        
        # Build system message with candidate prompt
        system_msg = candidate_prompt
        
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
                try:
                    answer = eval(match[-1][0])
                    answer = str(answer)
                except:
                    answer = match[-1][0]
            else:
                answer = "N/A"
        
        return answer, explanation
    
    async def evaluate_single_example(self,
                                     example: Dict,
                                     candidate_prompt: str,
                                     model: str) -> Dict[str, Any]:
        """Evaluate a single example with a specific prompt and model."""
        try:
            # Create prompt
            system_msg, user_msg = self.create_prompt_for_example(example, candidate_prompt)
            
            # Generate response
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.0  # Deterministic for evaluation
            )
            
            answer = response.choices[0].message.content
            answer = re.sub(r"\s+", " ", answer)
            
            # Extract answer
            answer_value, explanation = self.extract_answer(answer, int(example["Calculator ID"]))
            
            # Check correctness (wrap in try-except to handle eval errors in check_correctness)
            try:
                correctness = check_correctness(
                    answer_value,
                    example["Ground Truth Answer"],
                    example["Calculator ID"],
                    example.get("Upper Limit"),
                    example.get("Lower Limit")
                )
            except:
                # If check_correctness fails (e.g., eval error), mark as incorrect
                correctness = False
            
            return {
                "correct": correctness,
                "example": example,
                "answer": answer_value,
                "explanation": explanation
            }
            
        except Exception as e:
            return {
                "correct": False,
                "example": example,
                "answer": str(e),
                "explanation": str(e),
                "error": str(e)
            }
    
    async def evaluate_configuration(self,
                                    candidate_id: int,
                                    candidate_prompt: str,
                                    model: str) -> Dict[str, Any]:
        """Evaluate a (prompt, model) configuration on validation set."""
        
        print(f"\n   Evaluating: Candidate {candidate_id} + {model}")
        
        results = []
        
        # Process in batches
        for i in range(0, len(self.validation_examples), self.batch_size):
            batch = self.validation_examples[i:i+self.batch_size]
            
            # Evaluate batch concurrently
            tasks = []
            for example in batch:
                tasks.append(self.evaluate_single_example(example, candidate_prompt, model))
            
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        
        # Calculate metrics
        total = len(results)
        correct = sum(1 for r in results if r["correct"])
        accuracy = correct / total if total > 0 else 0
        
        print(f"      • Accuracy: {accuracy:.2%} ({correct}/{total})")
        
        return {
            "candidate_id": candidate_id,
            "model": model,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }
    
    def run_grid_search(self) -> Dict[str, Any]:
        """Run complete grid search to find best prompt for each model."""
        
        print("="*80)
        print("GRID SEARCH ON VALIDATION SET")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"📊 Grid Search Configuration:")
        print(f"   • Candidate prompts: {len(self.candidate_prompts)}")
        print(f"   • Models: {', '.join(self.MODELS)}")
        print(f"   • Validation size: {len(self.validation_examples)}")
        print(f"   • Total evaluations: {len(self.candidate_prompts) * len(self.MODELS)}")
        print(f"\n🎯 Goal: Find best prompt for EACH model")
        
        all_results = []
        
        # Evaluate each configuration
        print(f"\n{'='*80}")
        print("RUNNING GRID SEARCH")
        print(f"{'='*80}")
        
        for candidate in self.candidate_prompts:
            candidate_id = candidate['candidate_id']
            candidate_prompt = candidate['prompt']
            
            print(f"\n🔍 Candidate {candidate_id}: {candidate['angle_name']}")
            
            for model in self.MODELS:
                # Evaluate this configuration
                result = asyncio.run(self.evaluate_configuration(
                    candidate_id, candidate_prompt, model
                ))
                
                all_results.append({
                    "candidate_id": candidate_id,
                    "candidate_name": candidate['angle_name'],
                    "model": model,
                    "accuracy": result['accuracy'],
                    "correct": result['correct'],
                    "total": result['total']
                })
                
                # Save detailed results
                result_file = self.output_dir / "results" / f"candidate_{candidate_id}_{model}.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
        
        # Find optimal prompt for EACH model
        optimal_per_model = {}
        for model in self.MODELS:
            model_results = [r for r in all_results if r['model'] == model]
            best_for_model = max(model_results, key=lambda x: x['accuracy'])
            optimal_per_model[model] = best_for_model
        
        print(f"\n{'='*80}")
        print("GRID SEARCH RESULTS")
        print(f"{'='*80}")
        
        # Print all results sorted by accuracy
        print(f"\n📊 All Configurations (sorted by accuracy):")
        for r in sorted(all_results, key=lambda x: x['accuracy'], reverse=True):
            print(f"   • Candidate {r['candidate_id']} ({r['candidate_name']}) + {r['model']}: {r['accuracy']:.2%}")
        
        print(f"\n🏆 Optimal Prompt for Each Model:")
        for model, best in optimal_per_model.items():
            print(f"   • {model}:")
            print(f"      - Best prompt: Candidate {best['candidate_id']} ({best['candidate_name']})")
            print(f"      - Validation accuracy: {best['accuracy']:.2%}")
        
        # Save optimal configuration (one per model)
        optimal_config = {
            "timestamp": datetime.now().isoformat(),
            "optimal_per_model": {
                model: {
                    "candidate_id": best['candidate_id'],
                    "candidate_name": best['candidate_name'],
                    "validation_accuracy": best['accuracy'],
                    "validation_correct": best['correct'],
                    "validation_total": best['total']
                }
                for model, best in optimal_per_model.items()
            },
            "all_results": all_results
        }
        
        optimal_file = self.output_dir / "optimal" / "optimal_config.json"
        with open(optimal_file, 'w') as f:
            json.dump(optimal_config, f, indent=2)
        
        print(f"\n💾 Saved optimal configuration to: {optimal_file}")
        
        # Save optimal prompt for each model
        for model, best in optimal_per_model.items():
            optimal_candidate = [c for c in self.candidate_prompts if c['candidate_id'] == best['candidate_id']][0]
            model_safe_name = model.replace('-', '_')
            optimal_prompt_file = self.output_dir / "optimal" / f"optimal_prompt_{model_safe_name}.txt"
            with open(optimal_prompt_file, 'w') as f:
                f.write(optimal_candidate['prompt'])
            print(f"💾 Saved optimal prompt for {model} to: {optimal_prompt_file}")
        
        # Create summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "validation_split_dir": str(self.validation_split_dir),
            "refined_prompts_dir": str(self.refined_prompts_dir),
            "num_candidates": len(self.candidate_prompts),
            "num_models": len(self.MODELS),
            "models": self.MODELS,
            "validation_size": len(self.validation_examples),
            "optimal_per_model": optimal_per_model,
            "all_results": all_results,
            "output_dir": str(self.output_dir)
        }
        
        summary_file = self.output_dir / "grid_search_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Saved grid search summary to: {summary_file}")
        
        print(f"\n{'='*80}")
        print("GRID SEARCH COMPLETE")
        print(f"{'='*80}")
        print(f"\n📁 All outputs saved to: {self.output_dir}/")
        print(f"\n📋 Next step: Use each model with its optimal prompt in full test evaluation")
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Grid Search on Validation Set",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--refined-prompts-dir',
        type=str,
        required=True,
        help='Directory containing candidate prompts'
    )
    
    parser.add_argument(
        '--validation-split-dir',
        type=str,
        required=True,
        help='Directory containing validation split'
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
        '--batch-size',
        type=int,
        default=5,
        help='Batch size for concurrent API calls (default: 5)'
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
    
    args = parser.parse_args()
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Initialize and run grid search
    evaluator = GridSearchEvaluator(
        api_key=api_key,
        refined_prompts_dir=args.refined_prompts_dir,
        validation_split_dir=args.validation_split_dir,
        training_results_dir=args.training_results_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_positive=args.num_positive,
        num_negative=args.num_negative
    )
    
    results = evaluator.run_grid_search()
    
    print(f"\n✅ Grid search completed successfully!")
    print(f"📁 Outputs: {results['output_dir']}")


if __name__ == "__main__":
    main()

