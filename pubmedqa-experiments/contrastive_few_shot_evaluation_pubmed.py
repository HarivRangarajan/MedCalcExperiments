#!/usr/bin/env python3
"""
Contrastive Few-Shot Evaluation Pipeline for PubMedQA

This module evaluates unified refined prompts using contrastive few-shot examples on PubMedQA data.
It only evaluates the unified refined prompt (no original baseline).

Usage:
    python contrastive_few_shot_evaluation_pubmed.py \
        --refined-prompts-dir /path/to/refined/prompts \
        --test-data-path /path/to/pqal_train.json \
        --training-results-dir /path/to/training/results
"""

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


class ContrastiveFewShotEvaluatorPubMed:
    """Evaluator using contrastive few-shot examples for PubMedQA."""
    
    def __init__(self,
                 api_key: str,
                 refined_prompts_dir: str,
                 test_data_path: str,
                 training_results_dir: str = None,
                 output_dir: str = None,
                 num_test_examples: int = None,
                 num_positive: int = 2,
                 num_negative: int = 2,
                 batch_size: int = 10,
                 save_frequency: int = 50):
        """
        Initialize the evaluator.
        
        Args:
            api_key: OpenAI API key
            refined_prompts_dir: Directory containing refined prompts
            test_data_path: Path to PubMedQA test data (pqal_train.json)
            training_results_dir: Directory with training results (for contrastive examples)
            output_dir: Output directory for evaluation results
            num_test_examples: Number of test examples to evaluate (None = all)
            num_positive: Number of positive contrastive examples (default: 2)
            num_negative: Number of negative contrastive examples (default: 2)
            batch_size: Batch size for OpenAI API calls (default: 10)
            save_frequency: Save progress every N examples (default: 50)
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.refined_prompts_dir = Path(refined_prompts_dir)
        self.test_data_path = Path(test_data_path)
        self.training_results_dir = Path(training_results_dir) if training_results_dir else None
        self.num_test_examples = num_test_examples
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.batch_size = batch_size
        self.save_frequency = save_frequency
        
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            suffix = f"_test{num_test_examples}" if num_test_examples else ""
            output_dir = Path(__file__).parent / "outputs" / f"contrastive_evaluation_pubmed_{timestamp}{suffix}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in ["responses", "evaluations", "logs", "visualizations"]:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        # Load contrastive examples if available
        self.contrastive_examples = self._load_contrastive_examples() if training_results_dir else {'correct': [], 'incorrect': []}
        
        print(f"✅ PubMedQA Contrastive evaluator initialized")
        print(f"   • Refined prompts: {self.refined_prompts_dir}")
        print(f"   • Test data: {self.test_data_path}")
        if training_results_dir:
            print(f"   • Training results: {self.training_results_dir}")
        print(f"   • Output dir: {self.output_dir}")
    
    def _load_contrastive_examples(self) -> Dict[str, List[Dict]]:
        """Load contrastive examples (correct and incorrect) from training results."""
        contrastive = {'correct': [], 'incorrect': []}
        
        if not self.training_results_dir or not self.training_results_dir.exists():
            print(f"   ⚠️  Training results directory not found, using zero contrastive examples")
            return contrastive
        
        # Load correct examples from correct/ subdirectory
        correct_dir = self.training_results_dir / "correct"
        if correct_dir.exists():
            correct_files = list(correct_dir.glob("*.jsonl"))
            for file in correct_files:
                try:
                    with open(file, 'r') as f:
                        for line in f:
                            if line.strip():
                                example = json.loads(line)
                                contrastive['correct'].append(example)
                except Exception as e:
                    print(f"   ⚠️  Error loading correct examples from {file}: {e}")
        else:
            print(f"   ⚠️  Correct examples directory not found: {correct_dir}")
        
        # Load incorrect examples from incorrect/ subdirectory
        incorrect_dir = self.training_results_dir / "incorrect"
        if incorrect_dir.exists():
            incorrect_files = list(incorrect_dir.glob("*.jsonl"))
            for file in incorrect_files:
                try:
                    with open(file, 'r') as f:
                        for line in f:
                            if line.strip():
                                example = json.loads(line)
                                contrastive['incorrect'].append(example)
                except Exception as e:
                    print(f"   ⚠️  Error loading incorrect examples from {file}: {e}")
        else:
            print(f"   ⚠️  Incorrect examples directory not found: {incorrect_dir}")
        
        print(f"   ✓ Loaded contrastive examples:")
        print(f"      - {len(contrastive['correct'])} correct examples")
        print(f"      - {len(contrastive['incorrect'])} incorrect examples")
        
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
    
    def load_test_data(self) -> List[Dict]:
        """Load PubMedQA test data."""
        with open(self.test_data_path, 'r') as f:
            data = json.load(f)
        
        # Convert to list format
        test_examples = []
        for pmid, example in data.items():
            test_examples.append({
                "PMID": pmid,
                "Question": example["QUESTION"],
                "Context": example["CONTEXTS"],
                "Long_Answer": example["LONG_ANSWER"],
                "Final_Decision": example["FINAL_DECISION"],
                "Year": example.get("YEAR", "Unknown"),
                "Labels": example.get("LABELS", []),
                "Meshes": example.get("MESHES", [])
            })
        
        if self.num_test_examples is not None:
            test_examples = test_examples[:self.num_test_examples]
            print(f"   ✓ Loaded {len(test_examples)} test examples (limited to first {self.num_test_examples})")
        else:
            print(f"   ✓ Loaded {len(test_examples)} test examples")
        
        return test_examples
    
    def get_contrastive_examples(self, 
                                 num_positive: int = None,
                                 num_negative: int = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Get random contrastive examples (no category filtering for PubMedQA).
        
        Returns:
            Tuple of (positive_examples, negative_examples)
        """
        # Use instance defaults if not specified
        if num_positive is None:
            num_positive = self.num_positive
        if num_negative is None:
            num_negative = self.num_negative
        
        # Sample positive examples
        available_positive = self.contrastive_examples['correct']
        if len(available_positive) >= num_positive:
            positive = random.sample(available_positive, num_positive)
        else:
            positive = available_positive
        
        # Sample negative examples
        available_negative = self.contrastive_examples['incorrect']
        if len(available_negative) >= num_negative:
            negative = random.sample(available_negative, num_negative)
        else:
            negative = available_negative
        
        return positive, negative
    
    def create_contrastive_few_shot_prompt(self,
                                          context: str,
                                          question: str,
                                          unified_prompt: str) -> Tuple[str, str]:
        """Create a contrastive few-shot prompt with positive and negative examples using unified prompt."""
        
        # Get contrastive examples
        positive_examples, negative_examples = self.get_contrastive_examples()
        
        # Build system message starting with unified prompt
        system_msg = unified_prompt
        
        # Add positive demonstrations
        if positive_examples:
            system_msg += f'\n\n**Examples of CORRECT Responses:**\n'
            for i, ex in enumerate(positive_examples, 1):
                system_msg += f'\nExample {i} (CORRECT):\n'
                system_msg += f'Context: {ex.get("Context", "")[:400]}...\n'
                system_msg += f'Question: {ex.get("Question", "")}\n'
                system_msg += f'LLM Answer: {ex.get("LLM_Answer", "")}\n'
                system_msg += f'LLM Reasoning: {ex.get("LLM_Reasoning", "")}\n'
                system_msg += f'Ground Truth: {ex.get("Ground_Truth_Decision", "")}\n'
        
        # Add negative demonstrations (what NOT to do)
        if negative_examples:
            system_msg += f'\n\n**Examples to AVOID (Common Mistakes):**\n'
            for i, ex in enumerate(negative_examples, 1):
                system_msg += f'\nExample {i} (INCORRECT - Learn from this mistake):\n'
                system_msg += f'Context: {ex.get("Context", "")[:400]}...\n'
                system_msg += f'Question: {ex.get("Question", "")}\n'
                system_msg += f'Incorrect LLM Answer: {ex.get("LLM_Answer", "")}\n'
                system_msg += f'Incorrect Reasoning: {ex.get("LLM_Reasoning", "")}\n'
                system_msg += f'Correct Answer Should Be: {ex.get("Ground_Truth_Decision", "")}\n'
        
        # User message with current task
        user_msg = f'''Context: {context}

Question: {question}

Please provide your response in JSON format with reasoning and answer.'''
        
        return system_msg, user_msg, positive_examples, negative_examples
    
    # def extract_answer(self, response: str) -> Tuple[str, str]:
    #     """Extract answer and reasoning from LLM response."""
    #     try:
    #         # Try to parse as JSON first
    #         response_clean = response.strip()
    #         if response_clean.startswith('```json'):
    #             response_clean = response_clean[7:]
    #         if response_clean.endswith('```'):
    #             response_clean = response_clean[:-3]
            
    #         parsed = json.loads(response_clean)
    #         reasoning = parsed.get("reasoning", "No reasoning provided")
    #         answer = parsed.get("answer", "").lower().strip()
            
    #         # Normalize answer
    #         if answer in ["yes", "y", "true", "1"]:
    #             answer = "yes"
    #         elif answer in ["no", "n", "false", "0"]:
    #             answer = "no"
    #         elif answer in ["maybe", "uncertain", "unclear", "mixed"]:
    #             answer = "maybe"
    #         else:
    #             answer = "maybe"  # Default fallback
                
    #         return answer, reasoning
            
    #     except:
    #         # Fallback parsing if JSON fails
    #         response_lower = response.lower()
    #         reasoning = response
            
    #         # Extract answer using keyword matching
    #         if any(word in response_lower for word in ["answer\": \"yes", "answer: yes", "clearly yes", "definitely yes"]):
    #             answer = "yes"
    #         elif any(word in response_lower for word in ["answer\": \"no", "answer: no", "clearly no", "definitely no"]):
    #             answer = "no"
    #         elif any(word in response_lower for word in ["answer\": \"maybe", "answer: maybe", "uncertain", "mixed"]):
    #             answer = "maybe"
    #         else:
    #             # Count positive vs negative indicators
    #             positive_indicators = len([w for w in ["yes", "support", "confirm", "positive", "true"] if w in response_lower])
    #             negative_indicators = len([w for w in ["no", "not", "negative", "false", "lack"] if w in response_lower])
                
    #             if positive_indicators > negative_indicators:
    #                 answer = "yes"
    #             elif negative_indicators > positive_indicators:
    #                 answer = "no"
    #             else:
    #                 answer = "maybe"
            
    #         return answer, reasoning

    def extract_pubmed_answer(self, response: str) -> Tuple[str, str, str]:
        """Extract answer, reasoning, and decision from PubMed response."""
        try:
            # Try to parse JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                reasoning = parsed.get("reasoning", "No reasoning provided")
                answer = parsed.get("answer", "No answer provided")
                decision = parsed.get("decision", "").lower()
                
                # Normalize decision to yes/no/maybe
                if "yes" in decision:
                    decision = "yes"
                elif "no" in decision:
                    decision = "no"
                elif "maybe" in decision:
                    decision = "maybe"
                else:
                    decision = "unclear"
                    
                return answer, reasoning, decision
        except:
            pass
        
        # Fallback: extract from free text
        reasoning = response
        
        # Look for yes/no/maybe patterns
        decision_patterns = [
            r'\b(yes|no|maybe)\b',
            r'\b(positive|negative)\b',
            r'\b(supported|not supported)\b',
            r'\b(effective|ineffective)\b',
            r'\b(significant|not significant)\b'
        ]
        
        decision = "unclear"
        for pattern in decision_patterns:
            matches = re.findall(pattern, response.lower())
            if matches:
                found = matches[-1]  # Take last match
                if found in ["yes", "positive", "supported", "effective", "significant"]:
                    decision = "yes"
                elif found in ["maybe"]:
                    decision = "maybe"
                else:
                    decision = "no"
                break
        
        return response, reasoning, decision
    
    async def _process_single_example_async(self,
                                            example: Dict,
                                            unified_prompt: str) -> Dict[str, Any]:
        """Process a single example asynchronously."""
        pmid = example["PMID"]
        question = example["Question"]
        context = example["Context"]
        
        try:
            # Create contrastive few-shot prompt
            system_msg, user_msg, selected_positive, selected_negative = self.create_contrastive_few_shot_prompt(
                context, question, unified_prompt
            )
            
            # Generate response asynchronously
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            
            response = await self.async_client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            
            raw_response = response.choices[0].message.content
            raw_response = re.sub(r"\s+", " ", raw_response)
            
            # Extract answer and reasoning
            llm_answer, llm_reasoning, llm_decision = self.extract_pubmed_answer(raw_response)
            
            # Check correctness
            ground_truth = example["Final_Decision"].lower().strip()
            correctness = (llm_decision == ground_truth)
            status = "Correct" if correctness else "Incorrect"
            
            # Prepare selected examples for storage (keep only essential fields)
            positive_examples_info = []
            for ex in selected_positive:
                positive_examples_info.append({
                    "PMID": ex.get("PMID", ""),
                    "Question": ex.get("Question", "")[:100] + "..." if len(ex.get("Question", "")) > 100 else ex.get("Question", ""),
                    "LLM_Decision": ex.get("LLM_Decision", ""),
                    "Ground_Truth_Decision": ex.get("Ground_Truth_Decision", "")
                })
            
            negative_examples_info = []
            for ex in selected_negative:
                negative_examples_info.append({
                    "PMID": ex.get("PMID", ""),
                    "Question": ex.get("Question", "")[:100] + "..." if len(ex.get("Question", "")) > 100 else ex.get("Question", ""),
                    "LLM_Decision": ex.get("LLM_Decision", ""),
                    "Ground_Truth_Decision": ex.get("Ground_Truth_Decision", "")
                })
                
            # Return result
            return {
                "PMID": pmid,
                "Question": question,
                "Context": context,
                "LLM_Answer": llm_answer,
                "LLM_Reasoning": llm_reasoning,
                "LLM_Decision": llm_decision,
                "Ground_Truth_Decision": ground_truth,
                "Ground_Truth_Answer": example["Long_Answer"],
                "Result": status,
                "Prompt_Type": "contrastive_few_shot",
                "Year": example["Year"],
                "Labels": example["Labels"],
                "Meshes": example["Meshes"],
                "Raw_Response": raw_response,
                "Selected_Positive_Examples": positive_examples_info,
                "Selected_Negative_Examples": negative_examples_info,
                "Num_Positive_Used": len(selected_positive),
                "Num_Negative_Used": len(selected_negative),
                "Error": None
            }
            
        except Exception as e:
            return {
                "PMID": pmid,
                "Question": question,
                "Context": context,
                "LLM_Answer": str(e),
                "LLM_Reasoning": str(e),
                "LLM_Decision": "maybe",
                "Ground_Truth_Decision": example["Final_Decision"].lower().strip(),
                "Ground_Truth_Answer": example["Long_Answer"],
                "Result": "Incorrect",
                "Prompt_Type": "contrastive_few_shot",
                "Year": example["Year"],
                "Labels": example["Labels"],
                "Meshes": example["Meshes"],
                "Raw_Response": str(e),
                "Selected_Positive_Examples": [],
                "Selected_Negative_Examples": [],
                "Num_Positive_Used": 0,
                "Num_Negative_Used": 0,
                "Error": str(e)
            }
    
    async def _process_batch_async(self, 
                                   batch_examples: List[Dict],
                                   unified_prompt: str) -> List[Dict]:
        """Process a batch of examples concurrently using asyncio.gather."""
        tasks = []
        for example in batch_examples:
            tasks.append(self._process_single_example_async(example, unified_prompt))
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        return list(results)
    
    def generate_and_evaluate(self,
                            examples: List[Dict],
                            unified_prompt: str) -> List[Dict]:
        """Generate responses and evaluate for all test examples with concurrent batched API calls and periodic saving."""
        
        print(f"\n🔧 Generating responses for: contrastive_few_shot")
        print("="*60)
        print(f"   • API batch size: {self.batch_size} (concurrent API calls)")
        print(f"   • Save frequency: every {self.save_frequency} examples")
        print(f"   • Positive examples: {self.num_positive}")
        print(f"   • Negative examples: {self.num_negative}")
        
        results = []
        responses_file = self.output_dir / "responses" / "contrastive_few_shot_responses.jsonl"
        
        # Clear file if it exists (fresh start)
        if responses_file.exists():
            responses_file.unlink()
        
        # Process examples in concurrent batches
        total_processed = 0
        for batch_start in tqdm(range(0, len(examples), self.batch_size), 
                                desc="Processing contrastive_few_shot"):
            batch_end = min(batch_start + self.batch_size, len(examples))
            batch_examples = examples[batch_start:batch_end]
            
            # Process batch concurrently using async
            batch_results = asyncio.run(self._process_batch_async(batch_examples, unified_prompt))
            
            # Add batch results to overall results
            results.extend(batch_results)
            total_processed += len(batch_results)
            
            # Log any errors
            for i, result in enumerate(batch_results):
                if result.get("Error"):
                    print(f"\n   ⚠️  Error in batch at index {batch_start + i}: {result['Error']}")
            
            # Periodic save at save_frequency intervals
            if total_processed % self.save_frequency == 0 or total_processed == len(examples):
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
    
    def evaluate_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        total = len(results)
        correct = sum(1 for r in results if r["Result"] == "Correct")
        accuracy = correct / total if total > 0 else 0
        
        # By answer type
        by_answer = {}
        for result in results:
            answer = result["Ground_Truth_Decision"]
            if answer not in by_answer:
                by_answer[answer] = {"total": 0, "correct": 0}
            by_answer[answer]["total"] += 1
            if result["Result"] == "Correct":
                by_answer[answer]["correct"] += 1
        
        for answer in by_answer:
            by_answer[answer]["accuracy"] = by_answer[answer]["correct"] / by_answer[answer]["total"]
        
        # By year
        by_year = {}
        for result in results:
            year = result["Year"]
            if year not in by_year:
                by_year[year] = {"total": 0, "correct": 0}
            by_year[year]["total"] += 1
            if result["Result"] == "Correct":
                by_year[year]["correct"] += 1
        
        for year in by_year:
            by_year[year]["accuracy"] = by_year[year]["correct"] / by_year[year]["total"]
        
        evaluation = {
            "prompt_type": "contrastive_few_shot",
            "overall_accuracy": accuracy,
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "by_answer_type": by_answer,
            "by_year": by_year
        }
        
        return evaluation
    
    def run_complete_evaluation(self) -> Dict[str, Any]:
        """Run the complete contrastive evaluation pipeline."""
        
        print("="*80)
        print("CONTRASTIVE FEW-SHOT EVALUATION PIPELINE - PUBMEDQA")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Load test data
        print("📋 Loading test data...")
        examples = self.load_test_data()
        
        # Load unified prompt
        print("\n📄 Loading unified prompt...")
        unified_prompt = self.load_unified_prompt()
        
        # Evaluate contrastive few-shot (unified refined prompt)
        print("\n" + "="*80)
        print("EVALUATING: Contrastive Few-Shot Prompt (Unified Refined)")
        print("="*80)
        contrastive_results = self.generate_and_evaluate(examples, unified_prompt)
        contrastive_eval = self.evaluate_results(contrastive_results)
        
        print(f"\n📊 Contrastive Few-Shot Results:")
        print(f"   • Overall Accuracy: {contrastive_eval['overall_accuracy']:.2%}")
        print(f"   • Correct: {contrastive_eval['correct']}/{contrastive_eval['total']}")
        
        # Save evaluations
        eval_summary = {
            "timestamp": datetime.now().isoformat(),
            "test_set_size": len(examples),
            "configuration": {
                "num_positive_examples": self.num_positive,
                "num_negative_examples": self.num_negative,
                "api_batch_size": self.batch_size,
                "save_frequency": self.save_frequency,
                "unified_prompt_source": str(self.refined_prompts_dir / "final" / "unified_prompt.txt")
            },
            "contrastive_few_shot": contrastive_eval
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
            "contrastive_results": contrastive_results,
            "contrastive_eval": contrastive_eval,
            "output_dir": self.output_dir
        }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Contrastive Few-Shot Evaluation Pipeline for PubMedQA",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--refined-prompts-dir',
        type=str,
        required=True,
        help='Directory containing refined prompts'
    )
    
    parser.add_argument(
        '--test-data-path',
        type=str,
        required=True,
        help='Path to PubMedQA test data (pqal_train.json)'
    )
    
    parser.add_argument(
        '--training-results-dir',
        type=str,
        default=None,
        help='Directory containing training evaluation results (optional)'
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
        help='Number of test examples to evaluate (default: None = all examples)'
    )
    
    parser.add_argument(
        '--num-positive',
        type=int,
        default=2,
        help='Number of positive contrastive examples (default: 2)'
    )
    
    parser.add_argument(
        '--num-negative',
        type=int,
        default=2,
        help='Number of negative contrastive examples (default: 2)'
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
    evaluator = ContrastiveFewShotEvaluatorPubMed(
        api_key=api_key,
        refined_prompts_dir=args.refined_prompts_dir,
        test_data_path=args.test_data_path,
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
    print(f"📊 Final accuracy: {results['contrastive_eval']['overall_accuracy']:.2%}")


if __name__ == "__main__":
    main()