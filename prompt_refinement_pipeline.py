#!/usr/bin/env python3
"""
Iterative Prompt Refinement Pipeline

This module performs iterative refinement of prompts using feedback from
correct and incorrect responses. It processes responses in batches and
uses GPT-4o to make guided edits to improve prompt performance.

Usage:
    python prompt_refinement_pipeline.py --results-dir <path> --batch-size 17
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
import warnings
import random
import asyncio
import re
import csv
warnings.filterwarnings('ignore')


class PromptRefinementPipeline:
    """Pipeline for iterative prompt refinement using feedback signals."""
    
    def __init__(self, 
                 api_key: str,
                 results_dir: str,
                 batch_size: int = 17,
                 max_iterations: int = None,
                 dataset: str = "medcalc",
                 output_dir: str = None,
                 validation_data_path: str = None,):
        """
        Initialize the refinement pipeline.
        
        Args:
            api_key: OpenAI API key
            results_dir: Directory containing evaluation results
            batch_size: Number of examples per refinement batch
            max_iterations: Maximum number of iterations (None = use all examples)
            output_dir: Output directory for refined prompts
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)  # ADD THIS LINE

        self.results_dir = Path(results_dir)
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.dataset = dataset.lower()
        self.validation_data_path = validation_data_path

        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(__file__).parent.parent / "outputs" / f"refined_prompts_{self.dataset}_{timestamp}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "iterations").mkdir(exist_ok=True)
        (self.output_dir / "final").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        print(f"✅ Refinement pipeline initialized")
        print(f"   • Results dir: {self.results_dir}")
        print(f"   • Output dir: {self.output_dir}")
        print(f"   • Batch size: {self.batch_size}")
        print(f"   • Max iterations: {self.max_iterations or 'unlimited'}")
    
    def load_results(self, prompt_type: str) -> Tuple[List[Dict], List[Dict]]:
        """Load correct and incorrect responses for a prompt type."""
        correct_file = self.results_dir / "correct" / f"{prompt_type}_correct.jsonl"
        incorrect_file = self.results_dir / "incorrect" / f"{prompt_type}_incorrect.jsonl"
        
        correct_examples = []
        incorrect_examples = []
        
        if correct_file.exists():
            with open(correct_file, 'r') as f:
                for line in f:
                    correct_examples.append(json.loads(line))
        
        if incorrect_file.exists():
            with open(incorrect_file, 'r') as f:
                for line in f:
                    incorrect_examples.append(json.loads(line))
        
        print(f"   Loaded {len(correct_examples)} correct, {len(incorrect_examples)} incorrect examples")
        return correct_examples, incorrect_examples
    
    def load_original_prompt(self, prompt_type: str) -> str:
        """Load the original enhanced prompt."""
        prompts_file = self.results_dir / "prompts" / "enhanced_prompts.json"
        
        with open(prompts_file, 'r') as f:
            prompts = json.load(f)
        
        if prompt_type in prompts:
            return prompts[prompt_type]['prompt']
        else:
            raise ValueError(f"Prompt type '{prompt_type}' not found in enhanced prompts")
        
    def create_refinement_instruction_pubmed(self, 
                                           current_prompt: str,
                                           correct_batch: List[Dict],
                                           incorrect_batch: List[Dict],
                                           iteration: int) -> str:
        """Create refinement instruction for PubMed QA tasks."""
        
        instruction = f"""You are an expert prompt engineer specializing in biomedical question answering tasks. Your goal is to refine and improve a prompt based on feedback from its performance.
    
    **Current Prompt (Iteration {iteration}):**
    ```
    {current_prompt}
    ```
        
    **Performance Feedback:**
    
    CORRECT Responses ({len(correct_batch)} examples):
    These responses were CORRECT. Analyze what the prompt did well to produce accurate results.
    """
        
        for i, ex in enumerate(correct_batch[:5], 1):  # Show first 5 for context
            instruction += f"""
    Example {i}:
    - Question: {ex['Question'][:200]}...
    - Context: {ex['Context'][:300]}...
    - LLM Answer: {ex['LLM_Answer']}
    - LLM Decision: {ex['LLM_Decision']}
    - LLM Reasoning: {ex['LLM_Reasoning'][:150]}...
    - Ground Truth Decision: {ex['Ground_Truth_Decision']}
    - Ground Truth Answer: {ex['Ground_Truth_Answer'][:200]}...
    - Category: {ex['Category']}
    """
        
        if len(correct_batch) > 5:
            instruction += f"\n(+{len(correct_batch) - 5} more correct examples)\n"
        
        instruction += f"""
    
    INCORRECT Responses ({len(incorrect_batch)} examples):
    These responses were INCORRECT. Analyze what went wrong and how to fix it.
    """
        
        for i, ex in enumerate(incorrect_batch[:5], 1):
            instruction += f"""
    Example {i}:
    - Question: {ex['Question'][:200]}...
    - Context: {ex['Context'][:300]}...
    - LLM Answer: {ex['LLM_Answer']}
    - LLM Decision: {ex['LLM_Decision']}
    - LLM Reasoning: {ex['LLM_Reasoning'][:150]}...
    - Ground Truth Decision: {ex['Ground_Truth_Decision']}
    - Ground Truth Answer: {ex['Ground_Truth_Answer'][:200]}...
    - Category: {ex['Category']}
    """
        
        if len(incorrect_batch) > 5:
            instruction += f"\n(+{len(incorrect_batch) - 5} more incorrect examples)\n"
        
        instruction += f"""
    
    **Your Task:**
    1. Analyze the incorrect responses to identify patterns of failure
    2. Analyze the correct responses to understand what works well
    3. Refine the prompt to:
    - Fix the issues causing incorrect responses
    - Maintain the strengths that lead to correct responses
    - Improve clarity, specificity, and guidance
    - Keep the few-shot/demonstration-injectable nature intact
    - Ensure JSON output format is maintained

    **Important Constraints:**
    - The prompt MUST remain compatible with runtime injection of one-shot or few-shot examples
    - The prompt MUST specify JSON output format with "step_by_step_thinking" and "answer" fields
    - Do NOT make it overly complex - keep it clear and actionable
    - Focus on fixing the specific failure patterns you identified

    **Output Format:**
    Provide ONLY the refined prompt text. Do not include explanations or meta-commentary. Just output the improved prompt that can be directly used.
    """
        
        return instruction
    
    def create_refinement_instruction(self, 
                                     current_prompt: str,
                                     correct_batch: List[Dict],
                                     incorrect_batch: List[Dict],
                                     iteration: int) -> str:
        """Create the instruction for GPT-4o to refine the prompt."""
        
        instruction = f"""You are an expert prompt engineer specializing in medical calculation tasks. Your goal is to refine and improve a prompt based on feedback from its performance.

**Current Prompt (Iteration {iteration}):**
```
{current_prompt}
```

**Performance Feedback:**

CORRECT Responses ({len(correct_batch)} examples):
These responses were CORRECT. Analyze what the prompt did well to produce accurate results.
"""
        
        for i, ex in enumerate(correct_batch[:5], 1):  # Show first 5 for context
            instruction += f"""
Example {i}:
- Question: {ex['Question'][:200]}...
- LLM Answer: {ex['LLM Answer']}
- Ground Truth: {ex['Ground Truth Answer']}
- Calculator: {ex['Calculator Name']}
"""
        
        if len(correct_batch) > 5:
            instruction += f"\n(+{len(correct_batch) - 5} more correct examples)\n"
        
        instruction += f"""

INCORRECT Responses ({len(incorrect_batch)} examples):
These responses were INCORRECT. Analyze what went wrong and how to fix it.
"""
        
        for i, ex in enumerate(incorrect_batch[:5], 1):
            instruction += f"""
Example {i}:
- Question: {ex['Question'][:200]}...
- LLM Answer: {ex['LLM Answer']}
- Ground Truth: {ex['Ground Truth Answer']}
- Calculator: {ex['Calculator Name']}
- Explanation: {ex['LLM Explanation'][:150]}...
"""
        
        if len(incorrect_batch) > 5:
            instruction += f"\n(+{len(incorrect_batch) - 5} more incorrect examples)\n"
        
        instruction += """

**Your Task:**
1. Analyze the incorrect responses to identify patterns of failure
2. Analyze the correct responses to understand what works well
3. Refine the prompt to:
   - Fix the issues causing incorrect responses
   - Maintain the strengths that lead to correct responses
   - Improve clarity, specificity, and guidance
   - Keep the few-shot/demonstration-injectable nature intact
   - Ensure JSON output format is maintained

**Important Constraints:**
- The prompt MUST remain compatible with runtime injection of one-shot or few-shot examples
- The prompt MUST specify JSON output format with "step_by_step_thinking" and "answer" fields
- Do NOT make it overly complex - keep it clear and actionable
- Focus on fixing the specific failure patterns you identified

**Output Format:**
Provide ONLY the refined prompt text. Do not include explanations or meta-commentary. Just output the improved prompt that can be directly used.
"""
        
        return instruction
    
    def refine_prompt_with_llm(self, 
                              current_prompt: str,
                              correct_batch: List[Dict],
                              incorrect_batch: List[Dict],
                              iteration: int) -> str:
        """Use GPT-4o to refine the prompt based on feedback."""
        
        instruction = ""
        if self.dataset == "pubmed":
            instruction = self.create_refinement_instruction_pubmed(
                current_prompt, correct_batch, incorrect_batch, iteration
            )
        else:  # default to medcalc
            instruction = self.create_refinement_instruction(
                current_prompt, correct_batch, incorrect_batch, iteration
            )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert prompt engineer. Provide refined prompts based on performance feedback."},
                    {"role": "user", "content": instruction}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            
            refined_prompt = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if refined_prompt.startswith("```"):
                lines = refined_prompt.split('\n')
                refined_prompt = '\n'.join(lines[1:-1]) if len(lines) > 2 else refined_prompt
            
            return refined_prompt
            
        except Exception as e:
            print(f"   ⚠️  Error refining prompt: {e}")
            return current_prompt  # Return unchanged if error
    
    def iterative_refinement(self, 
                            prompt_type: str,
                            original_prompt: str,
                            correct_examples: List[Dict],
                            incorrect_examples: List[Dict],
                            validation_examples: List[Dict] = None) -> List[Dict]:
        """
        Perform iterative refinement on a prompt.
        
        Returns:
            List of refinement iterations with prompts and metadata
        """
        print(f"\n🔧 Starting iterative refinement for: {prompt_type}")
        print("="*60)
        
        current_prompt = original_prompt
        refinement_history = []
        
        # Determine number of iterations
        total_examples = len(correct_examples) + len(incorrect_examples)
        max_possible_iterations = (total_examples + self.batch_size - 1) // self.batch_size
        
        if self.max_iterations:
            num_iterations = min(self.max_iterations, max_possible_iterations)
        else:
            num_iterations = max_possible_iterations
        
        print(f"   • Total examples: {total_examples}")
        print(f"   • Planned iterations: {num_iterations}")
        
        # Create batches by interleaving correct and incorrect
        all_examples = []
        for c, i in zip(correct_examples, incorrect_examples):
            all_examples.extend([c, i])
        # Add remaining
        if len(correct_examples) > len(incorrect_examples):
            all_examples.extend(correct_examples[len(incorrect_examples):])
        elif len(incorrect_examples) > len(correct_examples):
            all_examples.extend(incorrect_examples[len(correct_examples):])
        
        for iteration in range(1, num_iterations + 1):
            start_idx = (iteration - 1) * self.batch_size
            end_idx = start_idx + self.batch_size
            batch = all_examples[start_idx:end_idx]
            
            if not batch:
                break
            
            # Split batch into correct and incorrect
            correct_batch = [ex for ex in batch if ex.get('Result') == 'Correct']
            incorrect_batch = [ex for ex in batch if ex.get('Result') == 'Incorrect']
            
            print(f"\n   Iteration {iteration}/{num_iterations}:")
            print(f"      • Processing {len(correct_batch)} correct, {len(incorrect_batch)} incorrect")
            
            # Refine the prompt
            refined_prompt = self.refine_prompt_with_llm(
                current_prompt, correct_batch, incorrect_batch, iteration
            )
            
            # Store iteration info
            iteration_info = {
                "iteration": iteration,
                "prompt": refined_prompt,
                "num_correct": len(correct_batch),
                "num_incorrect": len(incorrect_batch),
                "timestamp": datetime.now().isoformat()
            }

            if validation_examples is not None:
                print(f"      • Running validation for iteration {iteration}...")
                validation_results = asyncio.run(self.validate_prompt_async(
                    refined_prompt,
                    validation_examples,
                    correct_examples,
                    incorrect_examples,
                    iteration=iteration,
                    prompt_type=prompt_type
                ))
                iteration_info['validation'] = validation_results  # ADD validation to iteration info
                print(f"      • Validation accuracy: {validation_results['metrics']['accuracy']:.2%}")
            
            refinement_history.append(iteration_info)
            
            # Save intermediate result
            iter_file = self.output_dir / "iterations" / f"{prompt_type}_iteration_{iteration}.json"
            with open(iter_file, 'w') as f:
                json.dump(iteration_info, f, indent=2)
            
            print(f"      ✓ Refined prompt saved")
            
            # Update current prompt for next iteration
            current_prompt = refined_prompt
        
        print(f"\n   ✅ Completed {len(refinement_history)} refinement iterations")
        
        return refinement_history
    
    def combine_refined_prompts(self, 
                               refined_prompts: Dict[str, str]) -> str:
        """
        Combine insights from all refined prompts into a single unified prompt.
        
        Args:
            refined_prompts: Dict mapping prompt_type to final refined prompt
            
        Returns:
            Unified prompt combining best practices from all
        """
        print(f"\n🔗 Combining refined prompts into unified prompt")
        print("="*60)
        
        combination_instruction = """You are an expert prompt engineer. You have been given three different refined prompts that were optimized for medical calculation tasks. Each prompt has been iteratively refined based on performance feedback.

Your task is to analyze these three prompts and create ONE UNIFIED PROMPT that:
1. Combines the best practices and effective instructions from all three
2. Eliminates redundancy and contradictions
3. Creates a clear, coherent, and highly effective prompt
4. Maintains compatibility with few-shot examples (runtime injection)
5. Ensures JSON output format with "step_by_step_thinking" and "answer" fields

**Refined Prompts to Combine:**

"""
        
        for prompt_type, prompt_text in refined_prompts.items():
            combination_instruction += f"""
**{prompt_type.replace('_', ' ').title()} Prompt:**
```
{prompt_text}
```

"""
        
        combination_instruction += """
**Your Task:**
Create a SINGLE unified prompt that synthesizes the strengths of all three prompts above. The unified prompt should:
- Be clear and concise while capturing key insights from all three
- Work effectively across different types of medical calculations
- Maintain the ability to inject few-shot examples at runtime
- Specify the JSON output format clearly
- Include the best reasoning strategies from all three approaches

**Output Format:**
Provide ONLY the unified prompt text. Do not include explanations or meta-commentary. Just output the final prompt that can be directly used.
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert prompt engineer specializing in synthesizing multiple prompts into a unified, superior version."},
                    {"role": "user", "content": combination_instruction}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            
            unified_prompt = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if unified_prompt.startswith("```"):
                lines = unified_prompt.split('\n')
                unified_prompt = '\n'.join(lines[1:-1]) if len(lines) > 2 else unified_prompt
            
            print(f"   ✅ Unified prompt created ({len(unified_prompt)} characters)")
            
            return unified_prompt
            
        except Exception as e:
            print(f"   ⚠️  Error combining prompts: {e}")
            # Fallback: use the best performing one
            return list(refined_prompts.values())[0]
        
    def add_validation_loop_to_pipeline(self, validation_data_path: str):
        """Add validation loop functionality to the pipeline."""
        
        # Add validation data path to instance
        self.validation_data_path = Path(validation_data_path)
        
        # Create validation subdirectories
        (self.output_dir / "validation").mkdir(exist_ok=True)
        (self.output_dir / "validation" / "responses").mkdir(exist_ok=True)
        (self.output_dir / "validation" / "metrics").mkdir(exist_ok=True)
        
        # Initialize validation tracking
        self.validation_history = []
    
    def load_validation_data(self) -> List[Dict]:
        """Load PubMedQA validation data."""
        with open(self.validation_data_path, 'r') as f:
            data = json.load(f)
        
        # Convert to list format (same as test data)
        validation_examples = []
        for pmid, example in data.items():
            validation_examples.append({
                "PMID": pmid,
                "Question": example["QUESTION"],
                "Context": example["CONTEXTS"],
                "Long_Answer": example["LONG_ANSWER"],
                "Final_Decision": example["FINAL_DECISION"],
                "Year": example.get("YEAR", "Unknown"),
                "Labels": example.get("LABELS", []),
                "Meshes": example.get("MESHES", [])
            })
        
        print(f"   ✓ Loaded {len(validation_examples)} validation examples")
        return validation_examples
    
    def _calculate_validation_metrics(self, 
                                validation_results: List[Dict],
                                iteration: int) -> Dict[str, Any]:
        """Calculate validation metrics."""
        total = len(validation_results)
        correct = sum(1 for r in validation_results if r["Result"] == "Correct")
        accuracy = correct / total if total > 0 else 0
        
        # By answer type
        by_answer = {}
        for result in validation_results:
            answer = result["Ground_Truth_Decision"]
            if answer not in by_answer:
                by_answer[answer] = {"total": 0, "correct": 0}
            by_answer[answer]["total"] += 1
            if result["Result"] == "Correct":
                by_answer[answer]["correct"] += 1
        
        for answer in by_answer:
            by_answer[answer]["accuracy"] = by_answer[answer]["correct"] / by_answer[answer]["total"]
        
        # Error analysis
        errors = [r for r in validation_results if r["Result"] == "Incorrect"]
        error_types = {}
        for error in errors:
            pred = error["LLM_Decision"]
            true = error["Ground_Truth_Decision"]
            error_type = f"predicted_{pred}_actual_{true}"
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        metrics = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "total_examples": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": accuracy,
            "by_answer_type": by_answer,
            "error_types": error_types,
            "error_rate": (total - correct) / total if total > 0 else 0
        }
        
        return metrics
    
    def _save_validation_accuracies_csv(self, prompt_type: str, iteration: int, accuracy: float):
        """Save validation accuracies to a CSV file for easy tracking."""
        
        csv_file = self.output_dir / "validation_accuracies.csv"
        
        # Check if CSV exists, if not create with headers
        if not csv_file.exists():
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Prompt_Type', 'Iteration', 'Accuracy', 'Timestamp'])
        
        # Append new accuracy data
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                prompt_type, 
                iteration, 
                f"{accuracy:.4f}",  # 4 decimal places
                datetime.now().isoformat()
            ])
        
        print(f"   📊 Updated validation accuracies CSV")

    def _save_validation_results(self, 
                           validation_results: List[Dict],
                           metrics: Dict[str, Any],
                           iteration: int,
                           prompt_type: str):
        """Save validation results and metrics."""
        (self.output_dir / "validation").mkdir(exist_ok=True)
        (self.output_dir / "validation" / "responses").mkdir(exist_ok=True)
        (self.output_dir / "validation" / "metrics").mkdir(exist_ok=True)
        
        # Save individual results
        results_file = self.output_dir / "validation" / "responses" / f"{prompt_type}_validation.jsonl"
        with open(results_file, 'w') as f:
            for result in validation_results:
                f.write(json.dumps(result) + "\n")
        
        # Save metrics
        metrics_file = self.output_dir / "validation" / "metrics" / f"{prompt_type}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self._save_validation_accuracies_csv(prompt_type, iteration, metrics['accuracy'])

        
        print(f"   ✓ Validation accuracy: {metrics['accuracy']:.2%}")
        print(f"   💾 Saved validation results to {results_file.name}")

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

                # If decision is empty or unclear, try to extract from answer
                if not decision or decision in ["", "unclear"]:
                    answer_text = answer.lower()  # Use the "answer" field
                    
                    # Look for yes/no/maybe in the answer field
                    if any(word in answer_text for word in ["yes", "positive", "supported", "effective", "significant"]):
                        decision = "yes"
                    elif any(word in answer_text for word in ["no", "negative", "not supported", "ineffective", "not significant"]):
                        decision = "no"
                    elif "maybe" in answer_text or "unclear" in answer_text:
                        decision = "maybe"
                
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

    def get_contrastive_examples_for_validation(self, 
                                          correct_examples: List[Dict],
                                          incorrect_examples: List[Dict],
                                          num_positive: int = 2,
                                          num_negative: int = 2) -> Tuple[List[Dict], List[Dict]]:
        """
        Get contrastive examples for validation from existing correct/incorrect examples.
        
        Args:
            correct_examples: Correct examples from load_results()
            incorrect_examples: Incorrect examples from load_results()
            num_positive: Number of positive examples
            num_negative: Number of negative examples
            
        Returns:
            Tuple of (positive_examples, negative_examples)
        """
        # Sample the requested number from existing examples
        if len(correct_examples) >= num_positive:
            selected_positive = random.sample(correct_examples, num_positive)
        else:
            selected_positive = correct_examples
            
        if len(incorrect_examples) >= num_negative:
            selected_negative = random.sample(incorrect_examples, num_negative)
        else:
            selected_negative = incorrect_examples
        
        return selected_positive, selected_negative

    def create_validation_contrastive_prompt(self,
                                        context: str,
                                        question: str,
                                        refined_prompt: str,
                                        correct_examples: List[Dict],
                                        incorrect_examples: List[Dict],
                                        num_positive: int = 2,
                                        num_negative: int = 2) -> Tuple[str, str]:
        """Create contrastive few-shot prompt for validation using existing examples."""
        
        # Get contrastive examples from existing data
        positive_examples, negative_examples = self.get_contrastive_examples_for_validation(
            correct_examples, incorrect_examples, num_positive, num_negative
        )
        
        # Build system message starting with refined prompt
        system_msg = refined_prompt
        
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
        
        # Add negative demonstrations
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
        
        return system_msg, user_msg
    
    async def validate_prompt_async(self, 
                               refined_prompt: str,
                               validation_examples: List[Dict],
                               correct_examples: List[Dict],
                               incorrect_examples: List[Dict],
                               iteration: int,
                               prompt_type: str = "validation",
                               batch_size: int = 5) -> Dict[str, Any]:
        """
        Validate refined prompt on validation set using contrastive few-shot.
        
        Args:
            refined_prompt: The refined prompt to validate
            validation_examples: Validation dataset
            correct_examples: Correct examples from load_results()
            incorrect_examples: Incorrect examples from load_results()
            iteration: Current iteration number
            batch_size: Batch size for API calls
            
        Returns:
            Validation results and metrics
        """
        print(f"\n🔍 Running validation for iteration {iteration}...")
        print(f"   • Validation set size: {len(validation_examples)}")
        print(f"   • Available positive examples: {len(correct_examples)}")
        print(f"   • Available negative examples: {len(incorrect_examples)}")
        print(f"   • Batch size: {batch_size}")
        
        validation_results = []
        
        # Process in batches
        for batch_start in tqdm(range(0, len(validation_examples), batch_size), 
                            desc=f"Validating iteration {iteration}"):
            batch_end = min(batch_start + batch_size, len(validation_examples))
            batch_examples = validation_examples[batch_start:batch_end]
            
            # Process batch concurrently
            tasks = []
            for example in batch_examples:
                tasks.append(self._validate_single_example_async(
                    example, refined_prompt, correct_examples, incorrect_examples, iteration
                ))
            
            batch_results = await asyncio.gather(*tasks)
            validation_results.extend(batch_results)
        
        # Calculate metrics
        metrics = self._calculate_validation_metrics(validation_results, iteration)
        
        # Save validation results
        self._save_validation_results(validation_results, metrics, iteration, prompt_type)
        
        return {
            "results": validation_results,
            "metrics": metrics,
            "iteration": iteration
        }
    
    async def _validate_single_example_async(self,
                                       example: Dict,
                                       refined_prompt: str,
                                       correct_examples: List[Dict],
                                       incorrect_examples: List[Dict],
                                       iteration: int) -> Dict[str, Any]:
        """Validate a single example asynchronously."""
        pmid = example["PMID"]
        question = example["Question"]
        context = example["Context"]
        
        try:
            # Create contrastive prompt using existing examples
            system_msg, user_msg = self.create_validation_contrastive_prompt(
                context, question, refined_prompt, correct_examples, incorrect_examples
            )
            
            # Generate response
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
            
            # Extract answer using PubMed extraction logic
            llm_answer, llm_reasoning, llm_decision = self.extract_pubmed_answer(raw_response)
            
            # Check correctness
            ground_truth = example["Final_Decision"].lower().strip()
            correctness = (llm_decision == ground_truth)
            status = "Correct" if correctness else "Incorrect"
            
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
                "Iteration": iteration,
                "Prompt_Type": "validation_contrastive",
                "Year": example["Year"],
                "Labels": example["Labels"],
                "Meshes": example["Meshes"],
                "Raw_Response": raw_response,
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
                "Iteration": iteration,
                "Prompt_Type": "validation_contrastive",
                "Year": example["Year"],
                "Labels": example["Labels"],
                "Meshes": example["Meshes"],
                "Raw_Response": str(e),
                "Error": str(e)
            }
    
    def run_complete_refinement(self) -> Dict[str, Any]:
        """Run the complete refinement pipeline."""
        
        print("="*80)
        print("ITERATIVE PROMPT REFINEMENT PIPELINE")
        if self.validation_data_path:  # Use self.validation_data_path
            print("WITH VALIDATION")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Initialize validation
        validation_examples = None
        validation_history = {}
        if self.validation_data_path:  # Use self.validation_data_path
            print("📋 Loading validation data...")
            validation_examples = self.load_validation_data()
        
        # Load enhanced prompts file to get prompt types
        prompts_file = self.results_dir / "prompts" / "enhanced_prompts.json"
        with open(prompts_file, 'r') as f:
            enhanced_prompts = json.load(f)
        
        prompt_types = list(enhanced_prompts.keys())
        print(f"📋 Found {len(prompt_types)} prompt types to refine:")
        for pt in prompt_types:
            print(f"   • {pt}")
        
        all_refinement_history = {}
        final_refined_prompts = {}
        
        # Refine each prompt type
        for prompt_type in prompt_types:
            print(f"\n{'='*80}")
            print(f"Processing: {prompt_type.upper()}")
            print(f"{'='*80}")
            
            # Load original prompt
            original_prompt = self.load_original_prompt(prompt_type)
            print(f"\n📄 Original prompt length: {len(original_prompt)} characters")
            
            # Load results
            correct_examples, incorrect_examples = self.load_results(prompt_type)
            
            # Perform iterative refinement
            refinement_history = self.iterative_refinement(
                prompt_type,
                original_prompt,
                correct_examples,
                incorrect_examples,
                validation_examples
            )
            
            all_refinement_history[prompt_type] = refinement_history
            
            # Get final refined prompt
            if refinement_history:
                final_refined_prompt = refinement_history[-1]['prompt']
                final_refined_prompts[prompt_type] = final_refined_prompt

                # NEW: Run validation if requested
                # if self.validation_data_path and validation_examples:  # Use self.validation_data_path
                #     print(f"\n✅ Validating refined {prompt_type} prompt...")
                #     validation_results = asyncio.run(self.validate_prompt_async(
                #         final_refined_prompt,
                #         validation_examples,
                #         correct_examples,
                #         incorrect_examples,
                #         iteration=len(refinement_history),
                #         prompt_type=prompt_type 
                #     ))
                #     validation_history[prompt_type] = validation_results

            else:
                final_refined_prompts[prompt_type] = original_prompt
                if self.validation_data_path:  # Use self.validation_data_path
                    validation_history[prompt_type] = None
        
        # Save all refinement histories
        history_file = self.output_dir / "refinement_history.json"
        with open(history_file, 'w') as f:
            json.dump(all_refinement_history, f, indent=2)
        print(f"\n💾 Saved refinement history to: {history_file}")
        
        # Save final refined prompts
        final_prompts_file = self.output_dir / "final" / "final_refined_prompts.json"
        with open(final_prompts_file, 'w') as f:
            json.dump(final_refined_prompts, f, indent=2)
        print(f"💾 Saved final refined prompts to: {final_prompts_file}")
        
        # Combine into unified prompt
        unified_prompt = self.combine_refined_prompts(final_refined_prompts)
        
        # Save unified prompt
        unified_file = self.output_dir / "final" / "unified_prompt.txt"
        with open(unified_file, 'w') as f:
            f.write(unified_prompt)
        print(f"💾 Saved unified prompt to: {unified_file}")
        
        # Create summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "results_dir": str(self.results_dir),
            "output_dir": str(self.output_dir),
            "batch_size": self.batch_size,
            "max_iterations": self.max_iterations,
            "prompt_types_processed": prompt_types,
            "refinement_iterations": {
                pt: len(history) for pt, history in all_refinement_history.items()
            },
            "final_prompts_file": str(final_prompts_file),
            "unified_prompt_file": str(unified_file)
        }
        
        summary_file = self.output_dir / "refinement_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # NEW: Save validation summary if we have validation data
        if self.validation_data_path and validation_history:
            validation_summary = {}
            for pt, validation_result in validation_history.items():
                if validation_result:
                    validation_summary[pt] = {
                        "accuracy": validation_result["metrics"]["accuracy"],
                        "total_examples": validation_result["metrics"]["total_examples"],
                        "by_answer_type": validation_result["metrics"]["by_answer_type"]
                    }
            
            if validation_summary:
                validation_file = self.output_dir / "validation_summary.json"
                with open(validation_file, 'w') as f:
                    json.dump(validation_summary, f, indent=2)
                print(f"💾 Saved validation summary to: {validation_file}")
                
        print(f"\n{'='*80}")
        print("REFINEMENT COMPLETE")
        print(f"{'='*80}")
        print(f"\n📊 Summary:")
        print(f"   • Prompt types processed: {len(prompt_types)}")
        for pt in prompt_types:
            iters = len(all_refinement_history.get(pt, []))
            print(f"      - {pt}: {iters} iterations")
        print(f"   • Unified prompt created: {len(unified_prompt)} characters")
        print(f"\n📁 All outputs saved to: {self.output_dir}/")
        
        return {
            "refinement_history": all_refinement_history,
            "final_refined_prompts": final_refined_prompts,
            "unified_prompt": unified_prompt,
            "summary": summary,
            "output_dir": self.output_dir
        }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Iterative Prompt Refinement Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Directory containing evaluation results'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=17,
        help='Number of examples per refinement batch (default: 17)'
    )
    
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=None,
        help='Maximum number of iterations'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for refined prompts (default: auto-generated)'
    )

    parser.add_argument('--dataset', 
        choices=['medcalc', 'pubmed'], 
        default='medcalc',
        help='Dataset type (medcalc or pubmed)'
    )

    parser.add_argument(
        '--validation-data',
        type=str,
        default=None,
        help='Path to validation dataset (e.g., pqal_val.json) - optional'
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Initialize and run pipeline
    pipeline = PromptRefinementPipeline(
        api_key=api_key,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
        max_iterations=args.max_iterations,
        output_dir=args.output_dir,
        dataset=args.dataset,
        validation_data_path=args.validation_data
    )
    
    results = pipeline.run_complete_refinement()
    
    print(f"\n✅ Refinement pipeline completed successfully!")
    print(f"📁 Outputs: {results['output_dir']}")


if __name__ == "__main__":
    main()

