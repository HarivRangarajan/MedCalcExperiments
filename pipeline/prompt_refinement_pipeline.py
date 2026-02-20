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
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import asyncio
import re

# Import shared utilities
from shared_utils import (
    load_medcalc_one_shot_examples,
    extract_answer,
    create_one_shot_prompt,
    evaluate_answer
)


class PromptRefinementPipeline:
    """Pipeline for iterative prompt refinement using feedback signals."""
    
    def __init__(self, 
                 api_key: str,
                 results_dir: str,
                 batch_size: int = 10,
                 max_iterations: int = 5,
                 output_dir: str = None,
                 model: str = "gpt-5"):
        """
        Initialize the refinement pipeline.
        
        Args:
            api_key: OpenAI API key
            results_dir: Directory containing evaluation results
            batch_size: Number of examples per refinement batch
            max_iterations: Maximum number of iterations (None = use all examples)
            output_dir: Output directory for refined prompts
            model: OpenAI model to use for refinement (default: gpt-5)
        """
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.results_dir = Path(results_dir)
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(__file__).parent.parent / "outputs" / f"refined_prompts_{timestamp}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "iterations").mkdir(exist_ok=True)
        (self.output_dir / "final").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "evaluation_progress").mkdir(exist_ok=True)
        
        # Load training examples for evaluation
        self.training_examples = self._load_training_examples()
        
        # Load MedCalc one-shot examples for proper evaluation
        self.one_shot_examples = load_medcalc_one_shot_examples()
        
        # Track evaluation progress
        self.evaluation_history = []
        
        print(f"✅ Refinement pipeline initialized")
        print(f"   • Model: {self.model}")
        print(f"   • Results dir: {self.results_dir}")
        print(f"   • Output dir: {self.output_dir}")
        print(f"   • Batch size: {self.batch_size}")
        print(f"   • Max iterations: {self.max_iterations or 'unlimited'}")
        print(f"   • Training examples for evaluation: {len(self.training_examples)}")
        print(f"   • One-shot examples loaded: {len(self.one_shot_examples)}")
    
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
    
    def _load_training_examples(self) -> pd.DataFrame:
        """Load the 170 training examples used for contrastive generation."""
        # Load the saved indices
        indices_file = self.results_dir / "data" / "training_sample_indices.json"
        
        if not indices_file.exists():
            print(f"⚠️  Training sample indices not found, regenerating from correct/incorrect files...")
            # Regenerate from correct/incorrect files
            row_numbers = set()
            for subdir in ['correct', 'incorrect']:
                subdir_path = self.results_dir / subdir
                if subdir_path.exists():
                    for jsonl_file in subdir_path.glob('*.jsonl'):
                        with open(jsonl_file, 'r') as f:
                            for line in f:
                                data = json.loads(line)
                                row_numbers.add(data['Row Number'])
            
            # Save indices for future use
            (self.results_dir / "data").mkdir(exist_ok=True)
            with open(indices_file, 'w') as f:
                json.dump(sorted(list(row_numbers)), f)
            print(f"   ✓ Saved {len(row_numbers)} training indices")
        else:
            with open(indices_file, 'r') as f:
                row_numbers = json.load(f)
            print(f"   ✓ Loaded {len(row_numbers)} training sample indices")
        
        # Load the full train_data.csv
        train_data_path = Path(__file__).parent.parent / "MedCalc-Bench" / "dataset" / "train_data.csv"
        df = pd.read_csv(train_data_path)
        
        # Filter to only the 170 examples
        df_filtered = df[df['Row Number'].isin(row_numbers)].copy()
        
        print(f"   ✓ Loaded {len(df_filtered)} training examples for evaluation")
        return df_filtered
    
    async def _evaluate_single_example_async(self, prompt: str, row: pd.Series) -> Dict[str, Any]:
        """Evaluate a single example asynchronously."""
        try:
            # Get calculator-specific one-shot example
            calculator_id = str(row['Calculator ID'])
            one_shot_example = self.one_shot_examples.get(calculator_id, {})
            
            # Create messages with one-shot example using shared utility
            if one_shot_example:
                system_msg, user_msg = create_one_shot_prompt(
                    prompt,
                    row["Patient Note"],
                    row["Question"],
                    one_shot_example
                )
            else:
                # Fallback if no one-shot example available
                system_msg = prompt
                user_msg = f"Patient Note:\n{row['Patient Note']}\n\nQuestion: {row['Question']}"
            
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            
            # Generate response
            response = await self.async_client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            
            answer = response.choices[0].message.content
            answer = re.sub(r"\s+", " ", answer)
            
            # Extract answer value using shared utility
            answer_value = extract_answer(answer, int(row['Calculator ID']))
            
            # Check correctness using shared utility
            is_correct = evaluate_answer(
                answer_value,
                row['Ground Truth Answer'],
                int(row['Calculator ID']),
                row['Upper Limit'],
                row['Lower Limit']
            )
            
            return {
                'correct': is_correct,
                'answer': answer_value,
                'ground_truth': row['Ground Truth Answer']
            }
        except Exception as e:
            # Silently handle errors to avoid cluttering output
            return {'correct': False, 'answer': None, 'ground_truth': row['Ground Truth Answer']}
    
    async def _evaluate_prompt_on_training_set_async(self, prompt: str) -> float:
        """Evaluate a prompt on the 170 training examples."""
        print(f"\n      📊 Evaluating prompt on {len(self.training_examples)} training examples...")
        
        # Create tasks for all examples
        tasks = []
        for _, row in self.training_examples.iterrows():
            task = self._evaluate_single_example_async(prompt, row)
            tasks.append(task)
        
        # Run all evaluations in parallel (with batching to avoid rate limits)
        batch_size = 10
        all_results = []
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            batch_results = await asyncio.gather(*batch)
            all_results.extend(batch_results)
        
        # Calculate accuracy
        correct_count = sum(1 for r in all_results if r['correct'])
        accuracy = correct_count / len(all_results) if all_results else 0.0
        
        print(f"      ✓ Accuracy: {accuracy:.2%} ({correct_count}/{len(all_results)} correct)")
        return accuracy
    
    def evaluate_prompt_on_training_set(self, prompt: str) -> float:
        """Synchronous wrapper for evaluation."""
        return asyncio.run(self._evaluate_prompt_on_training_set_async(prompt))
    
    def plot_evaluation_progress(self):
        """Generate a plot showing accuracy improvement over iterations."""
        if not self.evaluation_history:
            print("⚠️  No evaluation history to plot")
            return
        
        iterations = [entry['iteration'] for entry in self.evaluation_history]
        accuracies = [entry['accuracy'] for entry in self.evaluation_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, accuracies, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Refinement Iteration', fontsize=12)
        plt.ylabel('Accuracy on Training Set (%)', fontsize=12)
        plt.title('Prompt Refinement Progress: Accuracy vs Iteration', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))
        
        # Add value labels on points
        for i, (it, acc) in enumerate(zip(iterations, accuracies)):
            plt.text(it, acc, f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "evaluation_progress" / "accuracy_progress.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Evaluation progress plot saved: {plot_path}")
        
        # Also save as PDF
        pdf_path = self.output_dir / "evaluation_progress" / "accuracy_progress.pdf"
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.close()
        
        # Save data as JSON
        data_path = self.output_dir / "evaluation_progress" / "accuracy_history.json"
        with open(data_path, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2)
        print(f"📄 Evaluation history saved: {data_path}")
    
    def create_refinement_instruction(self, 
                                     current_prompt: str,
                                     correct_batch: List[Dict],
                                     incorrect_batch: List[Dict],
                                     iteration: int) -> str:
        """Create the instruction for the LLM to refine the prompt."""
        
        instruction = f"""You are an expert prompt engineer specializing in medical calculation tasks. Your goal is to refine and improve a prompt based on feedback from its performance.

**Current Prompt (Iteration {iteration}):**
```
{current_prompt}
```

**Performance Feedback:**

CORRECT Responses ({len(correct_batch)} examples):
These responses were CORRECT. Analyze what the prompt did well to produce accurate results.
"""
        
        for i, ex in enumerate(correct_batch, 1):
            instruction += f"""
Example {i}:
- Calculator: {ex['Calculator Name']}
- Patient Note: {ex['Patient Note']}
- Question: {ex['Question']}
- LLM Answer: {ex['LLM Answer']}
- Ground Truth: {ex['Ground Truth Answer']}
"""

        instruction += f"""

INCORRECT Responses ({len(incorrect_batch)} examples):
These responses were INCORRECT. Analyze what went wrong and how to fix it.
"""

        for i, ex in enumerate(incorrect_batch, 1):
            instruction += f"""
Example {i}:
- Calculator: {ex['Calculator Name']}
- Patient Note: {ex['Patient Note']}
- Question: {ex['Question']}
- LLM Answer: {ex['LLM Answer']}
- Ground Truth: {ex['Ground Truth Answer']}
- Explanation: {ex['LLM Explanation']}
"""
        
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
        
        instruction = self.create_refinement_instruction(
            current_prompt, correct_batch, incorrect_batch, iteration
        )
        
        try:
            # GPT-5 only supports default temperature (1), other models support 0.7
            api_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an expert prompt engineer. Provide refined prompts based on performance feedback."},
                    {"role": "user", "content": instruction}
                ],
                "max_completion_tokens": 4000
            }
            
            # Only add temperature for non-GPT-5 models
            if "gpt-5" not in self.model.lower():
                api_params["temperature"] = 0.7
            
            response = self.client.chat.completions.create(**api_params)
            
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
                            incorrect_examples: List[Dict]) -> List[Dict]:
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
            
            # Evaluate the refined prompt on training set
            accuracy = self.evaluate_prompt_on_training_set(refined_prompt)
            
            # Store iteration info
            iteration_info = {
                "iteration": iteration,
                "prompt": refined_prompt,
                "num_correct": len(correct_batch),
                "num_incorrect": len(incorrect_batch),
                "accuracy": accuracy,
                "timestamp": datetime.now().isoformat()
            }
            
            refinement_history.append(iteration_info)
            
            # Track evaluation history
            self.evaluation_history.append({
                "iteration": iteration,
                "accuracy": accuracy,
                "prompt_type": prompt_type
            })
            
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
            # GPT-5 only supports default temperature (1), other models support 0.7
            api_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an expert prompt engineer specializing in synthesizing multiple prompts into a unified, superior version."},
                    {"role": "user", "content": combination_instruction}
                ],
                "max_completion_tokens": 4000
            }
            
            # Only add temperature for non-GPT-5 models
            if "gpt-5" not in self.model.lower():
                api_params["temperature"] = 0.7
            
            response = self.client.chat.completions.create(**api_params)
            
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
    
    def load_all_training_examples(self) -> Tuple[List[Dict], List[Dict]]:
        """Load ALL correct and incorrect examples from all prompt types (CoT, CoD, etc.)."""
        all_correct = []
        all_incorrect = []
        
        # Find all prompt types from the results directory
        correct_dir = self.results_dir / "correct"
        incorrect_dir = self.results_dir / "incorrect"
        
        if correct_dir.exists():
            for file in correct_dir.glob("*.jsonl"):
                with open(file, 'r') as f:
                    for line in f:
                        all_correct.append(json.loads(line))
        
        if incorrect_dir.exists():
            for file in incorrect_dir.glob("*.jsonl"):
                with open(file, 'r') as f:
                    for line in f:
                        all_incorrect.append(json.loads(line))
        
        print(f"📦 Loaded training examples:")
        print(f"   • Correct: {len(all_correct)}")
        print(f"   • Incorrect: {len(all_incorrect)}")
        print(f"   • Total: {len(all_correct) + len(all_incorrect)}")
        
        return all_correct, all_incorrect
    
    def load_enhanced_prompts(self) -> Dict[str, str]:
        """Load the enhanced prompts created by PromptEngineer."""
        prompts_file = self.results_dir / "prompts" / "enhanced_prompts.json"
        
        if not prompts_file.exists():
            raise FileNotFoundError(f"Enhanced prompts file not found: {prompts_file}")
        
        with open(prompts_file, 'r') as f:
            enhanced_prompts = json.load(f)
        
        print(f"📄 Loaded enhanced prompts:")
        for prompt_type in enhanced_prompts.keys():
            print(f"   • {prompt_type}")
        
        return enhanced_prompts
    
    def create_initial_unified_prompt(self) -> str:
        """Create initial unified prompt by combining CoT and CoD enhanced prompts."""
        print(f"🔗 Creating initial unified prompt from existing enhanced prompts...")
        
        # Load enhanced prompts
        enhanced_prompts = self.load_enhanced_prompts()
        
        # Prepare combination instruction
        combination_instruction = """You are an expert prompt engineer. You have been given enhanced prompts that were created by PromptEngineer for medical calculation tasks.

Your task is to analyze these prompts and create ONE UNIFIED PROMPT that:
1. Combines the best practices and effective instructions from all provided prompts
2. Eliminates redundancy and contradictions
3. Creates a clear, coherent, and highly effective prompt
4. Maintains compatibility with few-shot examples (runtime injection)
5. Ensures JSON output format with "step_by_step_thinking" and "answer" fields

**Enhanced Prompts to Combine:**

"""
        
        for prompt_type, prompt_text in enhanced_prompts.items():
            combination_instruction += f"""
**{prompt_type.replace('_', ' ').title()} Enhanced Prompt:**
```
{prompt_text}
```

"""
        
        combination_instruction += """
**Your Task:**
Create a SINGLE unified prompt that synthesizes the strengths of all the above prompts. The unified prompt should:
- Be clear and concise while capturing key insights from all prompts
- Work effectively across different types of medical calculations
- Maintain the ability to inject few-shot examples at runtime
- Specify the JSON output format clearly
- Include the best reasoning strategies from all approaches

**Output Format:**
Provide ONLY the unified prompt text. Do not include explanations or meta-commentary. Just output the final prompt that can be directly used.
"""
        
        try:
            print(f"   • Combining {len(enhanced_prompts)} enhanced prompts using {self.model}...")
            
            # GPT-5 only supports default temperature (1), other models support 0.7
            api_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an expert prompt engineer specializing in synthesizing multiple prompts into a unified, superior version."},
                    {"role": "user", "content": combination_instruction}
                ],
                "max_completion_tokens": 4000
            }
            
            # Only add temperature for non-GPT-5 models
            if "gpt-5" not in self.model.lower():
                api_params["temperature"] = 0.7
            
            response = self.client.chat.completions.create(**api_params)
            
            unified_prompt = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if unified_prompt.startswith("```"):
                lines = unified_prompt.split('\n')
                unified_prompt = '\n'.join(lines[1:-1]) if len(lines) > 2 else unified_prompt
            
            print(f"   ✓ Initial unified prompt created ({len(unified_prompt)} characters)")
            
            return unified_prompt
            
        except Exception as e:
            print(f"   ⚠️  Error combining prompts: {e}")
            # Fallback: use the first available prompt
            return list(enhanced_prompts.values())[0]
    
    def run_complete_refinement(self) -> Dict[str, Any]:
        """Run the complete refinement pipeline with unified prompt approach."""
        
        print("="*80)
        print("UNIFIED PROMPT ITERATIVE REFINEMENT PIPELINE")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model: {self.model}\n")
        
        # Load ALL training examples from all prompt types
        print("📋 Loading training examples from all sources (CoT, CoD, etc.)...")
        all_correct, all_incorrect = self.load_all_training_examples()
        
        # Start with initial unified prompt
        print(f"\n📝 Creating initial unified prompt...")
        current_prompt = self.create_initial_unified_prompt()
        print(f"   ✓ Initial prompt created ({len(current_prompt)} characters)")
        
        # Save initial prompt
        initial_file = self.output_dir / "iterations" / "unified_iteration_0.json"
        with open(initial_file, 'w') as f:
            json.dump({
                "iteration": 0,
                "prompt": current_prompt,
                "description": "Initial unified prompt (baseline)"
            }, f, indent=2)
        
        # Evaluate initial prompt
        print(f"\n🎯 Evaluating initial prompt...")
        initial_accuracy = self.evaluate_prompt_on_training_set(current_prompt)
        self.evaluation_history.append({
            "iteration": 0,
            "accuracy": initial_accuracy,
            "prompt_type": "unified"
        })
        
        # Perform iterative refinement on the unified prompt
        print(f"\n🔧 Starting iterative refinement of unified prompt")
        print("="*60)
        
        refinement_history = self.iterative_refinement(
            "unified",
            current_prompt,
            all_correct,
            all_incorrect
        )
        
        # Get final refined prompt
        if refinement_history:
            unified_prompt = refinement_history[-1]['prompt']
            final_accuracy = refinement_history[-1]['accuracy']
        else:
            unified_prompt = current_prompt
            final_accuracy = initial_accuracy
        
        # Save refinement history
        history_file = self.output_dir / "refinement_history.json"
        with open(history_file, 'w') as f:
            json.dump({"unified": refinement_history}, f, indent=2)
        print(f"\n💾 Saved refinement history to: {history_file}")
        
        # Save final unified prompt
        unified_file = self.output_dir / "final" / "unified_prompt.txt"
        with open(unified_file, 'w') as f:
            f.write(unified_prompt)
        print(f"💾 Saved final unified prompt to: {unified_file}")
        
        # Also save as JSON for compatibility
        final_json = self.output_dir / "final" / "final_refined_prompts.json"
        with open(final_json, 'w') as f:
            json.dump({"unified": unified_prompt}, f, indent=2)
        
        # Create summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "results_dir": str(self.results_dir),
            "output_dir": str(self.output_dir),
            "batch_size": self.batch_size,
            "max_iterations": self.max_iterations,
            "approach": "unified_prompt_refinement",
            "training_examples": {
                "correct": len(all_correct),
                "incorrect": len(all_incorrect),
                "total": len(all_correct) + len(all_incorrect)
            },
            "refinement_iterations": len(refinement_history),
            "initial_accuracy": initial_accuracy,
            "final_accuracy": final_accuracy,
            "accuracy_improvement": final_accuracy - initial_accuracy,
            "unified_prompt_file": str(unified_file)
        }
        
        summary_file = self.output_dir / "refinement_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*80}")
        print("REFINEMENT COMPLETE")
        print(f"{'='*80}")
        print(f"\n📊 Summary:")
        print(f"   • Model used: {self.model}")
        print(f"   • Training examples: {len(all_correct) + len(all_incorrect)}")
        print(f"   • Refinement iterations: {len(refinement_history)}")
        print(f"   • Initial accuracy: {initial_accuracy:.2%}")
        print(f"   • Final accuracy: {final_accuracy:.2%}")
        print(f"   • Improvement: {(final_accuracy - initial_accuracy):+.2%}")
        print(f"   • Unified prompt: {len(unified_prompt)} characters")
        
        # Generate evaluation progress plot
        self.plot_evaluation_progress()
        
        print(f"\n📁 All outputs saved to: {self.output_dir}/")
        
        return {
            "refinement_history": refinement_history,
            "unified_prompt": unified_prompt,
            "initial_accuracy": initial_accuracy,
            "final_accuracy": final_accuracy,
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
        default=10,
        help='Number of examples per refinement batch (default: 10)'
    )

    parser.add_argument(
        '--max-iterations',
        type=int,
        default=5,
        help='Maximum number of iterations (default: 5, processing 50 examples total with batch_size=10)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for refined prompts (default: auto-generated)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-5',
        help='OpenAI model to use for refinement (default: gpt-5)'
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
        model=args.model
    )
    
    results = pipeline.run_complete_refinement()
    
    print(f"\n✅ Refinement pipeline completed successfully!")
    print(f"📁 Outputs: {results['output_dir']}")


if __name__ == "__main__":
    main()

