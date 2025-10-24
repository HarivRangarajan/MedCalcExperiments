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
from openai import OpenAI
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class PromptRefinementPipeline:
    """Pipeline for iterative prompt refinement using feedback signals."""
    
    def __init__(self, 
                 api_key: str,
                 results_dir: str,
                 batch_size: int = 17,
                 max_iterations: int = None,
                 output_dir: str = None,
                 num_candidates: int = 5):
        """
        Initialize the refinement pipeline.
        
        Args:
            api_key: OpenAI API key
            results_dir: Directory containing evaluation results
            batch_size: Number of examples per refinement batch
            max_iterations: Maximum number of iterations (None = use all examples)
            output_dir: Output directory for refined prompts
            num_candidates: Number of candidate prompts to create (default: 5)
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.results_dir = Path(results_dir)
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.num_candidates = num_candidates
        
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(__file__).parent.parent / "outputs" / f"refined_prompts_{timestamp}"
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
        print(f"   • Number of candidates: {self.num_candidates}")
    
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
            
            # Store iteration info
            iteration_info = {
                "iteration": iteration,
                "prompt": refined_prompt,
                "num_correct": len(correct_batch),
                "num_incorrect": len(incorrect_batch),
                "timestamp": datetime.now().isoformat()
            }
            
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
    
    def create_candidate_prompts(self, 
                                refined_prompts: Dict[str, str],
                                num_candidates: int = 5) -> List[str]:
        """
        Create multiple candidate prompts from refined prompts, each with different angles.
        
        Args:
            refined_prompts: Dict mapping prompt_type to final refined prompt
            num_candidates: Number of candidate prompts to create
            
        Returns:
            List of candidate prompts
        """
        print(f"\n🔗 Creating {num_candidates} candidate prompts from refined prompts")
        print("="*60)
        
        # Define different angles for candidate prompts
        angles = [
            {
                "name": "Balanced Synthesis",
                "description": "Combines all refinements equally, balancing clarity, thoroughness, and conciseness"
            },
            {
                "name": "Precision-Focused",
                "description": "Emphasizes mathematical precision, unit conversion accuracy, and careful calculation steps"
            },
            {
                "name": "Context-Aware",
                "description": "Prioritizes understanding patient context, extracting relevant information, and clinical reasoning"
            },
            {
                "name": "Step-by-Step Methodical",
                "description": "Focuses on systematic problem-solving with clear, sequential reasoning steps"
            },
            {
                "name": "Error-Prevention",
                "description": "Emphasizes avoiding common mistakes, double-checking calculations, and validation"
            }
        ]
        
        candidates = []
        
        # Build base context
        base_context = """You are an expert prompt engineer. You have been given various different refined prompts that were optimized for medical calculation tasks through iterative refinement based on performance feedback.

**Refined Prompts:**

"""
        
        for prompt_type, prompt_text in refined_prompts.items():
            base_context += f"""
**{prompt_type.replace('_', ' ').title()} Prompt:**
```
{prompt_text}
```

"""
        
        # Create each candidate with a different angle
        for i in range(num_candidates):
            angle = angles[i % len(angles)]
            
            print(f"\n   Creating candidate {i+1}/{num_candidates}: {angle['name']}")
            
            instruction = base_context + f"""
**Your Task:**
Create a SINGLE UNIFIED PROMPT with a specific focus on: **{angle['name']}**

**Angle Description:** {angle['description']}

The unified prompt should:
1. Synthesize insights from all three refined prompts above
2. Apply the specific angle/focus described above while maintaining overall effectiveness
3. Be clear, coherent, and highly effective for medical calculations
4. Maintain compatibility with few-shot examples (runtime injection)
5. Ensure JSON output format with "step_by_step_thinking" and "answer" fields
6. Work well for different model types (GPT-4o, GPT-4o-mini, GPT-3.5-turbo)

**Output Format:**
Provide ONLY the unified prompt text. Do not include explanations, meta-commentary, or markdown formatting. Just output the final prompt that can be directly used.
"""
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert prompt engineer specializing in creating optimized prompts for different use cases."},
                        {"role": "user", "content": instruction}
                    ],
                    temperature=0.7,
                    max_tokens=4000
                )
                
                candidate_prompt = response.choices[0].message.content.strip()
                
                # Remove markdown code blocks if present
                if candidate_prompt.startswith("```"):
                    lines = candidate_prompt.split('\n')
                    candidate_prompt = '\n'.join(lines[1:-1]) if len(lines) > 2 else candidate_prompt
                
                candidates.append({
                    "candidate_id": i + 1,
                    "angle_name": angle['name'],
                    "angle_description": angle['description'],
                    "prompt": candidate_prompt
                })
                
                print(f"      ✓ Created ({len(candidate_prompt)} characters)")
                
            except Exception as e:
                print(f"      ⚠️  Error creating candidate: {e}")
                # Fallback: use a simple combination
                fallback = f"[Candidate {i+1} - {angle['name']}]\n\n" + list(refined_prompts.values())[0]
                candidates.append({
                    "candidate_id": i + 1,
                    "angle_name": angle['name'],
                    "angle_description": angle['description'],
                    "prompt": fallback
                })
        
        print(f"\n   ✅ Created {len(candidates)} candidate prompts")
        return candidates
    
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
    
    def run_complete_refinement(self) -> Dict[str, Any]:
        """Run the complete refinement pipeline."""
        
        print("="*80)
        print("ITERATIVE PROMPT REFINEMENT PIPELINE")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
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
                incorrect_examples
            )
            
            all_refinement_history[prompt_type] = refinement_history
            
            # Get final refined prompt
            if refinement_history:
                final_refined_prompts[prompt_type] = refinement_history[-1]['prompt']
            else:
                final_refined_prompts[prompt_type] = original_prompt
        
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
        
        # Create candidate prompts
        candidate_prompts = self.create_candidate_prompts(final_refined_prompts, self.num_candidates)
        
        # Save candidate prompts
        candidates_file = self.output_dir / "final" / "candidate_prompts.json"
        with open(candidates_file, 'w') as f:
            json.dump(candidate_prompts, f, indent=2)
        print(f"💾 Saved candidate prompts to: {candidates_file}")
        
        # Save each candidate as individual file for easy access
        (self.output_dir / "final" / "candidates").mkdir(exist_ok=True)
        for candidate in candidate_prompts:
            cand_file = self.output_dir / "final" / "candidates" / f"candidate_{candidate['candidate_id']}.txt"
            with open(cand_file, 'w') as f:
                f.write(candidate['prompt'])
        
        # Also create a unified prompt for backward compatibility
        unified_prompt = self.combine_refined_prompts(final_refined_prompts)
        unified_file = self.output_dir / "final" / "unified_prompt.txt"
        with open(unified_file, 'w') as f:
            f.write(unified_prompt)
        print(f"💾 Saved unified prompt (backward compatibility) to: {unified_file}")
        
        # Create summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "results_dir": str(self.results_dir),
            "output_dir": str(self.output_dir),
            "batch_size": self.batch_size,
            "max_iterations": self.max_iterations,
            "num_candidates": self.num_candidates,
            "prompt_types_processed": prompt_types,
            "refinement_iterations": {
                pt: len(history) for pt, history in all_refinement_history.items()
            },
            "final_prompts_file": str(final_prompts_file),
            "candidates_file": str(candidates_file),
            "unified_prompt_file": str(unified_file)
        }
        
        summary_file = self.output_dir / "refinement_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*80}")
        print("REFINEMENT COMPLETE")
        print(f"{'='*80}")
        print(f"\n📊 Summary:")
        print(f"   • Prompt types processed: {len(prompt_types)}")
        for pt in prompt_types:
            iters = len(all_refinement_history.get(pt, []))
            print(f"      - {pt}: {iters} iterations")
        print(f"   • Candidate prompts created: {len(candidate_prompts)}")
        print(f"   • Unified prompt (backward compat): {len(unified_prompt)} characters")
        print(f"\n📁 All outputs saved to: {self.output_dir}/")
        
        return {
            "refinement_history": all_refinement_history,
            "final_refined_prompts": final_refined_prompts,
            "candidate_prompts": candidate_prompts,
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
        help='Maximum number of iterations (default: None = use all examples)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for refined prompts (default: auto-generated)'
    )
    
    parser.add_argument(
        '--num-candidates',
        type=int,
        default=5,
        help='Number of candidate prompts to create (default: 5)'
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
        num_candidates=args.num_candidates
    )
    
    results = pipeline.run_complete_refinement()
    
    print(f"\n✅ Refinement pipeline completed successfully!")
    print(f"📁 Outputs: {results['output_dir']}")


if __name__ == "__main__":
    main()

