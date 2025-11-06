"""MedCalc-Bench specific evaluation pipeline."""

import json
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

from promptengineer import PromptPipeline
from promptengineer.techniques.base import PromptContext

from .base_pipeline import BaseEvaluationPipeline
from ..utils.api_utils import create_openai_client


class MedCalcEvaluationPipeline(BaseEvaluationPipeline):
    """Complete evaluation pipeline for MedCalc-Bench with prompt engineering comparison."""
    
    @property
    def dataset_name(self) -> str:
        """Name of the dataset."""
        return "MedCalc"
    
    def __init__(self, api_key: str, output_dir: str = None,
                 sample_size: int = None,
                 max_responses: int = None,
                 llm_judge_sample_size: int = None,
                 budget_limit: float = None,
                 model: str = None):
        """Initialize MedCalc evaluation pipeline."""
        # Initialize PromptPipeline with a temporary output dir
        temp_output = Path(__file__).parent.parent / "outputs" / "temp_prompt_pipeline"
        self.prompt_pipeline = PromptPipeline(api_key=api_key, output_dir=str(temp_output))
        
        # Load MedCalc one-shot examples
        self.one_shot_examples = self._load_medcalc_one_shot_examples()
        
        # MedCalc-specific components
        self.medcalc_evaluator = self._load_medcalc_evaluator()
        
        # Call parent initialization
        super().__init__(api_key=api_key, output_dir=output_dir, 
                        sample_size=sample_size, max_responses=max_responses,
                        llm_judge_sample_size=llm_judge_sample_size,
                        budget_limit=budget_limit, model=model)
    
    def _load_medcalc_evaluator(self):
        """Load MedCalc evaluation module if available."""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "MedCalc-Bench" / "evaluation"))
            from llm_inference import GPTInference
            return GPTInference()
        except ImportError:
            print("⚠️  MedCalc evaluator not available")
            return None
    
    def _load_medcalc_one_shot_examples(self) -> Dict[str, Any]:
        """Load MedCalc's original one-shot examples for calculator-specific prompting."""
        try:
            one_shot_file = Path("MedCalc-Bench/evaluation/one_shot_finalized_explanation.json")
            if one_shot_file.exists():
                with open(one_shot_file, 'r') as f:
                    examples = json.load(f)
                print(f"✅ Loaded {len(examples)} calculator-specific one-shot examples")
                return examples
            else:
                print("⚠️  MedCalc one-shot examples not found")
                return {}
        except Exception as e:
            print(f"⚠️  Error loading one-shot examples: {e}")
            return {}
    
    def load_data(self, sample_size: int = None) -> pd.DataFrame:
        """Load and sample MedCalc-Bench test data."""
        if sample_size is None:
            sample_size = self.sample_size
        
        print(f"\n📋 STEP 1: Loading MedCalc-Bench Data (Sample: {sample_size})")
        print("="*60)
        
        # Load test data
        test_data_path = Path(__file__).parent.parent / "MedCalc-Bench" / "dataset" / "test_data.csv"
        df = pd.read_csv(test_data_path)
        
        print(f"✅ Loaded {len(df)} total examples from MedCalc-Bench")
        
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
        print(f"\n   📊 Sample distribution by category:")
        for cat, count in category_dist.items():
            print(f"      • {cat}: {count} ({count/len(sampled_df)*100:.1f}%)")
        
        return sampled_df
    
    def create_context(self) -> PromptContext:
        """Create PromptContext for MedCalc tasks (zero-shot style)."""
        return PromptContext(
            task_description="""You are a medical AI assistant specialized in performing medical calculations. 
            Given a patient note and a specific question about a medical calculation, you must:
            1. Extract relevant clinical values from the patient note
            2. Apply the appropriate medical calculation formula or rule
            3. Provide the final numerical answer with proper units
            4. Show your step-by-step reasoning""",
            domain="medical_calculations",
            constraints=[
                "Must extract accurate values from patient notes",
                "Must use correct medical calculation formulas",
                "Must provide precise numerical answers",
                "Must include proper units in final answer",
                "Must show clear step-by-step reasoning",
                "Cannot make assumptions about missing values"
            ],
            target_audience="healthcare professionals using medical calculators",
            success_criteria=[
                "Provides numerically accurate final answer",
                "Uses correct calculation methodology",
                "Extracts correct values from patient note",
                "Shows clear reasoning process",
                "Includes appropriate units",
                "Handles edge cases appropriately"
            ]
        )
    
    def create_one_shot_context(self) -> PromptContext:
        """Create PromptContext for MedCalc tasks with one-shot demonstration capability."""
        return PromptContext(
            task_description="""You are a medical AI assistant specialized in performing medical calculations.
            Given a patient note and a specific question about a medical calculation, you must:
            1. Extract relevant clinical values from the patient note
            2. Apply the appropriate medical calculation formula or rule  
            3. Provide the final numerical answer with proper units
            4. Show your step-by-step reasoning
            
            You should provide prompts that can incorporate relevant demonstration examples 
            specific to the type of calculation being performed. The demonstration should show
            the pattern of reasoning, value extraction, and formula application.""",
            domain="medical_calculations_with_demonstrations",
            constraints=[
                "Must be able to incorporate calculator-specific demonstration examples",
                "Must extract accurate values from patient notes",
                "Must use correct medical calculation formulas", 
                "Must provide precise numerical answers",
                "Must include proper units in final answer",
                "Must show clear step-by-step reasoning similar to demonstration examples",
                "Cannot make assumptions about missing values",
                "Should follow the reasoning pattern shown in demonstrations"
            ],
            target_audience="healthcare professionals using medical calculators",
            success_criteria=[
                "Can adapt to different types of medical calculations",
                "Provides numerically accurate final answer",
                "Uses correct calculation methodology",
                "Extracts correct values from patient note", 
                "Shows clear reasoning process following demonstration pattern",
                "Includes appropriate units",
                "Handles edge cases appropriately",
                "Demonstrates understanding of the calculation approach shown in examples"
            ]
        )
    
    def generate_enhanced_prompts(self, context: PromptContext, context_type: str = "") -> Dict[str, Any]:
        """Generate enhanced prompts using PromptEngineer techniques."""
        context_label = f" ({context_type})" if context_type else ""
        print(f"\n🚀 Generating PromptEngineer Prompts{context_label}")
        print("="*60)
        
        techniques = ["chain_of_thought", "chain_of_thoughtlessness", "chain_of_draft"]
        enhanced_prompts = self.prompt_pipeline.generate_enhanced_prompts(context, techniques)
        
        print(f"✅ PromptEngineer prompts generated{context_label}:")
        for technique, prompt_data in enhanced_prompts.items():
            print(f"   • {technique.replace('_', ' ').title()}: {len(prompt_data['prompt']):,} characters")
        
        return enhanced_prompts
    
    def get_original_medcalc_prompts(self) -> Dict[str, str]:
        """Get the original MedCalc-Bench prompt templates."""
        print("\n📝 STEP 3: Loading Original MedCalc Prompts")
        print("="*60)
        
        # Based on MedCalc-Bench evaluation code, these are the original prompt styles
        original_prompts = {
            "direct_answer": """Given the patient note and question below, provide the final numerical answer.

Patient Note: {patient_note}

Question: {question}

Final Answer:""",
            
            "zero_shot_cot": """Given the patient note and question below, think step by step and provide the final numerical answer.

Patient Note: {patient_note}

Question: {question}

Let's think step by step:""",
            
            "one_shot_cot": "DYNAMIC"  # Will be generated per calculator ID
        }
        
        print("✅ Original MedCalc prompts loaded:")
        for prompt_type, prompt in original_prompts.items():
            print(f"   • {prompt_type}: {len(prompt):,} characters")
        
        # Save original prompts
        prompts_file = self.output_dir / "prompts" / "original_medcalc_prompts.json"
        with open(prompts_file, 'w') as f:
            json.dump(original_prompts, f, indent=2)
        
        # Save information about calculator-specific examples
        if self.one_shot_examples:
            calc_info = {}
            for calc_id, example in self.one_shot_examples.items():
                calc_info[calc_id] = {
                    "has_example": True,
                    "patient_note_length": len(example["Patient Note"]),
                    "thinking_length": len(example["Response"]["step_by_step_thinking"]),
                    "answer": example["Response"]["answer"]
                }
            
            calc_info_file = self.output_dir / "prompts" / "calculator_specific_examples_info.json"
            with open(calc_info_file, 'w') as f:
                json.dump(calc_info, f, indent=2)
            
            print(f"   💡 Calculator-specific examples available for {len(self.one_shot_examples)} calculators")
        
        return original_prompts
    
    def get_calculator_specific_one_shot_prompt(self, calculator_id: str, patient_note: str, question: str) -> str:
        """Generate calculator-specific one-shot CoT prompt using MedCalc's examples."""
        # Get the specific example for this calculator
        if str(calculator_id) in self.one_shot_examples:
            example = self.one_shot_examples[str(calculator_id)]
            example_note = example["Patient Note"]
            example_thinking = example["Response"]["step_by_step_thinking"]
            example_answer = example["Response"]["answer"]
            
            # Build the one-shot prompt with the specific example
            prompt = f"""Given the patient note and question below, think step by step and provide the final numerical answer.

Here's an example of how to approach this type of calculation:

Example Patient Note: {example_note}

Example Question: {question}

Example Answer: {example_thinking}

Now solve this problem:

Patient Note: {patient_note}

Question: {question}

Let's think step by step:"""
            
            return prompt
        
        else:
            raise ValueError(f"No calculator-specific example found for calculator ID: {calculator_id}")
    
    def get_enhanced_calculator_specific_prompt(self, technique: str, calculator_id: str, 
                                               patient_note: str, question: str) -> str:
        """Generate calculator-specific enhanced one-shot prompt using PromptEngineer techniques."""
        # Get the base enhanced prompt template for this technique
        enhanced_prompts = getattr(self, '_cached_enhanced_one_shot_prompts', {})
        
        if f"{technique}_one_shot" not in enhanced_prompts:
            return None
        
        base_enhanced_prompt = enhanced_prompts[f"{technique}_one_shot"]["prompt"]
        
        # Get calculator-specific example if available
        if str(calculator_id) in self.one_shot_examples:
            example = self.one_shot_examples[str(calculator_id)]
            example_note = example["Patient Note"]
            example_thinking = example["Response"]["step_by_step_thinking"]
            example_answer = example["Response"]["answer"]
            
            # Create calculator-specific enhanced prompt by incorporating the example
            calculator_specific_prompt = f"""{base_enhanced_prompt}

**Calculator-Specific Demonstration:**

Example Patient Note: {example_note}

Example Question: {question}

Example Step-by-Step Reasoning: {example_thinking}

Final Answer: {example_answer}

**Now apply this approach to the current problem:**

Patient Note: {patient_note}

Question: {question}

Please follow the demonstrated reasoning pattern:"""
            
            return calculator_specific_prompt
        
        else:
            # Use the base enhanced prompt without specific example
            return f"""{base_enhanced_prompt}

Patient Note: {patient_note}

Question: {question}

Please provide your step-by-step reasoning:"""
    
    def generate_responses(self, df: pd.DataFrame,
                          enhanced_prompts: Dict[str, Any],
                          original_prompts: Dict[str, str],
                          max_examples: int = None) -> Dict[str, List[Dict[str, Any]]]:
        """Generate responses using both original and enhanced prompts."""
        if max_examples is None:
            max_examples = self.max_responses
        
        print("\n💬 STEP 4: Generating Responses")
        print("="*60)
        
        client = create_openai_client(self.api_key)
        
        if max_examples and max_examples < len(df):
            df = df.sample(n=max_examples, random_state=42)
            print(f"   📊 Randomly sampled {max_examples} examples for response generation")
        
        print(f"   Processing {len(df)} examples...")
        
        all_responses = {}
        
        # Process original prompts
        for prompt_type, prompt_template in original_prompts.items():
            print(f"\n   🔄 Generating responses for Original {prompt_type}...")
            responses = []
            
            for idx, row in df.iterrows():
                try:
                    # Special handling for one-shot CoT to use calculator-specific examples
                    if prompt_type == "one_shot_cot":
                        formatted_prompt = self.get_calculator_specific_one_shot_prompt(
                            calculator_id=row['Calculator ID'],
                            patient_note=row['Patient Note'],
                            question=row['Question']
                        )
                    else:
                        # Format prompt normally
                        formatted_prompt = prompt_template.format(
                            patient_note=row['Patient Note'],
                            question=row['Question']
                        )
                    
                    # Generate response
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a medical AI assistant specialized in medical calculations."},
                            {"role": "user", "content": formatted_prompt}
                        ],
                        temperature=0.1  # Low temperature for consistency
                    )
                    
                    response_data = {
                        "row_number": row['Row Number'],
                        "calculator_id": row['Calculator ID'],
                        "calculator_name": row['Calculator Name'],
                        "category": row['Category'],
                        "question": row['Question'],
                        "patient_note": row['Patient Note'],
                        "ground_truth_answer": row['Ground Truth Answer'],
                        "ground_truth_explanation": row['Ground Truth Explanation'],
                        "response": response.choices[0].message.content,
                        "prompt_type": f"original_{prompt_type}",
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Add information about which example was used for one-shot CoT
                    if prompt_type == "one_shot_cot":
                        calculator_id_str = str(row['Calculator ID'])
                        if calculator_id_str in self.one_shot_examples:
                            response_data["one_shot_example_used"] = "calculator_specific"
                            response_data["example_calculator_id"] = calculator_id_str
                        else:
                            response_data["one_shot_example_used"] = "generic_fallback"
                            response_data["example_calculator_id"] = "generic"
                    
                    responses.append(response_data)
                    
                    if (len(responses) % 10) == 0:
                        print(f"      ✅ Generated {len(responses)} responses")
                    
                except Exception as e:
                    print(f"      ❌ Error for row {idx}: {str(e)}")
                    continue
            
            all_responses[f"original_{prompt_type}"] = responses
            print(f"   ✅ Completed: {len(responses)} responses")
        
        # Process enhanced prompts
        for technique, prompt_data in enhanced_prompts.items():
            print(f"\n   🔄 Generating responses for PromptEngineer {technique}...")
            responses = []
            
            for idx, row in df.iterrows():
                try:
                    # Special handling for enhanced one-shot techniques
                    if technique.endswith("_one_shot"):
                        base_technique = technique.replace("_one_shot", "")
                        formatted_prompt = self.get_enhanced_calculator_specific_prompt(
                            technique=base_technique,
                            calculator_id=row['Calculator ID'],
                            patient_note=row['Patient Note'],
                            question=row['Question']
                        )
                        if formatted_prompt is None:
                            # Fallback to basic formatting
                            base_prompt = prompt_data['prompt']
                            formatted_prompt = f"""{base_prompt}

Patient Note: {row['Patient Note']}

Question: {row['Question']}

Please provide your answer:"""
                    else:
                        # Format prompt - enhanced prompts need special formatting
                        base_prompt = prompt_data['prompt']
                        formatted_prompt = f"""{base_prompt}

Patient Note: {row['Patient Note']}

Question: {row['Question']}

Please provide your answer:"""
                    
                    # Generate response
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a medical AI assistant specialized in medical calculations."},
                            {"role": "user", "content": formatted_prompt}
                        ],
                        temperature=0.1
                    )
                    
                    response_data = {
                        "row_number": row['Row Number'],
                        "calculator_id": row['Calculator ID'],
                        "calculator_name": row['Calculator Name'],
                        "category": row['Category'],
                        "question": row['Question'],
                        "patient_note": row['Patient Note'],
                        "ground_truth_answer": row['Ground Truth Answer'],
                        "ground_truth_explanation": row['Ground Truth Explanation'],
                        "response": response.choices[0].message.content,
                        "prompt_type": f"enhanced_{technique}",
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Add information about which example was used for enhanced one-shot techniques
                    if technique.endswith("_one_shot"):
                        calculator_id_str = str(row['Calculator ID'])
                        if calculator_id_str in self.one_shot_examples:
                            response_data["enhanced_one_shot_example_used"] = "calculator_specific"
                            response_data["example_calculator_id"] = calculator_id_str
                        else:
                            response_data["enhanced_one_shot_example_used"] = "generic_fallback"
                            response_data["example_calculator_id"] = "generic"
                    
                    responses.append(response_data)
                    
                    if (len(responses) % 10) == 0:
                        print(f"      ✅ Generated {len(responses)} responses")
                    
                except Exception as e:
                    print(f"      ❌ Error for row {idx}: {str(e)}")
                    continue
            
            all_responses[f"enhanced_{technique}"] = responses
            print(f"   ✅ Completed: {len(responses)} responses")
        
        # Save all responses
        responses_file = self.output_dir / "responses" / "all_responses.json"
        with open(responses_file, 'w') as f:
            json.dump(all_responses, f, indent=2)
        
        print(f"\n✅ Total responses generated: {sum(len(r) for r in all_responses.values())}")
        return all_responses
    
    def run_complete_evaluation(self, sample_size: int = None, max_responses: int = None, 
                               budget_limit: float = None) -> Dict[str, Any]:
        """Run the complete evaluation pipeline."""
        # Use configured parameters as defaults
        sample_size = sample_size if sample_size is not None else self.sample_size
        max_responses = max_responses if max_responses is not None else self.max_responses
        budget_limit = budget_limit if budget_limit is not None else self.budget_limit
        
        print("="*100)
        print("MEDCALC-BENCH PROMPT EVALUATION PIPELINE")
        print("="*100)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Sample size: {sample_size}")
        print(f"Budget limit: ${budget_limit}")
        
        # Calculate estimated costs and adjust if needed
        total_prompts = 9  # 3 original + 3 enhanced zero-shot + 3 enhanced one-shot
        max_responses = max_responses or sample_size
        
        # Estimate API calls
        estimated_response_calls = total_prompts * max_responses
        estimated_judge_calls = total_prompts * min(50, max_responses) if self.llm_judge else 0
        total_estimated_calls = estimated_response_calls + estimated_judge_calls
        
        # Rough cost estimate (assuming $0.01 per call average)
        estimated_cost = total_estimated_calls * 0.01
        
        print(f"Estimated API calls: {total_estimated_calls:,}")
        print(f"Estimated cost: ${estimated_cost:.2f}")
        
        if estimated_cost > budget_limit:
            print(f"⚠️  Estimated cost (${estimated_cost:.2f}) exceeds budget limit (${budget_limit:.2f})")
            
            # Auto-adjust to stay within budget
            max_calls_allowed = int(budget_limit / 0.01)
            if self.llm_judge:
                judge_calls = total_prompts * 25
                response_calls = max_calls_allowed - judge_calls
                adjusted_responses = response_calls // total_prompts
            else:
                adjusted_responses = max_calls_allowed // total_prompts
            
            print(f"🔧 Auto-adjusting: max_responses reduced to {adjusted_responses} per technique")
            max_responses = min(max_responses or sample_size, adjusted_responses)
            
            # Recalculate
            estimated_response_calls = total_prompts * max_responses
            estimated_judge_calls = total_prompts * min(25, max_responses) if self.llm_judge else 0
            total_estimated_calls = estimated_response_calls + estimated_judge_calls
            estimated_cost = total_estimated_calls * 0.01
            print(f"📊 Adjusted estimate: {total_estimated_calls:,} calls, ${estimated_cost:.2f} cost")
        
        # Step 1: Load data
        df = self.load_data(sample_size)
        
        # Step 2: Generate PromptEngineer prompts (both zero-shot and one-shot enhanced)
        print("\n🚀 STEP 2: Generating PromptEngineer Prompts")
        print("="*60)
        
        context_zero_shot = self.create_context()
        enhanced_prompts_zero_shot = self.generate_enhanced_prompts(context_zero_shot, "Zero-Shot Enhanced")
        
        context_one_shot = self.create_one_shot_context()
        enhanced_prompts_one_shot = self.generate_enhanced_prompts(context_one_shot, "One-Shot Enhanced")
        
        # Combine all enhanced prompts with distinguishing names
        enhanced_prompts = {}
        for technique, prompt_data in enhanced_prompts_zero_shot.items():
            enhanced_prompts[f"{technique}_zero_shot"] = prompt_data
        
        for technique, prompt_data in enhanced_prompts_one_shot.items():
            enhanced_prompts[f"{technique}_one_shot"] = prompt_data
        
        # Cache the enhanced one-shot prompts for calculator-specific use
        self._cached_enhanced_one_shot_prompts = enhanced_prompts
        
        # Save all enhanced prompts
        prompts_file = self.output_dir / "prompts" / "all_enhanced_prompts.json"
        with open(prompts_file, 'w') as f:
            json.dump(enhanced_prompts, f, indent=2)
        
        # Save contexts used for generation
        contexts_file = self.output_dir / "prompts" / "prompt_contexts.json"
        with open(contexts_file, 'w') as f:
            json.dump({
                "zero_shot_context": context_zero_shot.__dict__,
                "one_shot_context": context_one_shot.__dict__
            }, f, indent=2)
        
        print(f"\n✅ Generated {len(enhanced_prompts)} total enhanced prompts:")
        print(f"   • Zero-shot enhanced: {len(enhanced_prompts_zero_shot)} prompts")
        print(f"   • One-shot enhanced: {len(enhanced_prompts_one_shot)} prompts")
        
        # Step 3: Get original prompts
        original_prompts = self.get_original_medcalc_prompts()
        
        # Step 4: Generate responses
        responses = self.generate_responses(df, enhanced_prompts, original_prompts, max_responses)
        
        # Step 5: Evaluate accuracy
        accuracy_results = self.evaluate_accuracy(responses)
        
        # Step 6: LLM judge evaluation
        judge_results = self.evaluate_with_llm_judge(responses)
        
        # Step 7: Create visualizations
        self.create_visualizations(accuracy_results, judge_results)
        
        # Step 8: Generate report
        self.generate_report(accuracy_results, judge_results)
        
        # Final summary
        print("\n" + "="*100)
        print("✅ EVALUATION COMPLETE")
        print("="*100)
        
        print(f"\n Key Results:")
        best_overall = max(accuracy_results.items(), key=lambda x: x[1]['overall_accuracy'])
        from ..utils.evaluation_utils import clean_prompt_name
        print(f"   • Best overall: {clean_prompt_name(best_overall[0])} - {best_overall[1]['overall_accuracy']:.1%}")
        
        original_results = {k: v for k, v in accuracy_results.items() if k.startswith('original_')}
        enhanced_results = {k: v for k, v in accuracy_results.items() if k.startswith('enhanced_')}
        
        if original_results:
            best_original = max(original_results.items(), key=lambda x: x[1]['overall_accuracy'])
            print(f"   • Best original: {clean_prompt_name(best_original[0])} - {best_original[1]['overall_accuracy']:.1%}")
        
        if enhanced_results:
            best_enhanced = max(enhanced_results.items(), key=lambda x: x[1]['overall_accuracy'])
            print(f"   • Best enhanced: {clean_prompt_name(best_enhanced[0])} - {best_enhanced[1]['overall_accuracy']:.1%}")
        
        print(f"\n All results saved to: {self.output_dir}/")
        
        return {
            "data": df,
            "enhanced_prompts": enhanced_prompts,
            "original_prompts": original_prompts,
            "responses": responses,
            "accuracy_results": accuracy_results,
            "judge_results": judge_results,
            "output_directory": self.output_dir
        }

