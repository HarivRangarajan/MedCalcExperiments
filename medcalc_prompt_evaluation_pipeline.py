#!/usr/bin/env python3
"""
MedCalc-Bench Prompt Evaluation Pipeline

This script evaluates different prompt engineering techniques on the MedCalc-Bench dataset,
comparing:
1. Original MedCalc prompts (Direct, Zero-shot CoT, One-shot CoT)
2. PromptEngineer generated prompts (Chain of Thought, Chain of Thoughtlessness, Chain of Draft)

Evaluation includes:
- Built-in accuracy metrics (numerical answer comparison)
- LLM-as-a-judge evaluation for response quality
- Comprehensive visualizations and statistical analysis

Usage:
  python medcalc_prompt_evaluation_pipeline.py --sample-size 300 --output-dir results_experiment1
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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "promptengineer"))
sys.path.insert(0, str(Path(__file__).parent.parent / "Demonstration_Co_Create"))

# Import shared components
from promptengineer import PromptPipeline
from promptengineer.techniques.base import PromptContext

# Import the demonstration pipeline
from src.core.pipeline import DemonstrationPipeline

def load_api_key():
    """Load OpenAI API key from environment or local config."""
    import os
    
    # Try environment variable first
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key and api_key != "your-api-key-here":
        print("✅ API key loaded from environment variable")
        return api_key
    
    # Try local config file
    try:
        from config import OPENAI_API_KEY
        if OPENAI_API_KEY and OPENAI_API_KEY != "your-api-key-here":
            print("✅ API key loaded from local config.py")
            return OPENAI_API_KEY
    except ImportError:
        print("⚠️  Local config.py not found")
        pass
    
    print("❌ No valid API key found")
    return None

# Load API key
OPENAI_API_KEY = load_api_key()

# Import our custom LLM Judge
try:
    # Ensure current directory is in path for local modules
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    from modules.custom_llm_judge import CustomLLMJudge as LLMJudge
    print("✅ Custom LLM Judge imported successfully")
except ImportError as e:
    print(f"⚠️  Import warning: {e}")
    print("Custom LLM Judge not available - skipping LLM-as-a-judge evaluation")
    print(f"   Current directory: {Path(__file__).parent}")
    print(f"   Looking for: {Path(__file__).parent / 'modules' / 'custom_llm_judge.py'}")
    LLMJudge = None

# MedCalc evaluation imports
sys.path.insert(0, str(Path(__file__).parent / "MedCalc-Bench" / "evaluation"))
try:
    from llm_inference import GPTInference
    from evaluate import evaluate_answer
except ImportError as e:
    print(f"⚠️  MedCalc evaluation imports failed: {e}")


class MedCalcEvaluationPipeline:
    """Complete evaluation pipeline for MedCalc-Bench with prompt engineering comparison."""
    
    # Configuration constants
    DEFAULT_LLM_JUDGE_SAMPLE_SIZE = 20  # Maximum responses to evaluate per technique with LLM judge
    DEFAULT_RESPONSE_GENERATION_MAX = 5  # Maximum responses to generate per technique
    DEFAULT_SAMPLE_SIZE = 20  # Default number of examples to sample from dataset
    DEFAULT_BUDGET_LIMIT = 10.0  # Default budget limit in USD
    DEFAULT_MODEL = "gpt-4o"  # Default model for evaluations
    
    def __init__(self, api_key: str, output_dir: str = None, 
                 sample_size: int = None,
                 max_responses: int = None,
                 llm_judge_sample_size: int = None,
                 budget_limit: float = None,
                 model: str = None):
        """Initialize the evaluation pipeline."""
        self.api_key = api_key
        
        # Set configuration parameters with defaults
        self.sample_size = sample_size if sample_size is not None else self.DEFAULT_SAMPLE_SIZE
        self.max_responses = max_responses if max_responses is not None else self.DEFAULT_RESPONSE_GENERATION_MAX
        self.llm_judge_sample_size = llm_judge_sample_size if llm_judge_sample_size is not None else self.DEFAULT_LLM_JUDGE_SAMPLE_SIZE
        self.budget_limit = budget_limit if budget_limit is not None else self.DEFAULT_BUDGET_LIMIT
        self.model = model if model is not None else self.DEFAULT_MODEL
        
        if output_dir is None:
            # Create output directory in centralized outputs folder (relative to project root)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            project_root = Path(__file__).parent.parent
            output_dir = project_root / "outputs" / f"medcalc_evaluation_{timestamp}"
        self.output_dir = Path(output_dir)
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ["data", "prompts", "responses", "evaluations", "visualizations", "reports", "judge_prompts"]:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        # Initialize components
        self.prompt_pipeline = PromptPipeline(api_key=api_key, output_dir=str(self.output_dir))
        self.demo_pipeline = DemonstrationPipeline(api_key=api_key, output_dir=str(self.output_dir))
        
        if OPENAI_API_KEY and LLMJudge:
            self.llm_judge = LLMJudge(api_key=api_key, model=self.model)
            print(f"✅ Custom LLM Judge initialized with model: {self.model}")
        else:
            self.llm_judge = None
            if not OPENAI_API_KEY:
                print("⚠️  LLM Judge not available - no API key configured")
            elif not LLMJudge:
                print("⚠️  LLM Judge not available - CustomLLMJudge class not found")
                print("   Check that modules/custom_llm_judge.py is working correctly")
            print("   Skipping LLM-as-a-judge evaluation")
        
        # MedCalc-specific components
        self.medcalc_evaluator = GPTInference() if 'GPTInference' in globals() else None
        
        # Load MedCalc one-shot examples
        self.one_shot_examples = self._load_medcalc_one_shot_examples()
        
        print(f"✅ Pipeline initialized with output directory: {self.output_dir}")
    
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
                print("⚠️  MedCalc one-shot examples not found - using generic example")
                return {}
        except Exception as e:
            print(f"⚠️  Error loading one-shot examples: {e}")
            return {}
    
    def load_medcalc_data_by_calculator(self, calculator_id: int, sample_size: int = None) -> pd.DataFrame:
        """Load and sample MedCalc-Bench test data for a specific calculator."""
        if sample_size is None:
            sample_size = self.sample_size
        
        print(f"\n📋 STEP 1: Loading MedCalc-Bench Data for Calculator ID {calculator_id} (Sample: {sample_size})")
        print("="*60)
        
        # Load test data
        test_data_path = Path(__file__).parent / "MedCalc-Bench" / "dataset" / "test_data.csv"
        df = pd.read_csv(test_data_path)
        
        # Filter by calculator ID
        filtered_df = df[df['Calculator ID'] == calculator_id]
        
        if len(filtered_df) == 0:
            print(f"❌ No data found for Calculator ID {calculator_id}")
            return pd.DataFrame()
        
        print(f"✅ Found {len(filtered_df)} examples for Calculator ID {calculator_id}")
        
        # Get calculator info
        calculator_name = filtered_df['Calculator Name'].iloc[0]
        category = filtered_df['Category'].iloc[0]
        output_type = filtered_df['Output Type'].iloc[0]
        
        print(f"   • Calculator: {calculator_name}")
        print(f"   • Category: {category}")
        print(f"   • Output Type: {output_type}")
        
        # Random sampling
        if sample_size < len(filtered_df):
            sampled_df = filtered_df.sample(n=sample_size, random_state=42)
            print(f"   • Randomly sampled {sample_size} examples")
        else:
            sampled_df = filtered_df
            print(f"   • Using all {len(filtered_df)} examples (requested sample size >= total)")
        
        # Save sampled data
        sample_file = self.output_dir / "data" / f"sampled_calculator_{calculator_id}_data.csv"
        sampled_df.to_csv(sample_file, index=False)
        
        print(f"\n   📊 Sample contains {len(sampled_df)} examples for {calculator_name}")
        
        return sampled_df
    
    def load_medcalc_data(self, sample_size: int = None) -> pd.DataFrame:
        """Load and sample MedCalc-Bench test data."""
        if sample_size is None:
            sample_size = self.sample_size
        
        print(f"\n📋 STEP 1: Loading MedCalc-Bench Data (Sample: {sample_size})")
        print("="*60)
        
        # Load test data
        test_data_path = Path(__file__).parent / "MedCalc-Bench" / "dataset" / "test_data.csv"
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
    
    def create_medcalc_context(self) -> PromptContext:
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
    
    def create_medcalc_one_shot_context(self) -> PromptContext:
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
    
    def generate_promptengineer_prompts(self, context: PromptContext, context_type: str = "") -> Dict[str, Any]:
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
            # No calculator-specific example found
            raise ValueError(f"No calculator-specific example found for calculator ID: {calculator_id}")
    
    def get_enhanced_calculator_specific_prompt(self, technique: str, calculator_id: str, patient_note: str, question: str) -> str:
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
        
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        
        if max_examples and max_examples < len(df):
            # Use random sampling instead of just taking the first N
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
        
        cache_file = Path("demonstrations") / "demonstration_cache.json"
        demonstration_cache = {}
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    demonstration_cache = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                demonstration_cache = {}
        
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
        
        for technique, prompt_data in enhanced_prompts.items():
            responses = []
            for idx, row in df.iterrows():
                try:
                    cache_key = f"{technique}_{row['Calculator ID']}"
                    if cache_key in demonstration_cache:
                        print(f"   📋 Running cached demonstration for {technique} - {row['Calculator ID']}")
                        
                        # Use the cached final_prompt directly without appending patient note/question
                        cached_final_prompt = demonstration_cache[cache_key]['pipeline_results']['pipeline_summary']['few_shot_prompt']
                        
                        # Generate response for cached demonstration
                        demo_response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "You are a medical AI assistant specialized in medical calculations."},
                                {"role": "user", "content": cached_final_prompt}
                            ],
                            temperature=0.1
                        )
                        
                        demo_response_data = {
                            "row_number": row['Row Number'],
                            "calculator_id": row['Calculator ID'],
                            "calculator_name": row['Calculator Name'],
                            "category": row['Category'],
                            "question": row['Question'],
                            "patient_note": row['Patient Note'],
                            "ground_truth_answer": row['Ground Truth Answer'],
                            "ground_truth_explanation": row['Ground Truth Explanation'],
                            "response": demo_response.choices[0].message.content,
                            "prompt_type": f"demonstration_{technique}",
                            "timestamp": datetime.now().isoformat(),
                            "isDemo": True
                        }
                        responses.append(demo_response_data)

                        if (len(responses) % 10) == 0:
                            print(f"      ✅ Generated {len(responses)} responses")
                except Exception as e:
                    print(f"      ❌ Error for row {idx}: {str(e)}")
                    continue

            all_responses[f"demonstration_{technique}"] = responses
            print(f"   ✅ Completed: {len(responses)} demonstration responses")

        # Filter out any empty response lists before saving
        filtered_responses = {k: v for k, v in all_responses.items() if v}
        
        # Save only non-empty responses
        responses_file = self.output_dir / "responses" / "all_responses.json"
        with open(responses_file, 'w') as f:
            json.dump(filtered_responses, f, indent=2)
        
        print(f"\n✅ Total responses generated: {sum(len(r) for r in filtered_responses.values())}")
        print(f"✅ Response types with data: {len(filtered_responses)}")
        
        return filtered_responses
    
    def evaluate_accuracy(self, responses: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
        """Evaluate numerical accuracy using MedCalc built-in metrics."""
        print("\n🎯 STEP 5: Evaluating Numerical Accuracy")
        print("="*60)
        
        accuracy_results = {}
        
        for prompt_type, response_list in responses.items():
            print(f"\n   📊 Evaluating {prompt_type}...")
            
            correct = 0
            total = 0
            category_stats = {}
            
            for response_data in response_list:
                try:
                    # Extract numerical answer from response
                    predicted_answer = self._extract_numerical_answer(response_data['response'])
                    ground_truth = float(response_data['ground_truth_answer'])
                    
                    # Use MedCalc's evaluation function if available
                    if 'evaluate_answer' in globals():
                        is_correct = evaluate_answer(predicted_answer, ground_truth)
                    else:
                        # Fallback evaluation with tolerance
                        is_correct = self._evaluate_with_tolerance(predicted_answer, ground_truth)
                    
                    if is_correct:
                        correct += 1
                    total += 1
                    
                    # Track by category
                    category = response_data['category']
                    if category not in category_stats:
                        category_stats[category] = {'correct': 0, 'total': 0}
                    category_stats[category]['total'] += 1
                    if is_correct:
                        category_stats[category]['correct'] += 1
                    
                except Exception as e:
                    print(f"      ⚠️  Error evaluating response: {str(e)}")
                    total += 1  # Count as incorrect
                    continue
            
            # Calculate accuracy
            overall_accuracy = correct / total if total > 0 else 0
            
            # Calculate category accuracies
            category_accuracies = {}
            for cat, stats in category_stats.items():
                category_accuracies[cat] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            
            accuracy_results[prompt_type] = {
                'overall_accuracy': overall_accuracy,
                'correct': correct,
                'total': total,
                'category_accuracies': category_accuracies
            }
            
            print(f"      ✅ Overall accuracy: {overall_accuracy:.1%} ({correct}/{total})")
            for cat, acc in category_accuracies.items():
                stats = category_stats[cat]
                print(f"         • {cat}: {acc:.1%} ({stats['correct']}/{stats['total']})")
        
        # Save accuracy results
        accuracy_file = self.output_dir / "evaluations" / "accuracy_results.json"
        with open(accuracy_file, 'w') as f:
            json.dump(accuracy_results, f, indent=2)
        
        return accuracy_results
    
    def _extract_numerical_answer(self, response_text: str) -> float:
        """Extract numerical answer from response text."""
        # Look for patterns like "Final Answer: 123.45" or just numbers
        patterns = [
            r'(?:final answer|answer|result)[:=]\s*([0-9]*\.?[0-9]+)',
            r'([0-9]*\.?[0-9]+)\s*(?:ml/min|mg/dl|mmhg|%|kg|cm)',
            r'([0-9]*\.?[0-9]+)$'  # Number at end of text
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_text.lower())
            if matches:
                try:
                    return float(matches[-1])  # Take the last match
                except ValueError:
                    continue
        
        # If no pattern found, try to find any number
        numbers = re.findall(r'([0-9]*\.?[0-9]+)', response_text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        
        raise ValueError("No numerical answer found in response")
    
    def _evaluate_with_tolerance(self, predicted: float, ground_truth: float, tolerance: float = 0.05) -> bool:
        """Evaluate with tolerance (5% by default)."""
        if ground_truth == 0:
            return abs(predicted) < tolerance
        return abs((predicted - ground_truth) / ground_truth) < tolerance
    
    def evaluate_with_llm_judge(self, responses: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Evaluate response quality using our Custom LLM-as-a-judge."""
        print(f"\n🔍 DEBUG: LLM Judge available: {self.llm_judge is not None}")
        
        if not self.llm_judge:
            print("\n⚠️  Skipping LLM-as-a-judge evaluation (not available)")
            return {}
        
        print("\n⚖️ STEP 6: Custom LLM-as-a-Judge Evaluation")
        print("="*60)
        
        # Get the actual system prompt from our Custom LLM Judge
        judge_system_prompt = self.llm_judge.system_prompt
        
        # Save the judge system prompt
        judge_prompt_file = self.output_dir / "judge_prompts" / "llm_judge_system_prompt.txt"
        with open(judge_prompt_file, 'w') as f:
            f.write(judge_system_prompt)
        
        # Save judge metadata
        judge_metadata = {
            "system_prompt": judge_system_prompt,
            "model": self.model,
            "temperature": 0.1,  # Our custom judge uses 0.1
            "evaluation_criteria": [
                "Accuracy: Is the final numerical answer correct?",
                "Methodology: Is the calculation approach correct?",
                "Reasoning: Is the step-by-step reasoning clear and logical?",
                "Value Extraction: Are the correct values extracted from the patient note?",
                "Clinical Appropriateness: Is the response clinically sound?"
            ],
            "input_format_example": """
**Patient Note:** {patient_note}

**Question:** {question}

**AI Response:** {ai_response}

**Ground Truth Answer:** {ground_truth_answer}

**Ground Truth Explanation:** {ground_truth_explanation}

Please evaluate this AI response and provide your assessment in the specified JSON format.""",
            "output_format": "JSON format with label (1/0), accuracy, methodology, reasoning, clinical, and reason fields",
            "judge_type": "Custom LLM Judge",
            "timestamp": datetime.now().isoformat()
        }
        
        judge_metadata_file = self.output_dir / "judge_prompts" / "judge_metadata.json"
        with open(judge_metadata_file, 'w') as f:
            json.dump(judge_metadata, f, indent=2)
        
        print(f"💾 Saved judge system prompt to: {judge_prompt_file}")
        print(f"💾 Saved judge metadata to: {judge_metadata_file}")
        
        judge_results = {}
        sample_interactions = []  # To save examples of judge interactions
        
        for prompt_type, response_list in responses.items():
            print(f"\n   🔍 Judging {prompt_type}...")
            
            evaluations = []
            
            # Sample for evaluation (to manage costs)
            sample_size = min(self.llm_judge_sample_size, len(response_list))  # Evaluate up to 50 per technique
            sampled_responses = random.sample(response_list, sample_size)
            
            for i, response_data in enumerate(sampled_responses):
                try:
                    # Use our custom LLM judge for evaluation
                    label, reason = self.llm_judge.evaluate_medical_response(
                        patient_note=response_data['patient_note'],
                        question=response_data['question'],
                        ai_response=response_data['response'],
                        ground_truth_answer=response_data['ground_truth_answer'],
                        ground_truth_explanation=response_data.get('ground_truth_explanation', '')
                    )
                    
                    evaluation_entry = {
                        **response_data,
                        "llm_judge_label": label if label is not None else 0,
                        "llm_judge_reason": reason if reason is not None else "Evaluation failed",
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    evaluations.append(evaluation_entry)
                    
                    # Save first few interactions as examples
                    if len(sample_interactions) < 5:  # Save first 5 interactions across all techniques
                        sample_interactions.append({
                            "technique": prompt_type,
                            "patient_note": response_data['patient_note'][:500] + "...",
                            "question": response_data['question'],
                            "ai_response": response_data['response'][:500] + "...",
                            "ground_truth": response_data['ground_truth_answer'],
                            "judge_label": label,
                            "judge_reason": reason,
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    if (i + 1) % 10 == 0:
                        print(f"      ✅ Evaluated {i + 1}/{len(sampled_responses)}")
                    
                except Exception as e:
                    print(f"      ❌ Judge evaluation error: {str(e)}")
                    # Add failed evaluation entry
                    evaluation_entry = {
                        **response_data,
                        "llm_judge_label": 0,
                        "llm_judge_reason": f"Evaluation failed: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }
                    evaluations.append(evaluation_entry)
                    continue
            
            judge_results[prompt_type] = evaluations
            
            if evaluations:
                pass_rate = sum(1 for e in evaluations if e["llm_judge_label"] == 1) / len(evaluations)
                print(f"      ✅ Judge pass rate: {pass_rate:.1%} ({len(evaluations)} evaluated)")
        
        # Save judge results
        judge_file = self.output_dir / "evaluations" / "llm_judge_results.json"
        with open(judge_file, 'w') as f:
            json.dump(judge_results, f, indent=2)
        print(f"      💾 Saved judge results to: {judge_file}")
        print(f"      📊 Judge results summary: {len(judge_results)} prompt types evaluated")
        
        # Save sample judge interactions
        if sample_interactions:
            interactions_file = self.output_dir / "judge_prompts" / "sample_interactions.json"
            with open(interactions_file, 'w') as f:
                json.dump(sample_interactions, f, indent=2)
            print(f"      💾 Saved {len(sample_interactions)} sample judge interactions to: {interactions_file}")
        else:
            print(f"      ⚠️  No sample interactions to save")
        
        return judge_results
    
    def create_visualizations(self, accuracy_results: Dict[str, Dict[str, float]], 
                            judge_results: Dict[str, Any] = None) -> None:
        """Create comprehensive visualizations of results."""
        print("\n STEP 7: Creating Visualizations")
        print("="*60)
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Overall Accuracy Comparison
        self._plot_overall_accuracy(accuracy_results)
        
        # 2. Category-wise Accuracy
        self._plot_category_accuracy(accuracy_results)
        
        # 3. Prompt Type Comparison
        self._plot_prompt_comparison(accuracy_results)
        
        # 4. LLM Judge Results (if available)
        if judge_results:
            self._plot_judge_results(judge_results)
        
        # 5. Statistical Significance Tests
        self._plot_statistical_tests(accuracy_results)
        
        print("✅ All visualizations saved to visualizations/ directory")
    
    def _plot_overall_accuracy(self, accuracy_results: Dict[str, Dict[str, float]]) -> None:
        """Plot overall accuracy comparison."""
        prompt_types = list(accuracy_results.keys())
        accuracies = [accuracy_results[pt]['overall_accuracy'] for pt in prompt_types]
        
        # Clean up prompt type names for display
        display_names = [self._clean_prompt_name(pt) for pt in prompt_types]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(display_names, accuracies, alpha=0.8)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Overall Accuracy Comparison: Original vs PromptEngineer Prompts', 
                 fontsize=16, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Prompt Type', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, max(accuracies) * 1.2)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / "visualizations" / "overall_accuracy.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_category_accuracy(self, accuracy_results: Dict[str, Dict[str, float]]) -> None:
        """Plot category-wise accuracy comparison."""
        # Collect all categories
        all_categories = set()
        for results in accuracy_results.values():
            all_categories.update(results['category_accuracies'].keys())
        
        categories = sorted(list(all_categories))
        prompt_types = list(accuracy_results.keys())
        
        # Prepare data for heatmap
        heatmap_data = []
        for prompt_type in prompt_types:
            row = []
            for category in categories:
                acc = accuracy_results[prompt_type]['category_accuracies'].get(category, 0)
                row.append(acc)
            heatmap_data.append(row)
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(heatmap_data, 
                   xticklabels=categories,
                   yticklabels=[self._clean_prompt_name(pt) for pt in prompt_types],
                   annot=True, 
                   fmt='.2f',
                   cmap='RdYlBu_r',
                   cbar_kws={'label': 'Accuracy'})
        
        plt.title('Category-wise Accuracy Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Medical Calculation Categories', fontsize=12)
        plt.ylabel('Prompt Types', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / "visualizations" / "category_accuracy_heatmap.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prompt_comparison(self, accuracy_results: Dict[str, Dict[str, float]]) -> None:
        """Plot comparison between original and enhanced prompts."""
        # Separate original and enhanced results
        original_results = {k: v for k, v in accuracy_results.items() if k.startswith('original_')}
        enhanced_results = {k: v for k, v in accuracy_results.items() if k.startswith('enhanced_')}
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Original prompts
        orig_names = [k.replace('original_', '') for k in original_results.keys()]
        orig_accs = [v['overall_accuracy'] for v in original_results.values()]
        
        bars1 = ax1.bar(orig_names, orig_accs, alpha=0.8, color='skyblue')
        ax1.set_title('Original MedCalc Prompts', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_ylim(0, 1)
        
        for bar, acc in zip(bars1, orig_accs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Enhanced prompts
        enh_names = [k.replace('enhanced_', '').replace('_', ' ').title() for k in enhanced_results.keys()]
        enh_accs = [v['overall_accuracy'] for v in enhanced_results.values()]
        
        bars2 = ax2.bar(enh_names, enh_accs, alpha=0.8, color='lightcoral')
        ax2.set_title('PromptEngineer Enhanced Prompts', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_ylim(0, 1)
        
        for bar, acc in zip(bars2, enh_accs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "visualizations" / "prompt_type_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_judge_results(self, judge_results: Dict[str, Any]) -> None:
        """Plot LLM-as-a-judge evaluation results."""
        prompt_types = list(judge_results.keys())
        
        # Calculate pass rates
        pass_rates = {}
        for prompt_type, evaluations in judge_results.items():
            if evaluations:
                pass_rate = sum(1 for e in evaluations if e["llm_judge_label"] == 1) / len(evaluations)
                pass_rates[prompt_type] = pass_rate
        
        if not pass_rates:
            return
        
        # Create pass rate comparison chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        prompt_names = [self._clean_prompt_name(pt) for pt in prompt_types]
        rates = [pass_rates[pt] for pt in prompt_types]
        
        bars = ax.bar(prompt_names, rates, alpha=0.8, color='skyblue')
        
        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Prompt Types', fontsize=12)
        ax.set_ylabel('Pass Rate', fontsize=12)
        ax.set_title('LLM-as-a-Judge Pass Rates by Prompt Type', fontsize=16, fontweight='bold')
        ax.set_ylim(0, max(rates) * 1.2 if rates else 1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / "visualizations" / "llm_judge_results.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_tests(self, accuracy_results: Dict[str, Dict[str, float]]) -> None:
        """Plot statistical significance tests."""
        # Compare original vs enhanced approaches
        original_accs = [v['overall_accuracy'] for k, v in accuracy_results.items() if k.startswith('original_')]
        enhanced_accs = [v['overall_accuracy'] for k, v in accuracy_results.items() if k.startswith('enhanced_')]
        
        if len(original_accs) > 1 and len(enhanced_accs) > 1:
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(original_accs, enhanced_accs)
            
            plt.figure(figsize=(10, 6))
            
            # Box plot comparison
            data = [original_accs, enhanced_accs]
            labels = ['Original MedCalc', 'PromptEngineer Enhanced']
            
            plt.boxplot(data, labels=labels, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
            
            plt.ylabel('Accuracy', fontsize=12)
            plt.title(f'Statistical Comparison\nt-statistic: {t_stat:.3f}, p-value: {p_value:.3f}', 
                     fontsize=14, fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
            
            # Add significance annotation
            if p_value < 0.05:
                plt.text(0.5, max(max(original_accs), max(enhanced_accs)) * 1.1,
                        f'Statistically Significant (p < 0.05)', 
                        ha='center', va='center', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "visualizations" / "statistical_comparison.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _clean_prompt_name(self, prompt_type: str) -> str:
        """Clean prompt type names for display."""
        if prompt_type.startswith('original_'):
            return prompt_type.replace('original_', 'Original ').replace('_', ' ').title()
        elif prompt_type.startswith('enhanced_'):
            return prompt_type.replace('enhanced_', 'Enhanced ').replace('_', ' ').title()
        return prompt_type.replace('_', ' ').title()
    
    def generate_report(self, accuracy_results: Dict[str, Dict[str, float]], 
                       judge_results: Dict[str, Any] = None) -> None:
        """Generate comprehensive evaluation report."""
        print("\n STEP 8: Generating Report")
        print("="*60)
        
        report_content = self._create_detailed_report(accuracy_results, judge_results)
        
        # Save as text report
        report_file = self.output_dir / "reports" / "evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        # Save as JSON summary
        summary_data = self._create_summary_data(accuracy_results, judge_results)
        summary_file = self.output_dir / "reports" / "evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"✅ Report saved to: {report_file}")
        print(f"✅ Summary saved to: {summary_file}")
    
    def _create_detailed_report(self, accuracy_results: Dict[str, Dict[str, float]], 
                              judge_results: Dict[str, Any] = None) -> str:
        """Create detailed text report."""
        report = f"""
MedCalc-Bench Prompt Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

OVERVIEW
--------
This report compares the performance of original MedCalc-Bench prompts against 
PromptEngineer-generated enhanced prompts on medical calculation tasks.

EVALUATION METRICS
-----------------
1. Numerical Accuracy: Exact match with ground truth answers (with tolerance)
2. LLM-as-a-Judge: Qualitative evaluation of response quality

RESULTS SUMMARY
--------------
"""
        
        # Overall accuracy results
        report += "\nNUMERICAL ACCURACY RESULTS:\n"
        report += "-" * 30 + "\n"
        
        for prompt_type, results in accuracy_results.items():
            clean_name = self._clean_prompt_name(prompt_type)
            accuracy = results['overall_accuracy']
            correct = results['correct']
            total = results['total']
            
            report += f"{clean_name:30}: {accuracy:6.1%} ({correct:3d}/{total:3d})\n"
        
        # Best performing approaches
        best_original = max((k, v) for k, v in accuracy_results.items() if k.startswith('original_'))
        best_enhanced = max((k, v) for k, v in accuracy_results.items() if k.startswith('enhanced_'))
        
        report += f"\nBEST PERFORMERS:\n"
        report += f"Original: {self._clean_prompt_name(best_original[0])} - {best_original[1]['overall_accuracy']:.1%}\n"
        report += f"Enhanced: {self._clean_prompt_name(best_enhanced[0])} - {best_enhanced[1]['overall_accuracy']:.1%}\n"
        
        # Category breakdown
        report += f"\nCATEGORY-WISE PERFORMANCE:\n"
        report += "-" * 30 + "\n"
        
        # Get all categories
        all_categories = set()
        for results in accuracy_results.values():
            all_categories.update(results['category_accuracies'].keys())
        
        for category in sorted(all_categories):
            report += f"\n{category}:\n"
            for prompt_type, results in accuracy_results.items():
                acc = results['category_accuracies'].get(category, 0)
                clean_name = self._clean_prompt_name(prompt_type)
                report += f"  {clean_name:25}: {acc:6.1%}\n"
        
        # LLM Judge results
        if judge_results:
            report += f"\nLLM-AS-A-JUDGE RESULTS:\n"
            report += "-" * 30 + "\n"
            
            for prompt_type, evaluations in judge_results.items():
                if evaluations:
                    pass_rate = sum(1 for e in evaluations if e["llm_judge_label"] == 1) / len(evaluations)
                    clean_name = self._clean_prompt_name(prompt_type)
                    report += f"{clean_name:30}: {pass_rate:6.1%} pass rate ({len(evaluations)} evaluated)\n"
        
        # Statistical analysis
        original_accs = [v['overall_accuracy'] for k, v in accuracy_results.items() if k.startswith('original_')]
        enhanced_accs = [v['overall_accuracy'] for k, v in accuracy_results.items() if k.startswith('enhanced_')]
        
        if original_accs and enhanced_accs:
            report += f"\nSTATISTICAL ANALYSIS:\n"
            report += "-" * 30 + "\n"
            report += f"Original approaches - Mean: {np.mean(original_accs):.1%}, Std: {np.std(original_accs):.1%}\n"
            report += f"Enhanced approaches - Mean: {np.mean(enhanced_accs):.1%}, Std: {np.std(enhanced_accs):.1%}\n"
            
            if len(original_accs) > 1 and len(enhanced_accs) > 1:
                t_stat, p_value = stats.ttest_ind(original_accs, enhanced_accs)
                report += f"T-test results: t={t_stat:.3f}, p={p_value:.3f}\n"
                if p_value < 0.05:
                    report += "Result: Statistically significant difference (p < 0.05)\n"
                else:
                    report += "Result: No statistically significant difference\n"
        
        # Conclusions
        report += f"\nCONCLUSIONS:\n"
        report += "-" * 30 + "\n"
        
        improvement = best_enhanced[1]['overall_accuracy'] - best_original[1]['overall_accuracy']
        if improvement > 0:
            report += f"• PromptEngineer enhanced prompts showed improvement of {improvement:.1%}\n"
        else:
            report += f"• Original prompts performed better by {-improvement:.1%}\n"
        
        report += f"• Best overall approach: {self._clean_prompt_name(max(accuracy_results.items(), key=lambda x: x[1]['overall_accuracy'])[0])}\n"
        
        return report
    
    def _create_summary_data(self, accuracy_results: Dict[str, Dict[str, float]], 
                           judge_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create summary data for JSON export."""
        summary = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "accuracy_results": accuracy_results,
            "best_performers": {
                "overall": max(accuracy_results.items(), key=lambda x: x[1]['overall_accuracy']),
                "original": max((k, v) for k, v in accuracy_results.items() if k.startswith('original_')),
                "enhanced": max((k, v) for k, v in accuracy_results.items() if k.startswith('enhanced_'))
            }
        }
        
        if judge_results:
            summary["judge_results"] = judge_results
        
        return summary
    
    def get_data_for_demonstrations_from_test_data(self, calculator_id):
        """
        Fetches data from test_data.csv for a given calculator_id and returns JSON format.
        
        Args:
            calculator_id (int): The Calculator ID to filter by
            
        Returns:
            list: List of dictionaries matching the specified JSON format
        """
        csv_path = Path(__file__).parent / "MedCalc-Bench" / "dataset" / "train_data.csv"
        df = pd.read_csv(csv_path)
        
        # Filter by calculator_id
        filtered_df = df[df['Calculator ID'] == calculator_id]
        
        result = []
        for _, row in filtered_df.iterrows():
            item = {
                "id": int(row['Row Number']),
                "patient_note": row['Patient Note'],
                "question": row['Question'],
                "ground_truth_answer": int(row['Ground Truth Answer']) if str(row['Ground Truth Answer']).isdigit() else row['Ground Truth Answer'],
                "ground_truth_explanation": row['Ground Truth Explanation']
            }
            result.append(item)
        
        return result
    
    def get_enhanced_demonstration_prompt(self, enhanced_prompts, df):
        """
        Process enhanced prompts and run demonstration pipeline for one-shot techniques.
        Uses caching to avoid re-running expensive demonstration generation.
        
        Args:
            enhanced_prompts: Dictionary of enhanced prompts from all_enhanced_prompts.json
            df: DataFrame containing test data
            
        Returns:
            Dict: Results from demonstration pipeline for each processed prompt technique
        """
        # Target techniques for demonstration pipeline
        target_techniques = ['chain_of_thought_one_shot', 'chain_of_draft_one_shot']
        
        print("\n🎭 GENERATING ENHANCED DEMONSTRATIONS")
        print("="*60)
        
        # Initialize cache file
        cache_file = Path("demonstrations") / "demonstration_cache.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing cache
        demonstration_cache = self._load_demonstration_cache(cache_file)
        
        all_demonstrations = {}
        cache_updated = False
        
        # Process enhanced prompts (outer loop - techniques)
        for technique, prompt_data in enhanced_prompts.items():
            # Skip if not a target technique
            if technique not in target_techniques:
                print(f"⏭️  Skipping {technique} (not a one-shot technique)")
                continue
                
            print(f"\n🔄 Processing technique: {technique}")
            print("-"*50)
            
            technique_results = []
            
            # Process each row in dataframe (inner loop - examples)
            for idx, row in df.iterrows():
                calculator_id = str(row['Calculator ID'])
                calculator_name = row['Calculator Name']
                
                try:
                    print(f"   📊 Calculator ID {calculator_id}: {calculator_name}")
                    
                    # Check cache first
                    cache_key = f"{technique}_{calculator_id}"
                    if cache_key in demonstration_cache:
                        print(f"      💾 Loading from cache...")
                        cached_result = demonstration_cache[cache_key]
                        
                        demonstration_data = {
                            "row_number": row['Row Number'],
                            "calculator_id": int(calculator_id),
                            "calculator_name": calculator_name,
                            "category": row['Category'],
                            "technique": technique,
                            "final_prompt": cached_result.get('final_prompt', ''),
                            "pipeline_results": cached_result.get('pipeline_results', {}),
                            "source": "cache",
                            "timestamp": cached_result.get('timestamp', datetime.now().isoformat())
                        }
                        
                        technique_results.append(demonstration_data)
                        print(f"      ✅ Loaded from cache")
                        continue
                    
                    # Not in cache - need to generate
                    print(f"      🚀 Generating new demonstrations...")
                    
                    # Get examples from test data for this Calculator ID
                    examples = self.get_data_for_demonstrations_from_test_data(int(calculator_id))
                    
                    if not examples:
                        print(f"      ⚠️  No examples found for Calculator ID {calculator_id}")
                        continue
                        
                    print(f"      📋 Found {len(examples)} examples for demonstrations")
                    
                    # Extract the prompt text
                    initial_prompt = prompt_data.get('prompt', '')
                    
                    if not initial_prompt:
                        print(f"      ⚠️  Warning: No prompt found for {technique}")
                        continue
                    
                    # Run demonstration pipeline
                    print(f"      ⚙️  Running demonstration pipeline...")
                    pipeline_results = self.demo_pipeline.run_complete_pipeline_medcalc(
                        initial_prompt=initial_prompt,
                        examples=examples,
                        n_demonstrations=3,
                        max_iterations=1
                    )
                    
                    # Extract final prompt from pipeline results
                    final_prompt = pipeline_results.get('final_prompt', '')
                    if not final_prompt and 'pipeline_summary' in pipeline_results:
                        final_prompt = pipeline_results['pipeline_summary'].get('final_prompt', '')
                    
                    # Store in cache
                    demonstration_cache[cache_key] = {
                        'final_prompt': final_prompt,
                        'pipeline_results': pipeline_results,
                        'calculator_id': int(calculator_id),
                        'technique': technique,
                        'timestamp': datetime.now().isoformat()
                    }
                    cache_updated = True
                    
                    # Create response data
                    demonstration_data = {
                        "row_number": row['Row Number'],
                        "calculator_id": int(calculator_id),
                        "calculator_name": calculator_name,
                        "category": row['Category'],
                        "technique": technique,
                        "final_prompt": final_prompt,
                        "pipeline_results": pipeline_results,
                        "source": "generated",
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    technique_results.append(demonstration_data)
                    
                    best_score = pipeline_results.get('pipeline_summary', {}).get('best_score', 0)
                    print(f"      ✅ Generated - Best Score: {best_score:.1%}")
                    
                    # Save cache periodically
                    if len(technique_results) % 5 == 0:
                        print(f"      💾 Saving cache...")
                        self._save_demonstration_cache(demonstration_cache, cache_file)
                    
                except Exception as e:
                    print(f"      ❌ Error for Calculator ID {calculator_id}: {str(e)}")
                    
                    # Store error in results
                    error_data = {
                        "row_number": row['Row Number'],
                        "calculator_id": int(calculator_id),
                        "calculator_name": calculator_name,
                        "category": row['Category'],
                        "technique": technique,
                        "error": str(e),
                        "source": "error",
                        "timestamp": datetime.now().isoformat()
                    }
                    technique_results.append(error_data)
                    continue
            
            all_demonstrations[f"enhanced_{technique}"] = technique_results
            print(f"   ✅ Completed {technique}: {len(technique_results)} demonstrations")
        
        # Save final cache
        if cache_updated:
            print(f"\n💾 Saving demonstration cache...")
            self._save_demonstration_cache(demonstration_cache, cache_file)
        
        # Save all demonstrations
        # demonstrations_file = self.output_dir / "demonstrations" / "all_demonstrations.json"
        # with open(demonstrations_file, 'w') as f:
        #     json.dump(all_demonstrations, f, indent=2)
        
        # Print summary
        total_generated = sum(len(r) for r in all_demonstrations.values())
        from_cache = sum(1 for technique_results in all_demonstrations.values() 
                        for result in technique_results if result.get('source') == 'cache')
        newly_generated = sum(1 for technique_results in all_demonstrations.values() 
                             for result in technique_results if result.get('source') == 'generated')
        errors = sum(1 for technique_results in all_demonstrations.values() 
                    for result in technique_results if result.get('source') == 'error')
        
        print(f"\n🎉 DEMONSTRATION GENERATION COMPLETED")
        print(f"   Total demonstrations: {total_generated}")
        print(f"   From cache: {from_cache}")
        print(f"   Newly generated: {newly_generated}")
        print(f"   Errors: {errors}")
        
        return all_demonstrations
    
    def _load_demonstration_cache(self, cache_file: Path) -> Dict[str, Any]:
        """Load demonstration cache from file."""
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
                print(f"   📂 Loaded cache with {len(cache)} entries")
                return cache
            except Exception as e:
                print(f"   ⚠️  Could not load cache: {e}")
                return {}
        else:
            print(f"   📂 No existing cache found, creating new cache")
            return {}
    
    def _save_demonstration_cache(self, cache: Dict[str, Any], cache_file: Path) -> None:
        """Save demonstration cache to file."""
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            print(f"   💾 Cache saved with {len(cache)} entries")
        except Exception as e:
            print(f"   ❌ Could not save cache: {e}")

    
    def run_complete_evaluation(self, sample_size: int = None, max_responses: int = None, budget_limit: float = None, calculator_id: int = None) -> Dict[str, Any]:
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
                # Reserve some calls for judge evaluation
                judge_calls = total_prompts * 25  # Reduced judge sample
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
        if calculator_id is not None:
            df = self.load_medcalc_data_by_calculator(calculator_id, sample_size)
            if df.empty:
                print(f"❌ No data found for Calculator ID {calculator_id}. Exiting.")
                return {"error": f"No data found for Calculator ID {calculator_id}"}
        else:
            df = self.load_medcalc_data(sample_size)
        
        # Step 2: Generate PromptEngineer prompts (both zero-shot and one-shot enhanced)
        print("\n🚀 STEP 2: Generating PromptEngineer Prompts")
        print("="*60)
        
        context_zero_shot = self.create_medcalc_context()
        enhanced_prompts_zero_shot = self.generate_promptengineer_prompts(context_zero_shot, "Zero-Shot Enhanced")
        
        context_one_shot = self.create_medcalc_one_shot_context()
        enhanced_prompts_one_shot = self.generate_promptengineer_prompts(context_one_shot, "One-Shot Enhanced")
        
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

        demonstration_results = self.get_enhanced_demonstration_prompt(enhanced_prompts, df)
        print(f"   • One-shot demonstrations generated and cached")
        
        # Step 3: Get original prompts
        original_prompts = self.get_original_medcalc_prompts()
        
        # Step 4: Generate responses
        responses = self.generate_responses(df, enhanced_prompts, original_prompts, max_responses)
        
        # Step 5: Evaluate accuracy
        accuracy_results = self.evaluate_accuracy(responses)
        
        # Step 6: LLM judge evaluation
        judge_results = self.evaluate_with_llm_judge(responses)
        
        # Step 7: Create visualizations
        # self.create_visualizations(accuracy_results, judge_results)
        
        # Step 8: Generate report
        self.generate_report(accuracy_results, judge_results)
        
        # Final summary
        print("\n" + "="*100)
        print("✅ EVALUATION COMPLETE")
        print("="*100)
        
        print(f"\n Key Results:")
        best_overall = max(accuracy_results.items(), key=lambda x: x[1]['overall_accuracy'])
        print(f"   • Best overall: {self._clean_prompt_name(best_overall[0])} - {best_overall[1]['overall_accuracy']:.1%}")
        
        original_results = {k: v for k, v in accuracy_results.items() if k.startswith('original_')}
        enhanced_results = {k: v for k, v in accuracy_results.items() if k.startswith('enhanced_')}
        
        if original_results:
            best_original = max(original_results.items(), key=lambda x: x[1]['overall_accuracy'])
            print(f"   • Best original: {self._clean_prompt_name(best_original[0])} - {best_original[1]['overall_accuracy']:.1%}")
        
        if enhanced_results:
            best_enhanced = max(enhanced_results.items(), key=lambda x: x[1]['overall_accuracy'])
            print(f"   • Best enhanced: {self._clean_prompt_name(best_enhanced[0])} - {best_enhanced[1]['overall_accuracy']:.1%}")
        
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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MedCalc-Bench Prompt Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python medcalc_prompt_evaluation_pipeline.py --sample-size 20 --max-responses 5
  python medcalc_prompt_evaluation_pipeline.py --sample-size 10 --max-responses 3 --budget-limit 5.0
  python medcalc_prompt_evaluation_pipeline.py --sample-size 50 --llm-judge-sample-size 30 --output-dir my_experiment
        """
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=20,
        help='Number of MedCalc examples to evaluate (default: 20)'
    )
    
    parser.add_argument(
        '--max-responses',
        type=int,
        default=5,
        help='Maximum responses to generate per technique (default: 5)'
    )
    
    parser.add_argument(
        '--llm-judge-sample-size',
        type=int,
        default=20,
        dest='llm_judge_sample_size',  # Use underscores for the attribute name
        help='Maximum responses to evaluate per technique with LLM judge (default: 20)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: auto-generated with timestamp)'
    )
    
    parser.add_argument(
        '--skip-judge',
        action='store_true',
        help='Skip LLM-as-a-judge evaluation to save costs'
    )
    
    parser.add_argument(
        '--budget-limit',
        type=float,
        default=10.0,
        help='Maximum budget in USD (default: $10.00)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Check API key
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your-api-key-here":
        print("❌ API key not configured properly")
        print("   Please set your OpenAI API key environment variable")
        sys.exit(1)
    
    print("✅ API key configured")
    
    # Initialize and run pipeline
    pipeline = MedCalcEvaluationPipeline(
        api_key=OPENAI_API_KEY,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        max_responses=args.max_responses,
        llm_judge_sample_size=args.llm_judge_sample_size,
        budget_limit=args.budget_limit
    )
    
    # Temporarily disable judge if requested
    if args.skip_judge:
        pipeline.llm_judge = None
        print("⚠️  LLM-as-a-judge evaluation disabled per user request")
    
    try:
        results = pipeline.run_complete_evaluation()
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📁 Results saved to: {results['output_directory']}")
        
    except KeyboardInterrupt:
        print("\n\n⚡ Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc() 