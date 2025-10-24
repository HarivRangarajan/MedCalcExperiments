#!/usr/bin/env python3
"""
PubMed PQAL Contrastive Boosted Edits Evaluation Pipeline

Usage:
  python pubmed_with_contrastive_boosted_edits.py --sample-size 1000 --output-dir results_pubmed
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "promptengineer"))

# Import shared components
from promptengineer import PromptPipeline
from promptengineer.techniques.base import PromptContext


class PubMedContrastiveEvaluationPipeline:
    """Complete evaluation pipeline for PubMed PQAL with prompt engineering comparison."""
    
    # Configuration constants
    DEFAULT_SAMPLE_SIZE = 10  # Default number of examples to sample from dataset
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
            # project_root = Path(__file__).parent.parent
            # output_dir = project_root / "outputs" / f"pubmed_contrastive_edits_evaluation_{timestamp}"
            output_dir = Path(__file__).parent / "outputs" / f"pubmed_contrastive_edits_evaluation_{timestamp}"
        self.output_dir = Path(output_dir)
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ["data", "prompts", "responses", "evaluations", "correct", "incorrect"]:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        # Initialize components
        self.prompt_pipeline = PromptPipeline(api_key=api_key, output_dir=str(self.output_dir))
        
        # Use custom OpenAI wrapper
        self.llm = self._create_openai_wrapper()
        
        # Load the 4 hand-picked few-shot examples
        self.few_shot_examples = self._load_hand_picked_examples()
        
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
    
    def _load_hand_picked_examples(self) -> List[Dict]:
        """Load the 4 hand-picked few-shot examples."""
        try:
            few_shot_file = Path(__file__).parent / "data" / "few_shot_examples.json"
            if not few_shot_file.exists():
                raise FileNotFoundError(f"Few-shot examples file not found: {few_shot_file}")
            
            with open(few_shot_file, 'r') as f:
                data = json.load(f)
            
            # Convert to list format for easier use
            examples = []
            for pmid, entry in data.items():
                examples.append({
                    "pmid": entry["PMID"],
                    "context": entry["CONTEXTS"],
                    "question": entry["QUESTION"],
                    "answer": entry["LONG_ANSWER"],
                    "decision": entry["FINAL_DECISION"],
                    "category": entry["CATEGORY"]
                })
            
            print(f"✅ Loaded {len(examples)} hand-picked few-shot examples")
            return examples
            
        except Exception as e:
            print(f"❌ Error loading hand-picked examples: {e}")
            sys.exit(1)
    
    def load_pubmed_data(self, sample_size: int = None) -> pd.DataFrame:
        """Load and sample PubMed PQAL dataset."""
        if sample_size is None:
            sample_size = self.sample_size
        
        print(f"\n📋 STEP 1: Loading PubMed PQAL Data (Sample: {sample_size})")
        print("="*60)
        
        # Load JSON data
        data_path = Path(__file__).parent / "data" / "pqal_train.json"
        if not data_path.exists():
            raise FileNotFoundError(f"PubMed data file not found: {data_path}")
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Loaded {len(data)} total examples from PubMed PQAL dataset")
        
        # Convert to DataFrame
        rows = []
        for pmid, entry in data.items():
            rows.append({
                "PMID": entry["PMID"],
                "Question": entry["QUESTION"],
                "Context": entry["CONTEXTS"],
                "Long_Answer": entry["LONG_ANSWER"],
                "Final_Decision": entry["FINAL_DECISION"],
                "Category": entry["CATEGORY"],
                "Labels": entry["LABELS"],
                "Meshes": entry["MESHES"],
                "Year": entry["YEAR"]
            })
        
        df = pd.DataFrame(rows)
        
        # Basic statistics
        print(f"   • Categories: {df['Category'].unique()}")
        print(f"   • Decision distribution: {df['Final_Decision'].value_counts().to_dict()}")
        
        # Simple random sampling
        actual_sample_size = min(sample_size, len(df))
        
        if actual_sample_size < len(df):
            # Random sample of requested size
            sampled_df = df.sample(n=actual_sample_size, random_state=42).reset_index(drop=True)
            print(f"   • Random sampling: {actual_sample_size} examples")
        else:
            # Use all data if requested sample size >= total data
            sampled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
            print(f"   • Using all {len(df)} examples (requested sample size >= total)")
        
        # Show warning if requested more than available
        if sample_size > len(df):
            print(f"   ⚠️  Warning: Requested {sample_size} examples, but dataset only has {len(df)}")
        
        # Save sampled data
        sample_file = self.output_dir / "data" / "sampled_pubmed_data.csv"
        sampled_df.to_csv(sample_file, index=False)
        
        # Category distribution in sample
        category_dist = sampled_df['Category'].value_counts()
        print(f"\n   Sample distribution by category:")
        for cat, count in category_dist.items():
            print(f"      • {cat}: {count} ({count/len(sampled_df)*100:.1f}%)")
        
        # Decision distribution in sample
        decision_dist = sampled_df['Final_Decision'].value_counts()
        print(f"\n   Sample distribution by decision:")
        for decision, count in decision_dist.items():
            print(f"      • {decision}: {count} ({count/len(sampled_df)*100:.1f}%)")
        
        return sampled_df
    
    def create_original_prompt_with_examples(self, context: str, question: str) -> Tuple[str, str]:
        """Create original PubMed prompt with all 4 hand-picked examples."""
        system_msg = '''You are an expert medical researcher analyzing scientific literature. 
Your task is to answer questions about research studies based on the provided abstract/context.

You should:
1. Carefully read the context (research abstract)
2. Answer the specific question asked
3. Provide your reasoning based on the evidence
4. Give a final yes/no/maybe decision when applicable

Output format: {"reasoning": "step-by-step analysis", "answer": "final answer", "decision": "yes/no/maybe"}

Here are some examples:'''
        
        # Add all 4 hand-picked examples
        for i, example in enumerate(self.few_shot_examples, 1):
            system_msg += f'\n\nExample {i}:'
            system_msg += f'\nContext: {example["context"]}'
            system_msg += f'\nQuestion: {example["question"]}'
            system_msg += f'\nOutput: {{"reasoning": "{example["answer"]}", "decision": "{example["decision"]}"}}\n'
        
        user_msg = f'Context: {context}\n\nQuestion: {question}\n\nPlease provide your analysis in the specified JSON format:'
        
        return system_msg, user_msg
    
    def create_enhanced_pubmed_prompt(self, enhanced_base: str, context: str, 
                                    question: str) -> Tuple[str, str]:
        """Create enhanced PubMed prompt with the same 4 examples."""
        # Start with the enhanced prompt base
        system_msg = enhanced_base
        
        # Add domain-specific guidance
        system_msg += '''\n\nFor biomedical question answering:
- Focus on study design, sample size, statistical significance
- Consider clinical relevance and practical implications
- Evaluate evidence quality and potential biases
- Provide clear yes/no/maybe decisions when asked

Output format: {"reasoning": "step-by-step analysis", "answer": "final answer", "decision": "yes/no/maybe"}

Here are some examples:'''
        
        # Add the same 4 hand-picked examples
        for i, example in enumerate(self.few_shot_examples, 1):
            system_msg += f'\n\nExample {i}:'
            system_msg += f'\nContext: {example["context"]}'
            system_msg += f'\nQuestion: {example["question"]}'
            system_msg += f'\nOutput: {{"reasoning": "{example["answer"]}", "decision": "{example["decision"]}"}}\n'
        
        user_msg = f'Context: {context}\n\nQuestion: {question}\n\nPlease provide your analysis in the specified JSON format:'
        
        return system_msg, user_msg
    
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
    
    def check_pubmed_correctness(self, predicted_decision: str, ground_truth_decision: str) -> bool:
        """Check if PubMed prediction matches ground truth."""
        # Normalize both decisions
        pred_norm = predicted_decision.lower().strip()
        truth_norm = ground_truth_decision.lower().strip()
        
        # Direct match
        if pred_norm == truth_norm:
            return True
        
        # Handle synonyms
        yes_synonyms = ["yes", "positive", "supported", "effective", "significant"]
        no_synonyms = ["no", "negative", "not supported", "ineffective", "not significant"]
        maybe_synonyms = ["maybe", "unclear", "uncertain"]
        
        if (pred_norm in yes_synonyms and truth_norm in yes_synonyms):
            return True
        if (pred_norm in no_synonyms and truth_norm in no_synonyms):
            return True
        if (pred_norm in maybe_synonyms and truth_norm in maybe_synonyms):
            return True
        
        return False
    
    def create_enhanced_prompts(self, original_prompt_with_examples: str) -> Dict[str, str]:
        """
        Create enhanced versions of the original prompt (with examples) using PromptEngineer.
        Returns dict with 'chain_of_thought' and 'chain_of_draft' enhanced prompts.
        """
        print(f"\n🚀 STEP 2: Creating Enhanced Prompts")
        print("="*60)
        
        # Create a PromptContext from the original prompt with examples
        context = PromptContext(
            task_description=original_prompt_with_examples,
            domain="Biomedical question answering and literature analysis with few-shot examples"
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
    
    def save_outputs_as_json(self, data_list, output_file):
        """Save outputs as single JSON file containing a list of entries."""
        with open(output_file, 'w') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)

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
                
                context = row["Context"]
                question = row["Question"]
                pmid = row["PMID"]
                category = row["Category"]
                
                try:
                    # Create messages based on prompt type
                    if prompt_type == "original":
                        # Use original prompt with all 4 examples
                        system_msg, user_msg = self.create_original_prompt_with_examples(
                            context, question
                        )
                    else:
                        # Use enhanced prompt with all 4 examples
                        enhanced_prompt = enhanced_prompts[prompt_type]["prompt"]
                        system_msg, user_msg = self.create_enhanced_pubmed_prompt(
                            enhanced_prompt, context, question
                        )
                    
                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ]
                    
                    # Generate answer
                    response = self.llm.answer(messages)
                    
                    # Extract answer, reasoning, and decision
                    answer, reasoning, decision = self.extract_pubmed_answer(response)
                    
                    # Check correctness
                    correctness = self.check_pubmed_correctness(
                        decision, row["Final_Decision"]
                    )
                    
                    status = "Correct" if correctness else "Incorrect"
                    
                    # Store result
                    result = {
                        "PMID": pmid,
                        "Category": category,
                        "Question": question,
                        "Context": context,
                        "LLM_Answer": answer,
                        "LLM_Reasoning": reasoning,
                        "LLM_Decision": decision,
                        "Ground_Truth_Decision": row["Final_Decision"],
                        "Ground_Truth_Answer": row["Long_Answer"],
                        "Result": status,
                        "Prompt_Type": prompt_type,
                        "Year": row["Year"],
                        "Labels": row["Labels"],
                        "Meshes": row["Meshes"]
                    }
                    
                    all_responses[prompt_type].append(result)
                    
                except Exception as e:
                    print(f"   ⚠️  Error processing PMID {pmid} with {prompt_type}: {e}")
                    # Store error result
                    result = {
                        "PMID": pmid,
                        "Category": category,
                        "Question": question,
                        "Context": context,
                        "LLM_Answer": str(e),
                        "LLM_Reasoning": str(e),
                        "LLM_Decision": "error",
                        "Ground_Truth_Decision": row["Final_Decision"],
                        "Ground_Truth_Answer": row["Long_Answer"],
                        "Result": "Incorrect",
                        "Prompt_Type": prompt_type,
                        "Year": row["Year"],
                        "Labels": row["Labels"],
                        "Meshes": row["Meshes"]
                    }
                    all_responses[prompt_type].append(result)
            
            # Save responses for this prompt type
            for prompt_type, responses in all_responses.items():
                output_file = self.output_dir / "responses" / f"{prompt_type}_responses.json"
                self.save_outputs_as_json(responses, output_file)
                print(f"Saved {len(responses)} responses to {output_file}")
        
        return all_responses
    
    def analyze_category_performance(self, all_responses: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """Analyze performance by PubMed category."""
        category_analysis = {}
        
        for prompt_type, responses in all_responses.items():
            category_stats = {}
            
            # Group by category
            for response in responses:
                category = response["Category"]
                if category not in category_stats:
                    category_stats[category] = {"total": 0, "correct": 0}
                
                category_stats[category]["total"] += 1
                if response["Result"] == "Correct":
                    category_stats[category]["correct"] += 1
            
            # Calculate accuracy for each category
            for category in category_stats:
                stats = category_stats[category]
                stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            
            category_analysis[prompt_type] = category_stats
        
        return category_analysis
    
    def evaluate_and_split_responses(self, all_responses: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Evaluate accuracy for each prompt type and split into correct/incorrect files.
        """
        print(f"\n📊 STEP 4: Evaluating and Splitting Responses")
        print("="*60)
        
        evaluation_results = {}
        
        contrastive_dir = Path(__file__).parent / "data" / "contrastive"
        contrastive_dir.mkdir(parents=True, exist_ok=True)

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
            correct_file = Path(__file__).parent / "data" / "contrastive" / f"{prompt_type}_correct.json"
            self.save_outputs_as_json(correct_responses, correct_file)
            
            # Save incorrect responses
            incorrect_file = Path(__file__).parent / "data" / "contrastive" / f"{prompt_type}_incorrect.json"
            self.save_outputs_as_json(incorrect_responses, incorrect_file)
            
            print(f"      Saved {len(correct_responses)} correct responses to: {correct_file.name}")
            print(f"      Saved {len(incorrect_responses)} incorrect responses to: {incorrect_file.name}")
            
            evaluation_results[prompt_type] = {
                "accuracy": accuracy,
                "total": total,
                "correct": correct_count,
                "incorrect": len(incorrect_responses)
            }
        
        # Analyze category performance
        category_analysis = self.analyze_category_performance(all_responses)
        
        # Save evaluation summary
        summary_file = self.output_dir / "evaluations" / "evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "overall_results": evaluation_results,
                "category_analysis": category_analysis
            }, f, indent=2)
        
        return evaluation_results
    
    def run_complete_evaluation(self) -> Dict[str, Any]:
        """Run the complete evaluation pipeline."""
        print("="*100)
        print("PUBMED PQAL WITH CONTRASTIVE BOOSTED EDITS EVALUATION PIPELINE")
        print("="*100)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Sample size: {self.sample_size}")
        print(f"Model: {self.model}")
        print(f"Using {len(self.few_shot_examples)} hand-picked few-shot examples")
        
        # Step 1: Load data
        df = self.load_pubmed_data()
        
        # Step 2: Create the base original prompt with all examples
        # We'll create this for one example to get the structure for enhancement
        sample_context = "Sample research abstract for prompt structure"
        sample_question = "Sample research question?"
        original_prompt_with_examples, _ = self.create_original_prompt_with_examples(
            sample_context, sample_question
        )
        
        # Extract just the system message part for enhancement
        original_system_prompt = original_prompt_with_examples
        
        # Step 3: Create enhanced prompts based on the original with examples
        enhanced_prompts = self.create_enhanced_prompts(original_system_prompt)
        
        # Step 4: Generate responses
        all_responses = self.generate_responses(df, enhanced_prompts)
        
        # Step 5: Evaluate and split responses
        evaluation_results = self.evaluate_and_split_responses(all_responses)
        
        # Final summary
        print("\n" + "="*100)
        print("EVALUATION COMPLETE")
        print("="*100)
        
        print(f"\nKey Results:")
        for prompt_type, results in evaluation_results.items():
            print(f"   • {prompt_type}: {results['accuracy']:.2%} accuracy ({results['correct']}/{results['total']})")
        
        # Calculate improvements
        original_acc = evaluation_results["original"]["accuracy"]
        print(f"\nImprovements over original:")
        for prompt_type in ["chain_of_thought", "chain_of_draft"]:
            if prompt_type in evaluation_results:
                acc = evaluation_results[prompt_type]["accuracy"]
                improvement = acc - original_acc
                print(f"   • {prompt_type}: {improvement:+.1%} ({acc:.1%} vs {original_acc:.1%})")
        
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
        description="PubMed PQAL Contrastive Boosted Edits Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pubmed_with_contrastive_boosted_edits.py --sample-size 1000
  python pubmed_with_contrastive_boosted_edits.py --sample-size 500 --output-dir ./pubmed_results
        """
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Number of PubMed PQAL examples to evaluate (default: 1000)'
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
    pipeline = PubMedContrastiveEvaluationPipeline(
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