#!/usr/bin/env python3
"""
Shared Utilities for MedCalc Evaluation Pipelines

This module contains common functions used across multiple evaluation scripts
to avoid code duplication and ensure consistency.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Add MedCalc evaluation imports
sys.path.insert(0, str(Path(__file__).parent.parent / "MedCalc-Bench" / "evaluation"))
try:
    from evaluate import check_correctness
except ImportError as e:
    print(f"⚠️  MedCalc evaluation imports failed: {e}")
    check_correctness = None


def load_medcalc_one_shot_examples() -> Dict[str, Any]:
    """
    Load MedCalc's original one-shot examples for calculator-specific prompting.
    
    Returns:
        Dict mapping calculator IDs to their one-shot examples
    """
    try:
        one_shot_file = Path(__file__).parent.parent / "MedCalc-Bench" / "evaluation" / "one_shot_finalized_explanation.json"
        if one_shot_file.exists():
            with open(one_shot_file, 'r') as f:
                examples = json.load(f)
            return examples
        else:
            print(f"⚠️  One-shot examples file not found: {one_shot_file}")
            return {}
    except Exception as e:
        print(f"⚠️  Error loading one-shot examples: {e}")
        return {}


def extract_answer(answer: str, calid: int) -> str:
    """
    Extract answer from LLM response (same method as other scripts).
    
    Args:
        answer: Raw LLM response string
        calid: Calculator ID (unused but kept for compatibility)
        
    Returns:
        Extracted answer string or "Not Found"
    """
    extracted_answer = re.findall(r'[Aa]nswer":\s*(.*?)\}', answer)
    
    if len(extracted_answer) == 0:
        extracted_answer = "Not Found"
    else:
        extracted_answer = extracted_answer[-1].strip().strip('"')
        if extracted_answer in ["str(short_and_direct_answer_of_the_question)", 
                               "str(value which is the answer to the question)", "X.XX"]:
            extracted_answer = "Not Found"
    
    return extracted_answer


def create_one_shot_prompt(prompt: str, 
                           patient_note: str, 
                           question: str, 
                           one_shot_example: Dict[str, Any]) -> Tuple[str, str]:
    """
    Create a one-shot prompt by combining the base prompt with a one-shot example.
    
    This follows the pattern used in contrastive_demonstration_generation.py (lines 318-323).
    
    Args:
        prompt: Base prompt text
        patient_note: Patient note for the current query
        question: Question to answer
        one_shot_example: One-shot example dict with "Patient Note" and "Response" keys
        
    Returns:
        Tuple of (system_msg, user_msg)
    """
    # Add one-shot example to the prompt
    system_msg = prompt + f'\n\nHere is an example patient note:\n\n{one_shot_example["Patient Note"]}'
    system_msg += f'\n\nHere is an example task:\n\n{question}'
    system_msg += f'\n\nHere is the expected output:\n\n{json.dumps({"step_by_step_thinking": one_shot_example["Response"]["step_by_step_thinking"], "answer": one_shot_example["Response"]["answer"]})}'
    user_msg = f'Here is the patient note:\n\n{patient_note}\n\nHere is the task:\n\n{question}\n\nPlease directly output the JSON dict.'
    
    return system_msg, user_msg


def create_original_one_shot_prompt(patient_note: str,
                                    question: str,
                                    example_note: str,
                                    example_output: Dict) -> Tuple[str, str]:
    """
    Extract the exact one-shot prompt from MedCalc's run.py.
    
    This is the original MedCalc baseline prompt format.
    
    Args:
        patient_note: Patient note for the current query
        question: Question to answer
        example_note: Example patient note for one-shot
        example_output: Example output dict with "step_by_step_thinking" and "answer"
        
    Returns:
        Tuple of (system_msg, user_msg)
    """
    system_msg = 'You are a helpful assistant for calculating a score for a given patient note. Please think step-by-step to solve the question and then generate the required score. Your output should only contain a JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}}.'
    system_msg += f'Here is an example patient note:\n\n{example_note}'
    system_msg += f'\n\nHere is an example task:\n\n{question}'
    system_msg += f'\n\nPlease directly output the JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(value which is the answer to the question)}}:\n\n{json.dumps(example_output)}'
    user_msg = f'Here is the patient note:\n\n{patient_note}\n\nHere is the task:\n\n{question}\n\nPlease directly output the JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}}:'
    
    return system_msg, user_msg


def evaluate_answer(answer_value: str, 
                    ground_truth: str, 
                    calculator_id: int, 
                    upper_limit: str, 
                    lower_limit: str) -> bool:
    """
    Evaluate if an answer is correct using MedCalc's check_correctness function.
    
    Args:
        answer_value: Extracted answer from LLM
        ground_truth: Ground truth answer
        calculator_id: Calculator ID
        upper_limit: Upper limit for range-based calculators
        lower_limit: Lower limit for range-based calculators
        
    Returns:
        True if correct, False otherwise
    """
    if not check_correctness or answer_value == "Not Found":
        return False

    try:
        is_correct = check_correctness(
            answer_value,
            ground_truth,
            calculator_id,
            upper_limit,
            lower_limit
        )
        return bool(is_correct)
    except Exception as e:
        # If evaluation fails, mark as incorrect
        return False


def embed_texts_batch(texts: List[str], client, batch_size: int = 100):
    """
    Batch-embed texts using text-embedding-3-small.
    Used by SubmodularBankBuilder (Stage 1) and SEACRRetriever (Stage 2).

    Args:
        texts:      list of strings to embed
        client:     synchronous OpenAI client instance
        batch_size: texts per API call (OpenAI max is 2048)

    Returns:
        np.ndarray of shape (len(texts), 1536), dtype float32
    """
    import numpy as np
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = [t[:8000] for t in texts[i:i + batch_size]]
        response = client.embeddings.create(model="text-embedding-3-small", input=batch)
        batch_embs = [r.embedding for r in sorted(response.data, key=lambda x: x.index)]
        all_embeddings.extend(batch_embs)
    return np.array(all_embeddings, dtype=np.float32)
