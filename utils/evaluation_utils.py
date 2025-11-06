"""Evaluation utilities for prompt assessment."""

import re
from typing import Tuple


def extract_numerical_answer(response_text: str) -> float:
    """Extract numerical answer from response text.
    
    Args:
        response_text: The text response to extract from
        
    Returns:
        float: The extracted numerical value
        
    Raises:
        ValueError: If no numerical answer can be found
    """
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


def evaluate_with_tolerance(predicted: float, ground_truth: float, tolerance: float = 0.05) -> bool:
    """Evaluate numerical answer with tolerance.
    
    Args:
        predicted: The predicted numerical value
        ground_truth: The ground truth value
        tolerance: The tolerance level (default: 5%)
        
    Returns:
        bool: True if predicted is within tolerance of ground truth
    """
    if ground_truth == 0:
        return abs(predicted) < tolerance
    return abs((predicted - ground_truth) / ground_truth) < tolerance


def clean_prompt_name(prompt_type: str) -> str:
    """Clean prompt type names for display.
    
    Args:
        prompt_type: The prompt type string to clean
        
    Returns:
        str: A cleaned, readable version of the prompt type name
    """
    if prompt_type.startswith('original_'):
        return prompt_type.replace('original_', 'Original ').replace('_', ' ').title()
    elif prompt_type.startswith('enhanced_'):
        return prompt_type.replace('enhanced_', 'Enhanced ').replace('_', ' ').title()
    return prompt_type.replace('_', ' ').title()

