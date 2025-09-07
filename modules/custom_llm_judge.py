"""
Custom LLM Judge for MedCalc evaluation pipeline.

This module provides LLM-based evaluation capabilities specifically designed
for medical calculation tasks, adapted from the original mohs-llm-as-a-judge
implementation but made self-contained and modular.
"""

import json
import re
from typing import List, Tuple, Optional, Any, Dict
from openai import OpenAI


class CustomLLMJudge:
    """
    Custom LLM Judge for evaluating medical calculation responses.
    
    This class provides comprehensive evaluation capabilities using LLM-based
    judgment, specifically tailored for medical calculation tasks.
    """
    
    # System prompt used for medical calculation evaluation
    SYSTEM_PROMPT = """You are an expert medical professional evaluating AI responses to medical calculation questions.

Evaluate the AI response based on these criteria:
1. **Accuracy**: Is the final numerical answer correct?
2. **Methodology**: Is the calculation approach correct?
3. **Reasoning**: Is the step-by-step reasoning clear and logical?
4. **Value Extraction**: Are the correct values extracted from the patient note?
5. **Clinical Appropriateness**: Is the response clinically sound?

Rate the response as:
- **PASS (1)**: The response is accurate, well-reasoned, and clinically appropriate
- **FAIL (0)**: The response has significant errors in calculation, reasoning, or clinical appropriateness

Return your evaluation in JSON format:
{
    "label": 1 or 0,
    "accuracy": "correct" or "incorrect",
    "methodology": "correct" or "incorrect", 
    "reasoning": "clear" or "unclear",
    "clinical": "appropriate" or "inappropriate",
    "reason": "Brief explanation of your decision"
}"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize Custom LLM Judge.
        
        Args:
            api_key: OpenAI API key
            model: Model name to use for evaluation (default: gpt-4o)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    @property
    def system_prompt(self) -> str:
        """Get the system prompt used for evaluation."""
        return self.SYSTEM_PROMPT
    
    def safe_json_load(self, response_content: str, top_level_key: str = None) -> Optional[Dict]:
        """
        Safely parse JSON from LLM response.
        
        Args:
            response_content: Raw response content from LLM
            top_level_key: Optional key to extract from parsed JSON
            
        Returns:
            Parsed JSON dictionary or None if parsing fails
        """
        try:
            # Find the start and end of the JSON object
            json_start = response_content.find('{')
            json_end = response_content.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                # No JSON found, try to extract from code blocks
                code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
                match = re.search(code_block_pattern, response_content, re.DOTALL)
                if match:
                    json_str = match.group(1)
                else:
                    raise ValueError("No JSON object found in response")
            else:
                json_str = response_content[json_start:json_end]
            
            # Parse the JSON response
            data = json.loads(json_str)
            
            if top_level_key:
                return data.get(top_level_key, None)
            return data
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON parsing error: {e}")
            print(f"Response content (first 500 chars): {response_content[:500]}")
            return None

    def safe_json_list_load(self, response_content: str) -> Optional[List[Dict]]:
        """
        Safely parse JSON list from LLM response.
        
        Args:
            response_content: Raw response content from LLM
            
        Returns:
            Parsed JSON list or None if parsing fails
        """
        try:
            # Find the start and end of the JSON array
            json_start = response_content.find('[')
            json_end = response_content.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                # Try to extract from code blocks
                code_block_pattern = r'```(?:json)?\s*(\[.*?\])\s*```'
                match = re.search(code_block_pattern, response_content, re.DOTALL)
                if match:
                    json_str = match.group(1)
                else:
                    raise ValueError("No JSON array found in response")
            else:
                json_str = response_content[json_start:json_end]
            
            data = json.loads(json_str)
            
            if not isinstance(data, list):
                raise ValueError("Parsed JSON is not a list")
                
            return data
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON list parsing error: {e}")
            print(f"Response content (first 500 chars): {response_content[:500]}")
            return None

    def evaluate_medical_response(self, 
                                patient_note: str,
                                question: str,
                                ai_response: str,
                                ground_truth_answer: str,
                                ground_truth_explanation: str = "") -> Tuple[Optional[int], Optional[str]]:
        """
        Evaluate a single medical calculation response.
        
        Args:
            patient_note: Original patient note
            question: Medical calculation question
            ai_response: AI-generated response to evaluate
            ground_truth_answer: Correct answer
            ground_truth_explanation: Explanation of correct answer
            
        Returns:
            Tuple of (label, reason) where label is 1 for pass, 0 for fail
        """
        # Create evaluation prompt
        user_prompt = f"""
**Patient Note:** {patient_note}

**Question:** {question}

**AI Response:** {ai_response}

**Ground Truth Answer:** {ground_truth_answer}

**Ground Truth Explanation:** {ground_truth_explanation}

Please evaluate this AI response and provide your assessment in the specified JSON format."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                seed=42
            )
            
            evaluation = self.safe_json_load(response.choices[0].message.content)
            
            if evaluation:
                return evaluation.get("label"), evaluation.get("reason")
            else:
                return None, "Failed to parse evaluation response"
                
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return None, f"Evaluation error: {str(e)}"

    def batch_evaluate_responses(self, 
                               evaluations_data: List[Dict[str, str]],
                               max_batch_size: int = 5) -> List[Dict[str, Any]]:
        """
        Evaluate multiple responses in batches for efficiency.
        
        Args:
            evaluations_data: List of dicts with keys: patient_note, question, 
                            ai_response, ground_truth_answer, ground_truth_explanation
            max_batch_size: Maximum number of evaluations per batch
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for i in range(0, len(evaluations_data), max_batch_size):
            batch = evaluations_data[i:i + max_batch_size]
            
            # For now, process individually for reliability
            # Can be optimized later for true batch processing
            for eval_data in batch:
                label, reason = self.evaluate_medical_response(
                    patient_note=eval_data.get("patient_note", ""),
                    question=eval_data.get("question", ""),
                    ai_response=eval_data.get("ai_response", ""),
                    ground_truth_answer=eval_data.get("ground_truth_answer", ""),
                    ground_truth_explanation=eval_data.get("ground_truth_explanation", "")
                )
                
                results.append({
                    "label": label,
                    "reason": reason,
                    "original_data": eval_data
                })
        
        return results

    def create_evaluation_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a comprehensive evaluation report from results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Report dictionary with statistics and analysis
        """
        if not results:
            return {"error": "No results to analyze"}
        
        # Calculate statistics
        total_evaluations = len(results)
        passed_evaluations = sum(1 for r in results if r.get("label") == 1)
        failed_evaluations = total_evaluations - passed_evaluations
        pass_rate = passed_evaluations / total_evaluations if total_evaluations > 0 else 0
        
        # Collect failure reasons
        failure_reasons = [r.get("reason", "Unknown") for r in results if r.get("label") == 0]
        
        report = {
            "summary": {
                "total_evaluations": total_evaluations,
                "passed": passed_evaluations,
                "failed": failed_evaluations,
                "pass_rate": pass_rate
            },
            "failure_analysis": {
                "failure_reasons": failure_reasons,
                "common_issues": self._analyze_common_issues(failure_reasons)
            },
            "detailed_results": results
        }
        
        return report

    def _analyze_common_issues(self, failure_reasons: List[str]) -> Dict[str, int]:
        """
        Analyze common patterns in failure reasons.
        
        Args:
            failure_reasons: List of failure reason strings
            
        Returns:
            Dictionary mapping issue types to counts
        """
        issue_patterns = {
            "calculation_error": ["calculation", "math", "arithmetic", "computed"],
            "methodology_error": ["method", "approach", "formula", "procedure"],
            "value_extraction": ["extract", "value", "missing", "incorrect value"],
            "reasoning_unclear": ["unclear", "confusing", "logic", "reasoning"],
            "clinical_error": ["clinical", "medical", "inappropriate", "unsafe"]
        }
        
        issue_counts = {issue_type: 0 for issue_type in issue_patterns.keys()}
        
        for reason in failure_reasons:
            reason_lower = reason.lower()
            for issue_type, patterns in issue_patterns.items():
                if any(pattern in reason_lower for pattern in patterns):
                    issue_counts[issue_type] += 1
                    break
        
        return issue_counts


def evaluate_medcalc_responses(responses_data: List[Dict[str, str]], 
                             api_key: str,
                             model: str = "gpt-4o") -> Dict[str, Any]:
    """
    Convenience function to evaluate a list of medical calculation responses.
    
    Args:
        responses_data: List of response data dictionaries
        api_key: OpenAI API key
        model: Model to use for evaluation
        
    Returns:
        Comprehensive evaluation report
    """
    judge = CustomLLMJudge(api_key=api_key, model=model)
    results = judge.batch_evaluate_responses(responses_data)
    report = judge.create_evaluation_report(results)
    
    return report 