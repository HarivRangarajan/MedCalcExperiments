# MedCalc-Bench Prompt Engineering Evaluation

This evaluation system compares the performance of original MedCalc-Bench prompts against PromptEngineer-generated enhanced prompts on medical calculation tasks.

## Overview

The system evaluates 6 different prompt approaches:

**Original MedCalc Prompts:**
- Direct Answer
- Zero-shot Chain of Thought  
- One-shot Chain of Thought

**PromptEngineer Enhanced Prompts:**
- Chain of Thought
- Chain of Thoughtlessness
- Chain of Draft

## Evaluation Metrics

1. **Numerical Accuracy**: Exact match with ground truth answers (with tolerance)
2. **LLM-as-a-Judge**: Qualitative evaluation of response quality including:
   - Overall pass/fail
   - Accuracy of extracted values
   - Correctness of methodology
   - Clarity of reasoning
   - Clinical appropriateness

## Directory Structure
```
medcalc-evaluation/
├── MedCalc-Bench/                           # Cloned repository with dataset
│   ├── dataset/test_data.csv                # 1,047 test examples 
│   └── evaluation/                          # Original evaluation code
├── medcalc_prompt_evaluation_pipeline.py    # Main evaluation system
├── run_medcalc_evaluation.py                # Simple runner script  
├── test_setup.py                           # Setup verification
├── requirements.txt                        # Dependencies
├── README.md                               # Comprehensive documentation
└── SETUP_SUMMARY.md                       # This file
```

## Setup

Please read SETUP.md!

## 🚀 How to Run

### Quick Start (Recommended)

After completing setup:

```bash
# Activate environment
source source medcalc-env/bin/activate
# Run with default settings (20 samples, 5 responses per technique)
python run_medcalc_evaluation.py
```

### Debugging

```bash
python medcalc_prompt_evaluation_pipeline.py --sample-size 2 --skip-judge 
```

### Advanced Usage

```bash
# Custom sample size and full responses
python medcalc_prompt_evaluation_pipeline.py --sample-size 30 --max-responses 25

# Skip LLM judge to save costs
python medcalc_prompt_evaluation_pipeline.py --sample-size 60 --skip-judge

# Custom output directory
python medcalc_prompt_evaluation_pipeline.py --output-dir hari_experiment_2025
```

### Command Line Options

- `--sample-size`: Number of MedCalc examples to evaluate (default: 20)
- `--max-responses`: Maximum responses per technique (default: 5 responses)
- `--output-dir`: Custom output directory (default: auto-generated)
- `--skip-judge`: Skip LLM-as-a-judge evaluation to reduce costs

## Output Structure

```
medcalc_evaluation_YYYYMMDD_HHMMSS/
├── data/
│   └── sampled_medcalc_data.csv          # Selected test examples
├── prompts/
│   ├── original_medcalc_prompts.json     # Original prompt templates
│   └── promptengineer_prompts.json       # Enhanced prompts
├── responses/
│   └── all_responses.json                # Generated responses
├── evaluations/
│   ├── accuracy_results.json             # Numerical accuracy results
│   └── llm_judge_results.json           # LLM judge evaluations
├── visualizations/
│   ├── overall_accuracy.png              # Overall performance comparison
│   ├── category_accuracy_heatmap.png     # Performance by medical category
│   ├── prompt_type_comparison.png        # Original vs Enhanced comparison
│   ├── llm_judge_results.png            # Judge evaluation breakdown
│   └── statistical_comparison.png        # Statistical significance tests
└── reports/
    ├── evaluation_report.txt             # Detailed text report
    └── evaluation_summary.json           # JSON summary for analysis
```

## Key Features

### Comprehensive Comparison
- Tests both original MedCalc approaches and PromptEngineer techniques
- Evaluates across all medical calculation categories (lab, risk, diagnosis, etc.)
- Provides statistical significance testing

### Dual Evaluation Approach
- **Built-in Accuracy**: Uses MedCalc's numerical answer evaluation
- **LLM-as-a-Judge**: Evaluates response quality and reasoning

### Rich Visualizations
- Overall accuracy comparison
- Category-wise performance heatmaps
- Statistical comparison charts
- Multi-metric evaluation breakdowns
## Understanding the Results

### Accuracy Metrics
- **Overall Accuracy**: Percentage of correct numerical answers
- **Category Accuracy**: Performance breakdown by medical calculation type
- **Statistical Tests**: T-tests comparing original vs enhanced approaches

### LLM Judge Metrics
- **Pass Rate**: Overall quality assessment
- **Accuracy**: Correctness of extracted values
- **Methodology**: Appropriateness of calculation approach
- **Reasoning**: Clarity of step-by-step explanation
- **Clinical**: Medical appropriateness of response

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the medcalc-evaluation directory
2. **API Key Issues**: Verify OpenAI API key is properly configured
3. **Memory Issues**: Reduce sample size for large evaluations
4. **Cost Concerns**: Use `--skip-judge` or reduce `--max-responses`

## Extending the System

### Adding New Prompt Techniques

1. Add technique to PromptEngineer library
2. Update `techniques` list in `generate_promptengineer_prompts()`
3. Modify visualization functions to handle new technique

### Custom Evaluation Metrics

1. Implement new metric in `evaluate_accuracy()`
2. Add visualization in `_plot_*()` methods
3. Update report generation

### Different Medical Domains

1. Modify `create_medcalc_context()` for domain-specific requirements
2. Update judge system prompt for domain expertise
3. Adjust evaluation criteria as needed
