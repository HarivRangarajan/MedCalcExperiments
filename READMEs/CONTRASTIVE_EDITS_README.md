# MedCalc-Bench with Contrastive Boosted Edits

This pipeline evaluates the MedCalc-Bench dataset using prompt engineering techniques (Chain of Thought and Chain of Draft) enhanced by the PromptEngineer library, comparing them against the original MedCalc one-shot prompts.

## Overview

The pipeline implements the following workflow:

1. **Extract Original One-Shot Prompt**: Uses the exact one-shot prompt structure from MedCalc's `run.py`
2. **Create Enhanced Prompts**: Generates two enhanced versions using:
   - Chain of Thought technique
   - Chain of Draft technique
3. **Load Train Data**: Reads a configurable number of samples from MedCalc train data
4. **Generate Responses**: Produces responses for all three prompt types (original, CoT, CoD)
5. **Evaluate**: Uses MedCalc's evaluation logic to check correctness
6. **Split Results**: Separates correct and incorrect responses into different files

## Usage

### Basic Usage

Run with default settings (500 samples):

```bash
cd medcalc-evaluation
source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate
export OPENAI_API_KEY="sk-..."
python medcalc_with_contrastive_boosted_edits.py
```

### Custom Sample Size

Run with a specific number of samples:

```bash
python medcalc_with_contrastive_boosted_edits.py --sample-size 100
```

### Using the Runner Script

For easier execution with quick results:

```bash
python run_medcalc_with_contrastive_boosted_edits.py
```

Note: Edit the `SAMPLE_SIZE` variable in `run_medcalc_with_contrastive_boosted_edits.py` to change the number of samples (default is 10 for testing).

### Command-Line Arguments

```bash
python medcalc_with_contrastive_boosted_edits.py \
  --sample-size 500 \
  --output-dir ./my_results \
  --model OpenAI/gpt-4o
```

**Arguments:**
- `--sample-size`: Number of train examples to evaluate (default: 500)
- `--output-dir`: Output directory for results (default: auto-generated with timestamp)
- `--model`: Model to use for inference (default: OpenAI/gpt-4o)

## Output Structure

The pipeline creates the following output structure:

```
outputs/medcalc_contrastive_edits_evaluation_TIMESTAMP/
├── data/
│   └── sampled_medcalc_data.csv          # The sampled train data used
├── prompts/
│   └── enhanced_prompts.json              # Enhanced prompts (CoT & CoD)
├── responses/
│   ├── original_responses.jsonl           # Responses using original prompt
│   ├── chain_of_thought_responses.jsonl   # Responses using CoT enhanced
│   └── chain_of_draft_responses.jsonl     # Responses using CoD enhanced
├── evaluations/
│   └── evaluation_summary.json            # Accuracy metrics for each prompt type
├── correct/
│   ├── original_correct.jsonl             # Correct responses (original)
│   ├── chain_of_thought_correct.jsonl     # Correct responses (CoT)
│   └── chain_of_draft_correct.jsonl       # Correct responses (CoD)
└── incorrect/
    ├── original_incorrect.jsonl           # Incorrect responses (original)
    ├── chain_of_thought_incorrect.jsonl   # Incorrect responses (CoT)
    └── chain_of_draft_incorrect.jsonl     # Incorrect responses (CoD)
```

## Output Files

### Response Files (JSONL format)

Each line contains:
```json
{
  "Row Number": 1,
  "Calculator Name": "Creatinine Clearance",
  "Calculator ID": "2",
  "Category": "lab test",
  "Note ID": "pmc-6477550-1",
  "Patient Note": "...",
  "Question": "...",
  "LLM Answer": "141.042",
  "LLM Explanation": "...",
  "Ground Truth Answer": "141.042",
  "Ground Truth Explanation": "...",
  "Result": "Correct",
  "Prompt Type": "chain_of_thought"
}
```

### Evaluation Summary (JSON format)

```json
{
  "original": {
    "accuracy": 0.85,
    "total": 500,
    "correct": 425,
    "incorrect": 75
  },
  "chain_of_thought": {
    "accuracy": 0.87,
    "total": 500,
    "correct": 435,
    "incorrect": 65
  },
  "chain_of_draft": {
    "accuracy": 0.86,
    "total": 500,
    "correct": 430,
    "incorrect": 70
  }
}
```

## Implementation Details

### Original One-Shot Prompt

The exact one-shot prompt from MedCalc's `run.py` (lines 26-32):

```python
def one_shot(note, question, example_note, example_output):
    system_msg = 'You are a helpful assistant...'
    system_msg += f'Here is an example patient note:\n\n{example_note}'
    system_msg += f'\n\nHere is an example task:\n\n{question}'
    system_msg += f'\n\nPlease directly output the JSON dict...'
    user_temp = f'Here is the patient note:\n\n{note}...'
    return system_msg, user_temp
```

### Enhanced Prompts

Enhanced prompts are generated using the PromptEngineer library:

1. **Chain of Thought**: Adds explicit step-by-step reasoning instructions
2. **Chain of Draft**: Incorporates iterative refinement prompting

### Response Generation

- Uses MedCalc's `LLMInference` class for OpenAI API calls
- Applies the same one-shot examples for each calculator type
- Maintains consistency with MedCalc's evaluation methodology

### Evaluation

Uses MedCalc's `check_correctness` function from `evaluate.py`:
- Handles different output types (date, integer, decimal, tuple)
- Applies appropriate tolerance ranges for numerical answers
- Calculator-specific validation logic

## Requirements

```bash
# Activate virtual environment
cd /path/to/PromptResearch/medcalc-evaluation
source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate

# Dependencies should already be installed in the virtual environment
# If needed, install:
# pip install pandas numpy openai tqdm

# Set environment variable
export OPENAI_API_KEY="your-api-key-here"
```

## Cost Estimation

For 500 samples with 3 prompt types (1,500 total API calls):
- Model: GPT-4o
- Average tokens per call: ~2,000 (input) + ~500 (output)
- Estimated cost: $15-25 USD

**Recommendation**: Start with a small sample size (10-50) to test before running the full evaluation.

## Example Session

```bash
$ cd medcalc-evaluation
$ export OPENAI_API_KEY="sk-..."
$ python medcalc_with_contrastive_boosted_edits.py --sample-size 50

====================================================================================================
MEDCALC-BENCH WITH CONTRASTIVE BOOSTED EDITS EVALUATION PIPELINE
====================================================================================================
Timestamp: 2025-10-10 14:23:45
Sample size: 50
Model: OpenAI/gpt-4o

📋 STEP 1: Loading MedCalc-Bench Data (Sample: 50)
============================================================
✅ Loaded 8649 total examples from MedCalc-Bench train data
   • Randomly sampled 50 examples

🚀 STEP 2: Creating Enhanced Prompts
============================================================
✅ Enhanced prompts generated:
   • chain_of_thought: 1,234 characters
   • chain_of_draft: 1,456 characters

🔧 STEP 3: Generating Responses
============================================================
   Generating responses for: original
Processing original: 100%|████████████| 50/50
   Generating responses for: chain_of_thought
Processing chain_of_thought: 100%|████████████| 50/50
   Generating responses for: chain_of_draft
Processing chain_of_draft: 100%|████████████| 50/50

📊 STEP 4: Evaluating and Splitting Responses
============================================================
   Evaluating: original
      Accuracy: 84.00% (42/50)
      Saved 42 correct responses to: original_correct.jsonl
      Saved 8 incorrect responses to: original_incorrect.jsonl
   
   Evaluating: chain_of_thought
      Accuracy: 88.00% (44/50)
      Saved 44 correct responses to: chain_of_thought_correct.jsonl
      Saved 6 incorrect responses to: chain_of_thought_incorrect.jsonl
   
   Evaluating: chain_of_draft
      Accuracy: 86.00% (43/50)
      Saved 43 correct responses to: chain_of_draft_correct.jsonl
      Saved 7 incorrect responses to: chain_of_draft_incorrect.jsonl

====================================================================================================
EVALUATION COMPLETE
====================================================================================================

Key Results:
   • original: 84.00% accuracy (42/50)
   • chain_of_thought: 88.00% accuracy (44/50)
   • chain_of_draft: 86.00% accuracy (43/50)

All results saved to: outputs/medcalc_contrastive_edits_evaluation_20251010_142345/

✅ Pipeline execution completed successfully!
```

## Troubleshooting

### Import Errors

If you see import errors for `llm_inference` or `evaluate`:

```bash
# Ensure you're running from the medcalc-evaluation directory
cd medcalc-evaluation

# Check that MedCalc-Bench is properly cloned
ls MedCalc-Bench/evaluation/
```

### Train Data Not Found

If train_data.csv is missing:

```bash
cd MedCalc-Bench/dataset
unzip train_data.csv.zip
```

### API Key Errors

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Verify it's set
echo $OPENAI_API_KEY
```

## Further Analysis

After running the pipeline, you can analyze the results:

```python
import json
import pandas as pd

# Load evaluation summary
with open('outputs/.../evaluations/evaluation_summary.json', 'r') as f:
    summary = json.load(f)

# Load correct responses for analysis
correct_cot = pd.read_json('outputs/.../correct/chain_of_thought_correct.jsonl', lines=True)
incorrect_cot = pd.read_json('outputs/.../incorrect/chain_of_thought_incorrect.jsonl', lines=True)

# Analyze by calculator type
print(correct_cot['Calculator Name'].value_counts())
print(incorrect_cot['Calculator Name'].value_counts())

# Compare explanations
for idx, row in correct_cot.head().iterrows():
    print(f"Question: {row['Question']}")
    print(f"LLM Explanation: {row['LLM Explanation']}")
    print(f"Ground Truth: {row['Ground Truth Explanation']}")
    print("-" * 80)
```

## Next Steps

1. **Analyze Results**: Compare accuracy across prompt types
2. **Error Analysis**: Examine incorrect responses to understand failure modes
3. **Prompt Refinement**: Use insights to further improve prompts
4. **Scale Up**: Run with larger sample sizes for statistical significance
5. **Try Other Techniques**: Experiment with additional PromptEngineer techniques

## References

- MedCalc-Bench: [GitHub Repository](https://github.com/ncbi-nlp/MedCalc)
- PromptEngineer Library: See `../promptengineer/README.md`

