# Quick Start Guide: MedCalc Contrastive Boosted Edits

## TL;DR

```bash
# 1. Setup
cd /path/to/PromptResearch/medcalc-evaluation
source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate
export OPENAI_API_KEY="sk-..."

# 2. Test (2 samples, ~$0.05)
python test_contrastive_edits.py

# 3. Run (500 samples, ~$15-25)
python medcalc_with_contrastive_boosted_edits.py --sample-size 500
```

## What This Does

This pipeline compares three prompt strategies on MedCalc-Bench:

1. **Original**: MedCalc's standard one-shot prompt
2. **Chain of Thought**: Enhanced with explicit reasoning steps
3. **Chain of Draft**: Enhanced with iterative refinement

For each strategy, it:
- Generates responses for medical calculation questions
- Evaluates correctness using MedCalc's evaluation logic
- Splits results into correct/incorrect files

## Files You'll Use

| File | Purpose | When to Use |
|------|---------|-------------|
| `test_contrastive_edits.py` | Quick test (2 samples) | Before running full evaluation |
| `run_medcalc_with_contrastive_boosted_edits.py` | Simple runner (10 samples) | Quick experiments |
| `medcalc_with_contrastive_boosted_edits.py` | Full pipeline (500 samples) | Production runs |

## Prerequisites

```bash
# 1. Activate virtual environment
cd /path/to/PromptResearch/medcalc-evaluation
source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate

# 2. Check directory structure
ls MedCalc-Bench/  # Should exist

# 3. Unzip train data (if needed)
cd MedCalc-Bench/dataset
unzip train_data.csv.zip
cd ../..

# 4. Set API key
export OPENAI_API_KEY="sk-..."

# 5. Verify PromptEngineer is available
ls ../promptengineer/  # Should exist
```

## Step-by-Step Usage

### Step 1: Test the Pipeline (Recommended First)

```bash
# Activate virtual environment first
source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate

# Run test
python test_contrastive_edits.py
```

This will:
- Check all prerequisites
- Test initialization and data loading
- Optionally test response generation (1 sample)
- Verify output structure

**Cost**: Free (no API calls) or ~$0.03-0.05 if you test API

### Step 2: Small Run (10 samples)

```bash
python run_medcalc_with_contrastive_boosted_edits.py
```

Edit `SAMPLE_SIZE` in the file to adjust (default: 10).

**Time**: ~2-5 minutes  
**Cost**: ~$0.30-0.50

### Step 3: Full Run (500 samples)

```bash
python medcalc_with_contrastive_boosted_edits.py --sample-size 500
```

**Time**: ~30-45 minutes  
**Cost**: ~$15-25

### Step 4: Analyze Results

```bash
# View summary
cat outputs/medcalc_contrastive_edits_evaluation_*/evaluations/evaluation_summary.json

# Count results
wc -l outputs/medcalc_contrastive_edits_evaluation_*/correct/*.jsonl
wc -l outputs/medcalc_contrastive_edits_evaluation_*/incorrect/*.jsonl

# View sample correct response
head -1 outputs/medcalc_contrastive_edits_evaluation_*/correct/chain_of_thought_correct.jsonl | jq .
```

## Output Structure Explained

```
outputs/medcalc_contrastive_edits_evaluation_TIMESTAMP/
│
├── 📊 evaluations/
│   └── evaluation_summary.json          # Accuracy metrics
│
├── ✅ correct/                           # Correct responses by prompt type
│   ├── original_correct.jsonl
│   ├── chain_of_thought_correct.jsonl
│   └── chain_of_draft_correct.jsonl
│
├── ❌ incorrect/                         # Incorrect responses by prompt type
│   ├── original_incorrect.jsonl
│   ├── chain_of_thought_incorrect.jsonl
│   └── chain_of_draft_incorrect.jsonl
│
├── 📝 responses/                         # All responses (correct + incorrect)
│   ├── original_responses.jsonl
│   ├── chain_of_thought_responses.jsonl
│   └── chain_of_draft_responses.jsonl
│
├── 🎯 prompts/
│   └── enhanced_prompts.json            # CoT & CoD enhanced prompts
│
└── 📋 data/
    └── sampled_medcalc_data.csv         # The samples used
```

## Command-Line Options

```bash
# Basic
python medcalc_with_contrastive_boosted_edits.py

# Custom sample size
python medcalc_with_contrastive_boosted_edits.py --sample-size 100

# Custom output directory
python medcalc_with_contrastive_boosted_edits.py --output-dir ./my_results

# All options
python medcalc_with_contrastive_boosted_edits.py \
  --sample-size 250 \
  --output-dir ./experiment_1 \
  --model OpenAI/gpt-4o
```

## Common Issues

### "MedCalc-Bench directory not found"
```bash
# Make sure you're in the right directory
cd /path/to/PromptResearch/medcalc-evaluation
pwd  # Should end with /medcalc-evaluation
```

### "train_data.csv not found"
```bash
cd MedCalc-Bench/dataset
unzip train_data.csv.zip
cd ../..
```

### "OPENAI_API_KEY not set"
```bash
export OPENAI_API_KEY="sk-..."
echo $OPENAI_API_KEY  # Verify it's set
```

### "Import llm_inference failed"
```bash
# Run from medcalc-evaluation directory
cd /path/to/PromptResearch/medcalc-evaluation
python medcalc_with_contrastive_boosted_edits.py
```

## Quick Analysis Examples

### Compare Accuracy Across Prompts

```python
import json

with open('outputs/.../evaluations/evaluation_summary.json') as f:
    results = json.load(f)

for prompt_type, metrics in results.items():
    print(f"{prompt_type}: {metrics['accuracy']:.1%}")
```

### Find Calculator Types with Most Errors

```python
import pandas as pd

incorrect = pd.read_json('outputs/.../incorrect/chain_of_thought_incorrect.jsonl', lines=True)
print(incorrect['Calculator Name'].value_counts().head(10))
```

### Compare Explanations

```python
import pandas as pd

correct = pd.read_json('outputs/.../correct/chain_of_thought_correct.jsonl', lines=True)

for idx, row in correct.head(3).iterrows():
    print(f"Question: {row['Question'][:100]}...")
    print(f"LLM: {row['LLM Explanation'][:200]}...")
    print(f"GT: {row['Ground Truth Explanation'][:200]}...")
    print("-" * 80)
```

**Note**: Each sample requires 3 API calls (original, CoT, CoD)

## Troubleshooting Decision Tree

```
Problem?
│
├─ Import Error
│  └─ Check: pwd ends with /medcalc-evaluation
│
├─ File Not Found
│  ├─ train_data.csv → unzip MedCalc-Bench/dataset/train_data.csv.zip
│  └─ MedCalc-Bench/ → clone/download MedCalc-Bench
│
├─ API Error
│  └─ export OPENAI_API_KEY="sk-..."
│
├─ Out of Memory
│  └─ Reduce --sample-size
│
└─ Slow Performance
   └─ Normal! 2-5 seconds per sample is expected
```

## Next Steps After Running

1. **Compare Accuracy**: Which prompt type performed best?
2. **Error Analysis**: What types of questions failed most?
3. **Cost Analysis**: Was the improvement worth the cost?
4. **Prompt Refinement**: Use insights to improve prompts
5. **Scale Up**: Run with larger sample sizes

## Files Modified/Created

### Main Implementation
- ✏️ **Modified**: `medcalc_with_contrastive_boosted_edits.py` (complete rewrite)
- ✏️ **Modified**: `run_medcalc_with_contrastive_boosted_edits.py` (updated)

### Documentation
- ✨ **New**: `CONTRASTIVE_EDITS_README.md` (comprehensive guide)
- ✨ **New**: `IMPLEMENTATION_SUMMARY.md` (technical details)
- ✨ **New**: `QUICK_START.md` (this file)

### Testing
- ✨ **New**: `test_contrastive_edits.py` (validation script)

## Support

For more details, see:
- `CONTRASTIVE_EDITS_README.md` - Full documentation
- `IMPLEMENTATION_SUMMARY.md` - Technical implementation details

## Feedback Loop

After running the evaluation:

1. Check `evaluation_summary.json` for accuracy metrics
2. Review `incorrect/*.jsonl` files to understand failures
3. Examine `enhanced_prompts.json` to see what was changed
4. Consider adjustments to PromptEngineer techniques
5. Re-run with refinements

## Success Criteria

You know it worked if:
- ✅ All three response files created
- ✅ Evaluation summary shows accuracy > 0%
- ✅ Correct/incorrect files properly split
- ✅ Sample count matches input parameter
- ✅ No errors in terminal output

---

**Ready to start?**

```bash
python test_contrastive_edits.py  # Test first!
```

