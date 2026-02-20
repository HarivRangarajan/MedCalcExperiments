# Unified Prompt Refinement Pipeline - User Guide

## 🎯 Overview

The refinement pipeline has been updated to use a **unified prompt approach** instead of refining separate CoT and CoD prompts independently. This new approach:

1. **Starts with a unified prompt** created by combining CoT and CoD enhanced prompts from PromptEngineer
2. **Refines this single unified prompt** using feedback from ALL training examples (both CoT and CoD results)
3. **Uses GPT-5 as the default model** for refinement (configurable)
4. **Evaluates on 170 training examples** after each iteration to track progress

---

## 🔄 What Changed

### **Before (Old Approach):**
1. Refine `chain_of_thought` prompt separately using CoT examples
2. Refine `chain_of_draft` prompt separately using CoD examples  
3. Combine the three refined prompts into a final unified prompt at the end
4. Always used GPT-4o for refinement

### **After (New Unified Approach):**
1. Start by combining CoT and CoD enhanced prompts into one unified prompt
2. Refine this single unified prompt using ALL training examples (CoT + CoD)
3. Each iteration improves the same unified prompt (no separate prompt types)
4. Use GPT-5 by default (configurable via `--model` parameter)

---

## 📋 How It Works

### **Step 1: Initial Unified Prompt Creation**
- Loads `enhanced_prompts.json` from the results directory
- Uses GPT-5 to intelligently combine CoT and CoD prompts
- Creates a single, cohesive starting prompt

### **Step 2: Load ALL Training Examples**
- Reads from `correct/` and `incorrect/` directories
- Combines examples from all prompt types (CoT, CoD, etc.)
- Example counts: ~170 total examples (85 correct + 85 incorrect)

### **Step 3: Iterative Refinement**
For each iteration (default: 10 iterations, batch size: 17):
1. Take next batch of examples (mix of correct + incorrect)
2. Analyze what works (correct) and what fails (incorrect)
3. Use GPT-5 to refine the unified prompt based on this feedback
4. **Evaluate refined prompt on ALL 170 training examples**
5. Track accuracy improvement
6. Save iteration result

### **Step 4: Final Output**
- Best refined unified prompt saved to `final/unified_prompt.txt`
- Evaluation progress plot showing accuracy over iterations
- Complete refinement history in JSON format

---

## 🚀 How to Run

### **Option 1: Full Pipeline (Recommended)**

Run refinement + evaluation + visualization together:

```bash
./run_refine_eval_visualize.sh --model gpt-4o
```

**What this does:**
1. **Refinement** (uses GPT-5): Creates and refines unified prompt
2. **Evaluation** (uses gpt-4o): Tests refined prompt on 1047 test examples
3. **Visualization**: Generates comparison plots

**To use different evaluation model:**
```bash
./run_refine_eval_visualize.sh --model gpt-5
```

### **Option 2: Refinement Only**

Run just the refinement pipeline:

```bash
cd medcalc-evaluation
source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate

python pipeline/prompt_refinement_pipeline.py \
  --results-dir /path/to/outputs/medcalc_contrastive_edits_evaluation_TIMESTAMP \
  --batch-size 17 \
  --max-iterations 10 \
  --model gpt-5
```

**Parameters:**
- `--results-dir` (required): Directory with training results (correct/incorrect examples)
- `--batch-size` (default: 17): Examples per refinement iteration
- `--max-iterations` (default: None = all): Number of refinement iterations
- `--output-dir` (optional): Where to save refined prompts
- `--model` (default: gpt-5): Model to use for refinement

**Example with GPT-4o for refinement:**
```bash
python pipeline/prompt_refinement_pipeline.py \
  --results-dir outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
  --batch-size 17 \
  --max-iterations 10 \
  --model gpt-4o
```

---

## 📂 Output Structure

After running refinement, you'll get:

```
outputs/refined_prompts_TIMESTAMP/
├── iterations/
│   ├── unified_iteration_0.json      # Initial unified prompt
│   ├── unified_iteration_1.json      # After 1st refinement
│   ├── unified_iteration_2.json      # After 2nd refinement
│   └── ...
├── final/
│   ├── unified_prompt.txt            # ⭐ Final refined prompt
│   └── final_refined_prompts.json    # Same, in JSON format
├── evaluation_progress/
│   ├── accuracy_progress.png         # 📊 Accuracy improvement plot
│   ├── accuracy_progress.pdf
│   └── accuracy_history.json         # Raw accuracy data
├── logs/
├── refinement_history.json           # Complete iteration history
└── refinement_summary.json           # Summary statistics
```

**Key files:**
- **`final/unified_prompt.txt`**: Use this for evaluation
- **`accuracy_progress.png`**: Shows if refinement is working
- **`refinement_summary.json`**: Accuracy improvement stats

---

## 📊 Example Output

```
🚀 Running pipeline
================================================
Refinement model: gpt-5
Evaluation model: gpt-4o

📋 Loading training examples from all sources (CoT, CoD, etc.)...
📦 Loaded training examples:
   • Correct: 85
   • Incorrect: 85
   • Total: 170

🔗 Creating initial unified prompt from existing enhanced prompts...
📄 Loaded enhanced prompts:
   • chain_of_thought
   • chain_of_draft
   • Combining 2 enhanced prompts using gpt-5...
   ✓ Initial unified prompt created (1234 characters)

🎯 Evaluating initial prompt...
      📊 Evaluating prompt on 170 training examples...
      ✓ Accuracy: 52.35% (89/170 correct)

🔧 Starting iterative refinement of unified prompt
============================================================

   Iteration 1/10:
      • Processing 9 correct, 8 incorrect
      ✓ Refined prompt saved
      📊 Evaluating prompt on 170 training examples...
      ✓ Accuracy: 54.71% (93/170 correct)

   Iteration 2/10:
      • Processing 8 correct, 9 incorrect
      ✓ Refined prompt saved
      📊 Evaluating prompt on 170 training examples...
      ✓ Accuracy: 56.47% (96/170 correct)

   ...

============================================================
REFINEMENT COMPLETE
============================================================

📊 Summary:
   • Model used: gpt-5
   • Training examples: 170
   • Refinement iterations: 10
   • Initial accuracy: 52.35%
   • Final accuracy: 61.18%
   • Improvement: +8.83%
   • Unified prompt: 1567 characters

📊 Evaluation progress plot saved
📁 All outputs saved to: outputs/refined_prompts_TIMESTAMP/
```

---

## 🔧 Configuration Options

### **Model Selection**

**For Refinement:**
```bash
--model gpt-5      # Default, recommended for best results
--model gpt-4o     # Faster, cheaper, still good
```

**For Evaluation** (in `run_refine_eval_visualize.sh`):
```bash
./run_refine_eval_visualize.sh --model gpt-4o    # Default
./run_refine_eval_visualize.sh --model gpt-5     # Higher quality
```

### **Batch Size & Iterations**

**Smaller batches, more iterations:**
```bash
--batch-size 10 --max-iterations 15
```
- More gradual refinement
- More opportunities to evaluate and adjust

**Larger batches, fewer iterations:**
```bash
--batch-size 25 --max-iterations 7
```
- Faster execution
- Each iteration sees more examples at once

**Default (recommended):**
```bash
--batch-size 17 --max-iterations 10
```

---

## 🎯 Best Practices

### **1. Check Training Data Quality**
Before refinement, verify your training results:
```bash
ls -lh outputs/medcalc_contrastive_edits_evaluation_TIMESTAMP/correct/
ls -lh outputs/medcalc_contrastive_edits_evaluation_TIMESTAMP/incorrect/
```

Should see:
- `chain_of_thought_correct.jsonl`
- `chain_of_thought_incorrect.jsonl`
- `chain_of_draft_correct.jsonl`
- `chain_of_draft_incorrect.jsonl`

### **2. Monitor Accuracy Progress**
After refinement, check the plot:
```bash
open outputs/refined_prompts_TIMESTAMP/evaluation_progress/accuracy_progress.png
```

**Good signs:**
- Accuracy increases over iterations
- Plateaus at a higher level than initial

**Bad signs:**
- Accuracy decreases or fluctuates wildly
- No improvement after multiple iterations
- → Try different batch size or check training data quality

### **3. Use Refined Prompt for Evaluation**
The `run_refine_eval_visualize.sh` script automatically picks up the latest refined prompt:
```bash
REFINED_DIR=$(ls -td outputs/refined_prompts_* | head -n 1)
```

Or manually specify:
```bash
python pipeline/evaluate_contrastive_fewshot_method.py \
  --refined-prompts-dir outputs/refined_prompts_20241205_123456 \
  --training-results-dir outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
  --num-test-examples 1047 \
  --model gpt-4o
```

---

## 🐛 Troubleshooting

### **Error: "Enhanced prompts file not found"**
```
FileNotFoundError: Enhanced prompts file not found: .../enhanced_prompts.json
```

**Solution:** Make sure your `--results-dir` points to a directory from contrastive demonstration generation that contains:
```
results-dir/
  ├── prompts/enhanced_prompts.json
  ├── correct/
  └── incorrect/
```

### **Error: "No training examples loaded"**
```
📦 Loaded training examples:
   • Correct: 0
   • Incorrect: 0
   • Total: 0
```

**Solution:** Run contrastive demonstration generation first:
```bash
python pipeline/contrastive_demonstration_generation.py \
  --sample-size 170 \
  --output-dir outputs/training_data
```

### **Low Initial Accuracy**
If initial accuracy is < 40%, check:
1. Training examples quality (are they actually correct/incorrect?)
2. One-shot examples are loading properly
3. Evaluation logic matches what was used during training

### **No Improvement After Refinement**
If final accuracy ≈ initial accuracy:
1. Try a different model (GPT-5 vs GPT-4o)
2. Increase `--max-iterations` to 15-20
3. Check if training examples have enough error diversity
4. Review refinement history to see what changes were made

---

## 📊 Understanding the Output

### **refinement_summary.json**
```json
{
  "timestamp": "2024-12-05T10:30:00",
  "model": "gpt-5",
  "approach": "unified_prompt_refinement",
  "training_examples": {
    "correct": 85,
    "incorrect": 85,
    "total": 170
  },
  "refinement_iterations": 10,
  "initial_accuracy": 0.5235,
  "final_accuracy": 0.6118,
  "accuracy_improvement": 0.0883
}
```

**Key metrics:**
- `accuracy_improvement`: How much the refinement helped (positive = good)
- `final_accuracy`: Performance on training set (target: > 60%)

### **Iteration Files**
Each `unified_iteration_N.json` contains:
```json
{
  "iteration": 3,
  "prompt": "The refined prompt text...",
  "num_correct": 9,
  "num_incorrect": 8,
  "accuracy": 0.5647,
  "timestamp": "2024-12-05T10:35:00"
}
```

---

## 🔄 Comparison: Old vs. New Approach

| Aspect | Old (Separate) | New (Unified) |
|--------|---------------|---------------|
| **Starting point** | Original MedCalc prompt | Combined CoT + CoD enhanced prompts |
| **Refinement target** | 3 separate prompts | 1 unified prompt |
| **Training data** | Split by prompt type | All examples together |
| **Iterations** | Per prompt type | Global refinement |
| **Final combination** | At the end | Throughout process |
| **Model** | Always GPT-4o | Configurable (default: GPT-5) |
| **Evaluation** | Only at end | After every iteration |

**Advantages of unified approach:**
1. ✅ Single cohesive prompt from start to finish
2. ✅ Learns from ALL examples, not just one prompt type
3. ✅ Continuous evaluation shows progress
4. ✅ Better model (GPT-5) for refinement
5. ✅ Simpler conceptual model (one prompt to rule them all)

---

## 🎓 Summary

**To run refinement + evaluation:**
```bash
./run_refine_eval_visualize.sh --model gpt-4o
```

**What happens:**
1. GPT-5 combines CoT + CoD prompts → unified prompt
2. GPT-5 refines prompt using 170 training examples (10 iterations)
3. GPT-4o evaluates refined prompt on 1047 test examples
4. Visualizations show performance comparison

**Output:** Publication-ready results in `outputs/` directory

**Key file:** `outputs/refined_prompts_TIMESTAMP/final/unified_prompt.txt`

---

**Questions?** Check the code comments in `pipeline/prompt_refinement_pipeline.py` for implementation details.

