# Setup Instructions: 

## Part 1: Generating the contrastive demonstrations

In a couple of sentences, this part takes care of the following:
1. Takes the base prompt used in the MedCalcBench paper
2. Creates `promptengineer` enhanced flavors of the base prompt (CoT, CoD...)
3. Takes a subset of the train examples from the MedCalcBench dataset and generates responses for the subset
4. Evaluates the above responses to create contrastive demonstrations (positve, negative)

## What do we need to set up?

1. You need to have the MedCalc-Bench dataset (of course)
2. Install the `promptengineer` library (this is our custom dependency), and the rest of the library dependencies
3. Have a Open AI API key
4. That is pretty much all. Let the experiments begin!

N.B: I am assuming that your virtual env is at `mohs-llm-as-a-judge/llm-judge-env/bin/activate"`. Check the `setup_and_run.sh` script and modify paths as necessary.

## Quick Setup (Using Automated Script)

The easiest way to get started:

```bash
cd to the place where you have cloned this repo
export OPENAI_API_KEY="sk-..."
./setup_and_run.sh
```

The script will:
1. ✅ Check all prerequisites
2. ✅ Activate virtual environment
3. ✅ Extract train data if needed

## Manual Setup

If you prefer to set things up manually:

### Step 1: Navigate to Directory

```bash
cd /to the place where you have cloned this repo
```

### Step 2: Activate Virtual Environment

```bash
source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate
```

**Important**: The virtual environment contains all required dependencies (pandas, numpy, openai, tqdm, etc.)

### Step 3: Verify Environment

```bash
# Check Python version
python --version  # Should be 3.7+

# Check if packages are available
python -c "import pandas, numpy, openai; print('✅ All packages available')"
```

### Step 4: Set API Key

```bash
export OPENAI_API_KEY="sk-..."

# Verify it's set
echo $OPENAI_API_KEY
```

### Step 5: Extract Train Data (if needed)

```bash
# Check if already extracted
if [ ! -f MedCalc-Bench/dataset/train_data.csv ]; then
    cd MedCalc-Bench/dataset
    unzip train_data.csv.zip
    cd ../..
fi
```

## Directory Structure Verification

Before running, ensure this structure exists:

```
PromptResearch/
├── medcalc-experiements/          ← You are here
│   ├── MedCalc-Bench/
│   │   ├── dataset/
│   │   │   ├── train_data.csv   ← Must exist (unzip if needed)
│   │   │   └── test_data.csv
│   │   └── evaluation/
│   │       ├── llm_inference.py
│   │       ├── evaluate.py
│   │       └── run.py
│   ├── medcalc_with_contrastive_boosted_edits.py
│   └── setup_and_run.sh         ← Automated setup script
├── promptengineer/               ← Must exist
└── mohs-llm-as-a-judge/
    └── llm-judge-env/            ← Virtual environment
        └── bin/
            └── activate          ← Activation script
```

## Common Setup Issues

### Issue: Virtual Environment Not Found

```bash
# Check if path exists
ls ../mohs-llm-as-a-judge/llm-judge-env/bin/activate

# If not found, you may need to create it or adjust the path
```

## Next Steps After Setup

Once setup is complete:

1. **Test with 2 samples**: `python test_contrastive_edits.py`
2. **Full run (600 samples)**: `python medcalc_with_contrastive_boosted_edits.py --sample-size 600`

## Summary: Minimal Setup Commands

```bash
# 1. Navigate
cd /path/to/PromptResearch/medcalc-evaluation

# 2. Activate venv
source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate

# 3. Set API key
export OPENAI_API_KEY="sk-..."

# 4. Extract data (if needed)
[ ! -f MedCalc-Bench/dataset/train_data.csv ] && \
  cd MedCalc-Bench/dataset && unzip train_data.csv.zip && cd ../..

# 5. Test
python test_contrastive_edits.py

# 6. Run
python medcalc_with_contrastive_boosted_edits.py --sample-size 600
```

---
