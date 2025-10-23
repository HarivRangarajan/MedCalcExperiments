# Setup Instructions: MedCalc Contrastive Boosted Edits

## Quick Setup (Using Automated Script)

The easiest way to get started:

```bash
cd /path/to/PromptResearch/medcalc-evaluation
export OPENAI_API_KEY="sk-..."
./setup_and_run.sh
```

The script will:
1. ✅ Check all prerequisites
2. ✅ Activate virtual environment
3. ✅ Extract train data if needed
4. ✅ Let you choose what to run

## Manual Setup

If you prefer to set things up manually:

### Step 1: Navigate to Directory

```bash
cd /path/to/PromptResearch/medcalc-evaluation
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

### Step 6: Run Test

```bash
python test_contrastive_edits.py
```

## Virtual Environment Details

**Location**: `../mohs-llm-as-a-judge/llm-judge-env/`

**Activation**:
```bash
# From medcalc-evaluation directory
source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate
```

**Deactivation**:
```bash
deactivate
```

**Check if Active**:
```bash
which python
# Should show: .../mohs-llm-as-a-judge/llm-judge-env/bin/python
```

## Required Packages

The virtual environment should already have:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `openai` - OpenAI API client
- `tqdm` - Progress bars
- `promptengineer` - Custom prompt engineering library

If any are missing, install them:
```bash
pip install pandas numpy openai tqdm
```

## Directory Structure Verification

Before running, ensure this structure exists:

```
PromptResearch/
├── medcalc-evaluation/          ← You are here
│   ├── MedCalc-Bench/
│   │   ├── dataset/
│   │   │   ├── train_data.csv   ← Must exist (unzip if needed)
│   │   │   └── test_data.csv
│   │   └── evaluation/
│   │       ├── llm_inference.py
│   │       ├── evaluate.py
│   │       └── run.py
│   ├── medcalc_with_contrastive_boosted_edits.py
│   ├── test_contrastive_edits.py
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

### Issue: train_data.csv Missing

```bash
cd MedCalc-Bench/dataset
ls train_data.csv.zip  # Should exist
unzip train_data.csv.zip
cd ../..
```

### Issue: Import Errors

```bash
# Make sure you're in the right directory
pwd  # Should end with /medcalc-evaluation

# Make sure virtual environment is active
which python  # Should show venv path
```

### Issue: OpenAI API Key Not Set

```bash
# Set it in current session
export OPENAI_API_KEY="sk-..."

# Or add to your shell profile for persistence
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc  # or ~/.zshrc
source ~/.bashrc  # or ~/.zshrc
```

## Environment Variables

Required:
- `OPENAI_API_KEY` - Your OpenAI API key

Optional:
- `HUGGINGFACE_TOKEN` - For HuggingFace models (not needed for OpenAI)

## Test Your Setup

Run the automated test to verify everything works:

```bash
# Make sure you're in medcalc-evaluation directory
cd /path/to/PromptResearch/medcalc-evaluation

# Activate virtual environment
source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate

# Set API key
export OPENAI_API_KEY="sk-..."

# Run test
python test_contrastive_edits.py
```

Expected output:
```
🧪 Testing MedCalc Contrastive Edits Pipeline
============================================================

✅ Virtual environment active

1️⃣ Checking Prerequisites...
   ✓ MedCalc-Bench directory found
   ✓ train_data.csv found
   ✓ OPENAI_API_KEY found

2️⃣ Testing Pipeline Initialization...
   ✓ Pipeline initialized successfully

[... more tests ...]

✅ ALL TESTS PASSED!
```

## Next Steps After Setup

Once setup is complete:

1. **Test with 2 samples**: `python test_contrastive_edits.py`
2. **Quick run (10 samples)**: `python run_medcalc_with_contrastive_boosted_edits.py`
3. **Full run (500 samples)**: `python medcalc_with_contrastive_boosted_edits.py --sample-size 500`

## Troubleshooting

If you encounter issues:

1. **Check virtual environment is active**:
   ```bash
   which python  # Should show venv path
   ```

2. **Verify all files exist**:
   ```bash
   ls MedCalc-Bench/dataset/train_data.csv
   ls ../promptengineer/
   ```

3. **Test imports**:
   ```bash
   python -c "import pandas, numpy, openai; print('OK')"
   ```

4. **Check API key**:
   ```bash
   echo $OPENAI_API_KEY  # Should show your key
   ```

## Getting Help

If setup still fails:

1. Check `QUICK_START.md` for usage examples
2. Check `CONTRASTIVE_EDITS_README.md` for detailed documentation
3. Check `IMPLEMENTATION_SUMMARY.md` for technical details

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
python medcalc_with_contrastive_boosted_edits.py --sample-size 500
```

---

**Ready to go?** Run the test script to verify your setup! ✨

