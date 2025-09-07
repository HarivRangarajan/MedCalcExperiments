# MedCalc-Bench Evaluation Setup

### Prerequisites

1. **API Key**: OpenAI API key configured in `../mohs-llm-as-a-judge/configs/config.py` or as environment variable
2. **Dependencies**: Install required packages

### Installation

```bash
# Install requirements
pip install -r requirements.txt

# Verify MedCalc-Bench data is available
ls MedCalc-Bench/dataset/test_data.csv
```

### Integration Points
- Shares PromptEngineer library with wound care pipeline
- Uses same LLM judge evaluation framework
- Compatible with existing API key configuration
- Modular design for easy extension