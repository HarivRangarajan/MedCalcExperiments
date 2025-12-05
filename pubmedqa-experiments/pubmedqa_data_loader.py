#!/usr/bin/env python3
"""
PubMedQA Data Loader (Labeled Dataset Only) - Refactored

Loads and processes PQA-L (labeled) dataset with semantic categorization
based on the paper's manual analysis. Creates train/test splits from existing files.
"""

import json
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
from collections import Counter


# --- regex fallbacks (kept lightweight) ---
THERAPY_RX = re.compile(r"\b(therapy|treat(?:ment)?|efficac|effective|safe(?:ty)?|feasible|recommended|indicated|useful|accuracy|valid|reliable)\b", re.I)
ASSOC_RX   = re.compile(r"\b(association|associated|relationship|related|correlat|risk factor|predict(?:or|ive)?|marker|determinant|link)\b", re.I)
CAUSAL_RX  = re.compile(r"\b(effect of|impact of|role of|influence of|increase|decrease|reduce|improve|worsen|promote|prevent|cause|lead to|contribute|mediate|modulate)\b", re.I)

# --- MeSH buckets ---
THERAPY_MESH = {
    "Treatment Outcome","Therapeutics","Drug Therapy","Randomized Controlled Trials as Topic",
    "Clinical Trials as Topic","Cost-Benefit Analysis","Safety","Feasibility Studies",
    "Sensitivity and Specificity","ROC Curve","Reproducibility of Results","Predictive Value of Tests",
    "Diagnosis","Diagnostic Imaging","Tomography, X-Ray Computed","Magnetic Resonance Imaging",
    "Ultrasonography","Biopsy","Questionnaires","Assay","Biomarkers"
}
ASSOC_MESH = {
    "Risk Factors","Cohort Studies","Case-Control Studies","Cross-Sectional Studies",
    "Longitudinal Studies","Odds Ratio","Logistic Models","Incidence","Prevalence",
    "Correlation (Statistics)","Concordance","Risk Assessment","Regression Analysis",
    "Surveys and Questionnaires","Epidemiologic Studies","Predictive Value of Tests"
}
CAUSAL_MESH = {
    "Drug Effects","Adverse Effects","Etiology","Causality",
    "Dose-Response Relationship, Drug","Time Factors","Disease Models, Animal",
    "Treatment Failure","Complications"
}
# helpers for nudges
DIAG_MODALITY = {"Diagnostic Imaging","Tomography, X-Ray Computed","Magnetic Resonance Imaging","Ultrasonography","Biopsy","Questionnaires"}
DIAG_METRICS  = {"Sensitivity and Specificity","ROC Curve","Predictive Value of Tests","Reproducibility of Results"}
OUTCOME_MESH  = {"Complications","Treatment Outcome","Mortality","Survival Rate","Cardiovascular Diseases"}

def categorize_question(question: str, meshes: Optional[List[str]] = None) -> str:
    """
    Returns one of:
      'therapy_evaluation', 'association_relatedness', 'factor_influence', 'statement_truthiness'
    If meshes are provided, uses a MeSH-first scorer; otherwise falls back to regex on the title.
    """
    # 1) MeSH-first scoring (only if meshes provided and non-empty)
    if meshes:
        mset = set(meshes)
        scores = Counter({
            "therapy_evaluation":     len(mset & THERAPY_MESH) * 2,
            "association_relatedness":len(mset & ASSOC_MESH)   * 2,
            "factor_influence":       len(mset & CAUSAL_MESH)  * 2,
        })

        # cross-signal nudges
        if (mset & DIAG_MODALITY) and (mset & DIAG_METRICS):
            scores["therapy_evaluation"] += 2
        if (mset & ASSOC_MESH) and not (mset & DIAG_METRICS):
            scores["association_relatedness"] += 1
        if (mset & {"Therapeutics","Drug Therapy"}) and (mset & OUTCOME_MESH):
            scores["factor_influence"] += 1

        best, val = scores.most_common(1)[0]
        if val > 0:
            return best  # done if MeSH gave a signal

    # 2) Title fallback (regex)
    s = (question or "").lower()
    if THERAPY_RX.search(s): return "therapy_evaluation"
    if ASSOC_RX.search(s):   return "association_relatedness"
    if CAUSAL_RX.search(s):  return "factor_influence"
    return "statement_truthiness"


class PubMedQADataLoader:
    """Load and manage PQA-L (labeled) dataset with train/test split from existing files."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory to save processed data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Define source file paths relative to current directory
        self.source_paths = {
            "dev_set": Path("../pubmedqa-master/data/pqal_fold0/dev_set.json"),
            "train_set": Path("../pubmedqa-master/data/pqal_fold0/train_set.json"),
            "test_set": Path("../pubmedqa-master/data/test_set.json")
        }
        
        print(f"✅ PubMedQA data loader initialized")
        print(f"   • Output directory: {self.data_dir}")
    
    def load_and_process_json(self, file_path: Path, max_samples: int = None) -> List[Dict]:
        """
        Load and process a single JSON file.
        
        Args:
            file_path: Path to the JSON file
            max_samples: Maximum number of samples to load (None for all)
            
        Returns:
            List of processed records
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")
            
        print(f"   📁 Loading {file_path.name}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        records = []
        items = list(data.items())
        
        # Limit samples if specified
        if max_samples:
            items = items[:max_samples]
            
        for pmid, item in tqdm(items, desc=f"Processing {file_path.name}"):
            # Extract context text (handle both string and list formats)
            if isinstance(item.get("CONTEXTS", []), list):
                context_text = " ".join(item["CONTEXTS"])
            else:
                context_text = item.get("CONTEXTS", "")
            
            # Categorize question semantically
            question_category = categorize_question(item["QUESTION"])
            
            # Extract final decision
            final_decision = item.get("final_decision")
            
            record = {
                "PMID": pmid,
                "QUESTION": item["QUESTION"],
                "CONTEXTS": context_text,
                "LONG_ANSWER": item.get("LONG_ANSWER", ""),
                "FINAL_DECISION": final_decision,
                "CATEGORY": question_category,
                "LABELS": item.get("LABELS", []),
                "MESHES": item.get("MESHES", []),
                "YEAR": item.get("YEAR", None)
            }
            records.append(record)
        
        print(f"   ✓ Processed {len(records)} records from {file_path.name}")
        return records
    
    def create_train_dataset(self, max_samples: int = 500) -> List[Dict]:
        """
        Create training dataset by merging dev_set and train_set.
        
        Args:
            max_samples: Maximum number of samples in final train set
            
        Returns:
            List of training records
        """
        print(f"\n📊 Creating training dataset (max {max_samples} samples)...")
        
        # Load dev_set
        dev_records = self.load_and_process_json(self.source_paths["dev_set"])
        
        # Load train_set
        train_records = self.load_and_process_json(self.source_paths["train_set"])
        
        # Merge datasets
        all_train_records = dev_records + train_records
        
        # Shuffle and limit to max_samples
        import random
        random.seed(42)  # For reproducibility
        random.shuffle(all_train_records)
        
        if len(all_train_records) > max_samples:
            all_train_records = all_train_records[:max_samples]
        
        print(f"   ✓ Created training dataset: {len(all_train_records)} samples")
        print(f"     • From dev_set: {len(dev_records)} samples")
        print(f"     • From train_set: {len(train_records)} samples")
        
        return all_train_records
    
    def create_test_dataset(self, max_samples: int = 500) -> List[Dict]:
        """
        Create test dataset from test_set.
        
        Args:
            max_samples: Maximum number of samples in test set
            
        Returns:
            List of test records
        """
        print(f"\n📊 Creating test dataset (max {max_samples} samples)...")
        
        test_records = self.load_and_process_json(self.source_paths["test_set"], max_samples)
        
        print(f"   ✓ Created test dataset: {len(test_records)} samples")
        
        return test_records
    
    def print_dataset_statistics(self, records: List[Dict], dataset_name: str):
        """Print statistics for a dataset."""
        print(f"\n📈 {dataset_name} Dataset Statistics:")
        print(f"   • Total samples: {len(records)}")
        
        # Answer distribution
        answers = [r["FINAL_DECISION"] for r in records if r["FINAL_DECISION"]]
        if answers:
            from collections import Counter
            answer_counts = Counter(answers)
            print(f"   • Answer distribution:")
            for answer, count in answer_counts.most_common():
                pct = count / len(answers) * 100
                print(f"     {answer}: {count} ({pct:.1f}%)")
        
        # Category distribution
        categories = [r["CATEGORY"] for r in records]
        if categories:
            from collections import Counter
            category_counts = Counter(categories)
            print(f"   • Category distribution:")
            for category, count in category_counts.most_common():
                pct = count / len(categories) * 100
                print(f"     {category}: {count} ({pct:.1f}%)")
    
    def save_dataset(self, records: List[Dict], filename: str):
        """Save dataset to JSON file."""
        output_path = self.data_dir / filename
        
        # Convert to dict format (PMID as key)
        data_dict = {record["PMID"]: record for record in records}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)
        
        print(f"   💾 Saved {len(records)} records to {output_path}")
    
    def process_all_datasets(self):
        """Process and save both train and test datasets."""
        print("🚀 Starting PubMedQA dataset processing...\n")
        
        # Create train dataset (dev_set + train_set)
        train_records = self.create_train_dataset(max_samples=500)
        self.print_dataset_statistics(train_records, "Training")
        self.save_dataset(train_records, "pqal_train.json")
        
        # Create test dataset (test_set)
        test_records = self.create_test_dataset(max_samples=500)
        self.print_dataset_statistics(test_records, "Test")
        self.save_dataset(test_records, "pqal_test.json")
        
        print(f"\n✅ Dataset processing complete!")
        print(f"📁 Files created in {self.data_dir}:")
        print(f"   • pqal_train.json - {len(train_records)} training samples")
        print(f"   • pqal_test.json - {len(test_records)} test samples")
        
        return train_records, test_records


def test_categorization():
    """Test the semantic categorization function."""
    test_questions = [
        "Is aspirin effective for preventing heart attacks?",  # therapy_evaluation
        "Are smoking and lung cancer associated?",             # association_relatedness  
        "Does exercise reduce blood pressure?",                # factor_influence
        "Mitochondria are the powerhouse of the cell.",       # statement_truthiness
    ]
    
    print("🧪 Testing semantic categorization:")
    for question in test_questions:
        category = categorize_question(question)
        print(f"   '{question}' → {category}")


def main():
    """Main execution function."""
    # Test categorization
    test_categorization()
    print()
    
    # Process datasets
    loader = PubMedQADataLoader()
    train_records, test_records = loader.process_all_datasets()
    
    # Show sample records
    print(f"\n📋 Sample Training Record:")
    if train_records:
        sample = train_records[0]
        for key, value in sample.items():
            if key in ["CONTEXTS", "LONG_ANSWER"]:
                print(f"   {key}: {str(value)[:100]}...")
            else:
                print(f"   {key}: {value}")
    
    print(f"\n📋 Sample Test Record:")
    if test_records:
        sample = test_records[0]
        for key, value in sample.items():
            if key in ["CONTEXTS", "LONG_ANSWER"]:
                print(f"   {key}: {str(value)[:100]}...")
            else:
                print(f"   {key}: {value}")


if __name__ == "__main__":
    main()