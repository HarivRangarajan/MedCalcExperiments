#!/usr/bin/env python3
"""
Data Validation Script for PubMedQA Datasets

Validates that:
1. No PMIDs overlap between train and test sets
2. All required fields are present in each record
3. All field names are in uppercase
"""

import json
from pathlib import Path
from typing import Set, Dict, Any, List

# Required fields (all uppercase)
REQUIRED_FIELDS = {
    "QUESTION",
    "CONTEXTS", 
    "LABELS",
    "MESHES",
    "YEAR",
    "REASONING_REQUIRED_PRED",
    "REASONING_FREE_PRED",
    "FINAL_DECISION",
    "LONG_ANSWER",
    "PMID"
}

def load_json_data(file_path: Path) -> Dict[str, Any]:
    """Load JSON data from file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def validate_key_uniqueness(train_data: Dict, test_data: Dict) -> bool:
    """
    Validate that no PMIDs overlap between train and test sets.
    
    Args:
        train_data: Training dataset
        test_data: Test dataset
        
    Returns:
        True if validation passes, False otherwise
    """
    print("🔍 Validating PMID uniqueness between datasets...")
    
    train_keys = set(train_data.keys())
    test_keys = set(test_data.keys())
    
    # Check for overlapping keys
    overlapping_keys = train_keys.intersection(test_keys)
    
    if overlapping_keys:
        print(f"❌ VALIDATION FAILED: Found {len(overlapping_keys)} overlapping PMIDs:")
        for key in sorted(list(overlapping_keys)[:10]):  # Show first 10
            print(f"   • {key}")
        if len(overlapping_keys) > 10:
            print(f"   • ... and {len(overlapping_keys) - 10} more")
        return False
    else:
        print(f"✅ VALIDATION PASSED: No overlapping PMIDs found")
        print(f"   • Train set: {len(train_keys)} unique PMIDs")
        print(f"   • Test set: {len(test_keys)} unique PMIDs")
        print(f"   • Total unique: {len(train_keys) + len(test_keys)} PMIDs")
        return True

def validate_record_fields(data: Dict, dataset_name: str) -> bool:
    """
    Validate that all records have required fields in uppercase.
    
    Args:
        data: Dataset to validate
        dataset_name: Name of dataset for logging
        
    Returns:
        True if validation passes, False otherwise
    """
    print(f"\n🔍 Validating required fields in {dataset_name} dataset...")
    
    missing_fields_records = []
    invalid_case_records = []
    
    for pmid, record in data.items():
        # Check if record has all required fields
        record_fields = set(record.keys())
        missing_fields = REQUIRED_FIELDS - record_fields
        
        if missing_fields:
            missing_fields_records.append({
                'pmid': pmid,
                'missing': missing_fields,
                'present': record_fields
            })
        
        # Check if all field names are uppercase
        non_uppercase_fields = [field for field in record_fields if field != field.upper()]
        if non_uppercase_fields:
            invalid_case_records.append({
                'pmid': pmid,
                'non_uppercase': non_uppercase_fields
            })
    
    # Report missing fields
    if missing_fields_records:
        print(f"❌ VALIDATION FAILED: {len(missing_fields_records)} records missing required fields")
        for i, record_info in enumerate(missing_fields_records[:5]):  # Show first 5
            print(f"   • PMID {record_info['pmid']}: Missing {record_info['missing']}")
        if len(missing_fields_records) > 5:
            print(f"   • ... and {len(missing_fields_records) - 5} more records")
        return False
    
    # Report case issues
    if invalid_case_records:
        print(f"❌ VALIDATION FAILED: {len(invalid_case_records)} records have non-uppercase field names")
        for i, record_info in enumerate(invalid_case_records[:5]):  # Show first 5
            print(f"   • PMID {record_info['pmid']}: Non-uppercase fields {record_info['non_uppercase']}")
        if len(invalid_case_records) > 5:
            print(f"   • ... and {len(invalid_case_records) - 5} more records")
        return False
    
    print(f"✅ VALIDATION PASSED: All {len(data)} records have required uppercase fields")
    return True

def validate_field_types(data: Dict, dataset_name: str) -> bool:
    """
    Validate basic field types and non-empty values.
    
    Args:
        data: Dataset to validate
        dataset_name: Name of dataset for logging
        
    Returns:
        True if validation passes, False otherwise
    """
    print(f"\n🔍 Validating field types in {dataset_name} dataset...")
    
    issues = []
    
    for pmid, record in data.items():
        # Check PMID matches key
        if record.get("PMID") != pmid:
            issues.append(f"PMID {pmid}: PMID field '{record.get('PMID')}' doesn't match key")
        
        # Check required string fields are not empty
        string_fields = ["QUESTION", "CONTEXTS", "LONG_ANSWER", "FINAL_DECISION"]
        for field in string_fields:
            if field in record:
                if not isinstance(record[field], str) or not record[field].strip():
                    issues.append(f"PMID {pmid}: {field} is empty or not a string")
        
        # Check list fields are lists
        list_fields = ["LABELS", "MESHES"]
        for field in list_fields:
            if field in record:
                if not isinstance(record[field], list):
                    issues.append(f"PMID {pmid}: {field} is not a list")
        
        # Check FINAL_DECISION is valid
        if "FINAL_DECISION" in record:
            valid_decisions = {"yes", "no", "maybe"}
            decision = record["FINAL_DECISION"].lower().strip()
            if decision not in valid_decisions:
                issues.append(f"PMID {pmid}: Invalid FINAL_DECISION '{decision}' (should be yes/no/maybe)")
    
    if issues:
        print(f"❌ VALIDATION FAILED: Found {len(issues)} field type issues")
        for issue in issues[:10]:  # Show first 10
            print(f"   • {issue}")
        if len(issues) > 10:
            print(f"   • ... and {len(issues) - 10} more issues")
        return False
    
    print(f"✅ VALIDATION PASSED: All field types are valid")
    return True

def print_dataset_summary(data: Dict, dataset_name: str):
    """Print summary statistics for a dataset."""
    print(f"\n📊 {dataset_name} Dataset Summary:")
    print(f"   • Total records: {len(data)}")
    
    # Count field presence
    field_counts = {}
    for record in data.values():
        for field in record.keys():
            field_counts[field] = field_counts.get(field, 0) + 1
    
    print(f"   • Field presence:")
    for field in sorted(REQUIRED_FIELDS):
        count = field_counts.get(field, 0)
        pct = count / len(data) * 100 if data else 0
        print(f"     {field}: {count}/{len(data)} ({pct:.1f}%)")
    
    # Show answer distribution if available
    if data:
        decisions = []
        for record in data.values():
            decision = record.get("FINAL_DECISION", "").lower().strip()
            if decision:
                decisions.append(decision)
        
        if decisions:
            from collections import Counter
            decision_counts = Counter(decisions)
            print(f"   • Answer distribution:")
            for decision, count in decision_counts.most_common():
                pct = count / len(decisions) * 100
                print(f"     {decision}: {count} ({pct:.1f}%)")

def validate_category_distribution(train_data: Dict, test_data: Dict) -> Dict[str, Dict[str, int]]:
    """
    Count keys by category in both train and test datasets.
    
    Categories to analyze:
    - therapy_evaluation
    - association_relatedness  
    - factor_influence
    - statement_truthiness
    
    Args:
        train_data: Training dataset
        test_data: Test dataset
        
    Returns:
        Dictionary with category counts for train and test
    """
    target_categories = {
        'therapy_evaluation',
        'association_relatedness', 
        'factor_influence',
        'statement_truthiness'
    }
    
    def count_categories_in_dataset(data: Dict, dataset_name: str) -> Dict[str, int]:
        """Count categories in a single dataset."""
        category_counts = {cat: 0 for cat in target_categories}
        total_records = 0
        unmatched_records = 0
        
        for pmid, record in data.items():
            total_records += 1
            
            # Check if record has required fields to determine category
            if not isinstance(record, dict):
                continue
                
            # Look for category indicators in the record
            # This might be in QUESTION field or a separate CATEGORY field
            question = record.get('QUESTION', '').lower()
            category_field = record.get('CATEGORY', '').lower()
            
            # Try to classify based on question content or category field
            categorized = False
            
            # Check explicit category field first
            if category_field:
                for cat in target_categories:
                    if cat.replace('_', ' ') in category_field or cat in category_field:
                        category_counts[cat] += 1
                        categorized = True
                        break
            
            # If no explicit category, try to infer from question content
            if not categorized and question:
                # Therapy evaluation keywords
                if any(keyword in question for keyword in ['treatment', 'therapy', 'drug', 'medication', 'efficacy', 'effective']):
                    category_counts['therapy_evaluation'] += 1
                    categorized = True
                # Association/relatedness keywords  
                elif any(keyword in question for keyword in ['associated', 'related', 'correlation', 'relationship']):
                    category_counts['association_relatedness'] += 1
                    categorized = True
                # Factor/influence keywords
                elif any(keyword in question for keyword in ['factor', 'influence', 'cause', 'risk', 'predictor']):
                    category_counts['factor_influence'] += 1
                    categorized = True
                # Statement truthiness keywords
                elif any(keyword in question for keyword in ['true', 'false', 'correct', 'accurate', 'statement']):
                    category_counts['statement_truthiness'] += 1
                    categorized = True
            
            if not categorized:
                unmatched_records += 1
        
        print(f"\n{dataset_name} Dataset Analysis:")
        print(f"  Total records: {total_records}")
        print(f"  Categorized records: {total_records - unmatched_records}")
        print(f"  Unmatched records: {unmatched_records}")
        
        return category_counts
    
    # Count categories in both datasets
    train_counts = count_categories_in_dataset(train_data, "Training")
    test_counts = count_categories_in_dataset(test_data, "Test")
    
    # Create summary
    results = {
        'train': train_counts,
        'test': test_counts,
        'combined': {cat: train_counts[cat] + test_counts[cat] for cat in target_categories}
    }
    
    # Print detailed results
    print("\n" + "="*60)
    print("CATEGORY DISTRIBUTION ANALYSIS")
    print("="*60)
    
    print(f"\n{'Category':<25} {'Train':<10} {'Test':<10} {'Total':<10}")
    print("-" * 55)
    
    for category in target_categories:
        train_count = results['train'][category]
        test_count = results['test'][category] 
        total_count = results['combined'][category]
        print(f"{category:<25} {train_count:<10} {test_count:<10} {total_count:<10}")
    
    # Calculate totals
    train_total = sum(results['train'].values())
    test_total = sum(results['test'].values()) 
    combined_total = sum(results['combined'].values())
    
    print("-" * 55)
    print(f"{'TOTAL':<25} {train_total:<10} {test_total:<10} {combined_total:<10}")
    
    # Calculate percentages
    if train_total > 0 or test_total > 0:
        print(f"\nPercentage Distribution:")
        print(f"{'Category':<25} {'Train %':<10} {'Test %':<10}")
        print("-" * 45)
        
        for category in target_categories:
            train_pct = (results['train'][category] / train_total * 100) if train_total > 0 else 0
            test_pct = (results['test'][category] / test_total * 100) if test_total > 0 else 0
            print(f"{category:<25} {train_pct:<10.1f} {test_pct:<10.1f}")
    
    return results

# Add this function to the main validation routine
def main():
    """Main validation function."""
    # Define paths to datasets
    train_path = Path("data/pqal_train.json")  # Adjust path as needed
    test_path = Path("data/pqal_test.json")    # Adjust path as needed
    
    try:
        # Load datasets
        print("Loading datasets...")
        train_data = load_json_data(train_path)
        test_data = load_json_data(test_path)
        
        print(f"✅ Loaded {len(train_data)} training records")
        print(f"✅ Loaded {len(test_data)} test records")
        
        # Validate PMID uniqueness
        is_unique = validate_key_uniqueness(train_data, test_data)
        
        # Validate category distribution
        category_results = validate_category_distribution(train_data, test_data)
        
        # Save results
        results_file = Path("validation_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'pmid_uniqueness': is_unique,
                'category_distribution': category_results,
                'validation_timestamp': str(Path(__file__).stat().st_mtime)
            }, f, indent=2)
        
        print(f"\n✅ Validation results saved to: {results_file}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Validation failed: {e}")

if __name__ == "__main__":
    main()