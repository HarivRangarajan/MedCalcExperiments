import json
import random

def split_validation_data():
    # Load the test data
    test_file_path = "/Users/sadrishya/Acads/IS/MedCalcExperiments/pubmedqa-experiments/data/pqal_test.json"
    val_file_path = "/Users/sadrishya/Acads/IS/MedCalcExperiments/pubmedqa-experiments/data/pqal_val.json"
    
    with open(test_file_path, 'r') as f:
        test_data = json.load(f)
    
    # Get all PMIDs (keys) from the test data
    all_pmids = list(test_data.keys())
    
    # Randomly select 100 PMIDs for validation (truly random, no seed)
    val_pmids = random.sample(all_pmids, 100)
    
    # Create validation dataset
    val_data = {pmid: test_data[pmid] for pmid in val_pmids}
    
    # Remove selected PMIDs from test data
    updated_test_data = {pmid: test_data[pmid] for pmid in all_pmids if pmid not in val_pmids}
    
    # Save validation data
    with open(val_file_path, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    # Update test data file
    with open(test_file_path, 'w') as f:
        json.dump(updated_test_data, f, indent=2)
    
    print(f"Successfully created validation set with {len(val_data)} examples")
    print(f"Updated test set now has {len(updated_test_data)} examples")
    print(f"Validation data saved to: {val_file_path}")

if __name__ == "__main__":
    split_validation_data()