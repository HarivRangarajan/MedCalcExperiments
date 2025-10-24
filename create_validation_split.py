#!/usr/bin/env python3
"""
Create Validation Split from Training Results

This script randomly samples examples from training results to create a
validation set for grid search optimization.

Usage:
    python create_validation_split.py \
        --training-results-dir <path> \
        --validation-size 100 \
        --output-dir <path>
"""

import json
import random
from pathlib import Path
import argparse
from typing import List, Dict
from datetime import datetime


def load_training_examples(training_dir: Path) -> List[Dict]:
    """Load all training examples (both correct and incorrect)."""
    examples = []
    
    # Load correct examples
    correct_dir = training_dir / "correct"
    if correct_dir.exists():
        for file in correct_dir.glob("*.jsonl"):
            with open(file, 'r') as f:
                for line in f:
                    example = json.loads(line)
                    examples.append(example)
    
    # Load incorrect examples
    incorrect_dir = training_dir / "incorrect"
    if incorrect_dir.exists():
        for file in incorrect_dir.glob("*.jsonl"):
            with open(file, 'r') as f:
                for line in f:
                    example = json.loads(line)
                    examples.append(example)
    
    return examples


def create_validation_split(training_dir: str,
                           validation_size: int,
                           output_dir: str = None,
                           seed: int = 42) -> Dict:
    """
    Create validation split by randomly sampling from training examples.
    
    Args:
        training_dir: Directory with training results
        validation_size: Number of validation examples
        output_dir: Output directory
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with validation split info
    """
    random.seed(seed)
    
    training_dir = Path(training_dir)
    
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(__file__).parent / "validation_split" / f"val_{validation_size}_{timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("VALIDATION SPLIT CREATION")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load all training examples
    print("📋 Loading training examples...")
    all_examples = load_training_examples(training_dir)
    print(f"   ✓ Loaded {len(all_examples)} total examples")
    
    # Separate correct and incorrect
    correct_examples = [ex for ex in all_examples if ex.get('Result') == 'Correct']
    incorrect_examples = [ex for ex in all_examples if ex.get('Result') == 'Incorrect']
    
    print(f"      - {len(correct_examples)} correct")
    print(f"      - {len(incorrect_examples)} incorrect")
    
    # Calculate how many of each to sample (maintain ratio)
    total = len(all_examples)
    correct_ratio = len(correct_examples) / total
    incorrect_ratio = len(incorrect_examples) / total
    
    num_correct_val = int(validation_size * correct_ratio)
    num_incorrect_val = validation_size - num_correct_val
    
    print(f"\n📊 Creating validation split of {validation_size} examples...")
    print(f"   • Correct: {num_correct_val} ({correct_ratio:.1%})")
    print(f"   • Incorrect: {num_incorrect_val} ({incorrect_ratio:.1%})")
    
    # Random sample
    if len(correct_examples) >= num_correct_val:
        val_correct = random.sample(correct_examples, num_correct_val)
    else:
        print(f"   ⚠️  Warning: Not enough correct examples, using all {len(correct_examples)}")
        val_correct = correct_examples
        num_correct_val = len(correct_examples)
    
    if len(incorrect_examples) >= num_incorrect_val:
        val_incorrect = random.sample(incorrect_examples, num_incorrect_val)
    else:
        print(f"   ⚠️  Warning: Not enough incorrect examples, using all {len(incorrect_examples)}")
        val_incorrect = incorrect_examples
        num_incorrect_val = len(incorrect_examples)
    
    # Combine and shuffle
    validation_examples = val_correct + val_incorrect
    random.shuffle(validation_examples)
    
    print(f"   ✓ Created validation split with {len(validation_examples)} examples")
    
    # Save validation examples
    val_file = output_dir / "validation_examples.jsonl"
    with open(val_file, 'w') as f:
        for example in validation_examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"\n💾 Saved validation examples to: {val_file}")
    
    # Create metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "training_dir": str(training_dir),
        "total_training_examples": total,
        "validation_size": len(validation_examples),
        "num_correct": num_correct_val,
        "num_incorrect": num_incorrect_val,
        "correct_ratio": correct_ratio,
        "incorrect_ratio": incorrect_ratio,
        "random_seed": seed,
        "validation_file": str(val_file)
    }
    
    # By calculator breakdown
    by_calculator = {}
    for ex in validation_examples:
        calc_name = ex['Calculator Name']
        if calc_name not in by_calculator:
            by_calculator[calc_name] = {"total": 0, "correct": 0, "incorrect": 0}
        by_calculator[calc_name]["total"] += 1
        if ex['Result'] == 'Correct':
            by_calculator[calc_name]["correct"] += 1
        else:
            by_calculator[calc_name]["incorrect"] += 1
    
    metadata["by_calculator"] = by_calculator
    
    # Save metadata
    metadata_file = output_dir / "validation_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Saved metadata to: {metadata_file}")
    
    print(f"\n📊 Validation split by calculator:")
    for calc_name, stats in sorted(by_calculator.items()):
        print(f"   • {calc_name}: {stats['total']} ({stats['correct']} correct, {stats['incorrect']} incorrect)")
    
    print(f"\n{'='*80}")
    print("VALIDATION SPLIT COMPLETE")
    print(f"{'='*80}")
    print(f"\n📁 Output directory: {output_dir}/")
    
    return {
        "validation_examples": validation_examples,
        "metadata": metadata,
        "output_dir": output_dir
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create validation split from training results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--training-results-dir',
        type=str,
        required=True,
        help='Directory containing training evaluation results'
    )
    
    parser.add_argument(
        '--validation-size',
        type=int,
        default=100,
        help='Number of validation examples (default: 100)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: auto-generated)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    result = create_validation_split(
        training_dir=args.training_results_dir,
        validation_size=args.validation_size,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    print(f"\n✅ Validation split created successfully!")
    print(f"📁 Outputs: {result['output_dir']}")


if __name__ == "__main__":
    main()

