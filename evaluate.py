#!/usr/bin/env python3
"""
Evaluation script to calculate the number of correct predictions for each category.
"""

import json
import argparse
from collections import defaultdict
from typing import Dict, Any


def load_answers(filename: str) -> Dict[str, Any]:
    """Load answers from JSON file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{filename}': {e}")
        exit(1)


def evaluate_predictions(data: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """
    Evaluate predictions and calculate accuracy per category.
    
    Returns:
        Dictionary with category names as keys and accuracy stats as values.
    """
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    # Process each record
    for record in data.get("record", []):
        category = record.get("category", "unknown")
        gt_response = record.get("gt_response", "").strip()
        pred_response = record.get("pred_response", "").strip()
        
        category_stats[category]["total"] += 1
        
        if gt_response == pred_response:
            category_stats[category]["correct"] += 1
    
    return dict(category_stats)


def calculate_total_time(data: Dict[str, Any]) -> float:
    """
    Calculate total time from the 'time' key in the data.
    
    Returns:
        Total time as the sum of all time entries, or 0 if no time data found.
    """
    time_data = data.get("time", {})
    if not time_data:
        return 0.0
    
    total_time = sum(time_data.values())
    return total_time


def print_results(category_stats: Dict[str, Dict[str, int]], total_time: float = None) -> None:
    """Print evaluation results in a formatted table."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"{'Category':<25} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-"*60)
    
    total_correct = 0
    total_samples = 0
    
    for category, stats in sorted(category_stats.items()):
        correct = stats["correct"]
        total = stats["total"]
        accuracy = (correct / total * 100) if total > 0 else 0
        
        print(f"{category:<25} {correct:<10} {total:<10} {accuracy:.1f}%")
        
        total_correct += correct
        total_samples += total
    
    print("-"*60)
    overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
    print(f"{'OVERALL':<25} {total_correct:<10} {total_samples:<10} {overall_accuracy:.1f}%")
    print("="*60)
    
    if total_time is not None:
        print(f"\nTOTAL TIME: {total_time:.2f} seconds")
        print("="*60)


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(
        description="Calculate the number of correct predictions for each category from a JSON answer file."
    )
    parser.add_argument(
        "answer_file",
        type=str,
        help="Path to the JSON answer file"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed breakdown for each sample"
    )
    
    args = parser.parse_args()
    
    # Load and evaluate answers
    print(f"Loading answers from: {args.answer_file}")
    data = load_answers(args.answer_file)
    
    category_stats = evaluate_predictions(data)
    
    if not category_stats:
        print("No records found in the answer file.")
        return
    
    # Calculate total time
    total_time = calculate_total_time(data)
    
    # Print results
    print_results(category_stats, total_time)
    
    # Show detailed breakdown if requested
    if args.detailed:
        print("\nDETAILED BREAKDOWN:")
        print("="*60)
        for record in data.get("record", []):
            sample_id = record.get("sample_id", "N/A")
            category = record.get("category", "unknown")
            gt_response = record.get("gt_response", "").strip()
            pred_response = record.get("pred_response", "").strip()
            correct = "✓" if gt_response == pred_response else "✗"
            
            print(f"Sample {sample_id} [{category}]: GT={gt_response}, Pred={pred_response} {correct}")


if __name__ == "__main__":
    main()
