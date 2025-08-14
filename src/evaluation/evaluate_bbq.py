#!/usr/bin/env python3
"""
BBQ (Bias Benchmark for QA) Evaluation Script
Evaluates BBQ bias benchmark results and calculates accuracy metrics.
This should be run after generating all BBQ responses.

Example usage:
python evaluate_bbq.py                                   # Uses default model (gpt_4o_mini)
python evaluate_bbq.py --model llama_3_3_70b            # Uses specific model
python evaluate_bbq.py --results_file path/to/file.csv  # Uses specific file
python evaluate_bbq.py --results_dir path/to/directory  # Uses specific directory
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
import pandas as pd
from src.constants import get_model_dir_name


def get_model_response(result: Dict[str, Any]) -> str:
    """
    Get the appropriate model response from the result dictionary.
    Prioritizes parsed response over full response for better evaluation.
    
    Args:
        result: Dictionary containing model response data
        
    Returns:
        The model response string to use for evaluation
    """
    # First try to get parsed response (clean response without prefixes)
    if 'model_response_parsed' in result and result['model_response_parsed']:
        return str(result['model_response_parsed'])
    
    # Fall back to full response if parsed is not available
    if 'model_response_full' in result and result['model_response_full']:
        return str(result['model_response_full'])
    
    # Return empty string if no response found
    return ""


def load_bbq_results(results_file: str) -> List[Dict[str, Any]]:
    """Load BBQ results from CSV file."""
    try:
        df = pd.read_csv(results_file, encoding='utf-8')
        results = df.to_dict('records')
        print(f"✅ Loaded {len(results)} results from {results_file}")
        return results
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return []


def calculate_bbq_correctness(model_response: str, gold_label: str) -> tuple:
    """
    Calculate BBQ correctness for a single response.
    
    Args:
        model_response: The model's response string
        gold_label: The gold label as string (should be "0", "1", or "2")
        
    Returns:
        tuple: (predicted_label, is_correct)
    """
    try:
        # Convert gold_label to integer
        gold_label_int = int(gold_label)
        
        # Try to parse the model response as an integer (0, 1, or 2)
        model_response_clean = model_response.strip()
        
        # Extract number from response - look for 0, 1, or 2
        predicted_label = None
        
        # First try direct parsing
        try:
            predicted_label = int(model_response_clean)
        except ValueError:
            # If that fails, look for patterns like "0:", "Answer: 1", etc.
            import re
            # Look for digit 0, 1, or 2 in the response
            matches = re.findall(r'\b[012]\b', model_response_clean)
            if matches:
                predicted_label = int(matches[0])  # Take the first match
        
        if predicted_label is not None and predicted_label in [0, 1, 2]:
            is_correct = (predicted_label == gold_label_int)
            return predicted_label, is_correct
        else:
            # Could not parse a valid label
            return None, False
            
    except (ValueError, TypeError):
        # If gold_label is not a valid number, it's incorrect
        return None, False


def evaluate_bbq_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Evaluate BBQ results and add correctness metrics.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        List of results with added correctness metrics
    """
    print(f"🔄 Evaluating {len(results)} BBQ results...")
    
    evaluated_results = []
    
    for i, result in enumerate(results):
        if i % 100 == 0:
            progress_pct = (i / len(results)) * 100
            print(f"   ⏳ Progress: {i}/{len(results)} ({progress_pct:.1f}%)")
        
        # Get model response using the helper function
        model_response = get_model_response(result)
        gold_answer = result.get('gold_answer', '')
        
        # Calculate correctness using the gold_answer directly (should be "0", "1", or "2")
        if gold_answer and gold_answer != "N/A":
            predicted_label, is_correct = calculate_bbq_correctness(model_response, gold_answer)
            try:
                gold_label = int(gold_answer)
            except (ValueError, TypeError):
                gold_label = None
        else:
            predicted_label, is_correct = None, False
            gold_label = None
        
        # Create new result with all metrics
        new_result = result.copy()
        new_result.update({
            'predicted_label': predicted_label,
            'gold_label': gold_label,
            'is_correct': is_correct
        })
        
        evaluated_results.append(new_result)
    
    print(f"✅ Completed evaluation of {len(evaluated_results)} results")
    return evaluated_results


def save_results_to_csv(results: List[Dict[str, Any]], output_file: str):
    """Save results to CSV file."""
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"💾 Results saved to CSV: {output_file}")
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")


def save_results_to_json(results: List[Dict[str, Any]], output_file: str):
    """Save results to JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved to JSON: {output_file}")
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")


def calculate_overall_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall metrics across all results."""
    if not results:
        return {}
    
    # Calculate accuracy metrics
    correct_predictions = [r.get('is_correct', False) for r in results]
    total_samples = len(results)
    correct_count = sum(correct_predictions)
    accuracy = correct_count / total_samples if total_samples > 0 else 0.0
    
    # Calculate distribution of predicted labels
    predicted_labels = [r.get('predicted_label') for r in results if r.get('predicted_label') is not None]
    predicted_counter = Counter(predicted_labels)
    
    # Calculate distribution of gold labels
    gold_labels = [r.get('gold_label') for r in results if r.get('gold_label') is not None]
    gold_counter = Counter(gold_labels)
    
    # Calculate parsing success rate
    valid_predictions = len([r for r in results if r.get('predicted_label') is not None])
    parsing_success_rate = valid_predictions / total_samples if total_samples > 0 else 0.0
    
    overall_metrics = {
        'total_samples': total_samples,
        'correct_predictions': correct_count,
        'accuracy': accuracy,
        'parsing_success_rate': parsing_success_rate,
        'predicted_label_distribution': dict(predicted_counter),
        'gold_label_distribution': dict(gold_counter)
    }
    
    return overall_metrics


def print_evaluation_summary(overall_metrics: Dict[str, Any]):
    """Print evaluation summary."""
    print(f"\n📊 BBQ Evaluation Results:")
    print(f"   Total samples evaluated: {overall_metrics.get('total_samples', 0)}")
    print(f"   🎯 Correct predictions: {overall_metrics.get('correct_predictions', 0)}")
    print(f"   🎯 Accuracy: {overall_metrics.get('accuracy', 0.0):.4f} ({overall_metrics.get('accuracy', 0.0)*100:.2f}%)")
    print(f"   🎯 Parsing success rate: {overall_metrics.get('parsing_success_rate', 0.0):.4f} ({overall_metrics.get('parsing_success_rate', 0.0)*100:.2f}%)")
    
    # Print label distributions
    predicted_dist = overall_metrics.get('predicted_label_distribution', {})
    gold_dist = overall_metrics.get('gold_label_distribution', {})
    
    if predicted_dist:
        print(f"   📊 Predicted label distribution:")
        for label, count in sorted(predicted_dist.items()):
            percentage = (count / overall_metrics.get('total_samples', 1)) * 100
            print(f"      Label {label}: {count} ({percentage:.1f}%)")
    
    if gold_dist:
        print(f"   📊 Gold label distribution:")
        for label, count in sorted(gold_dist.items()):
            percentage = (count / overall_metrics.get('total_samples', 1)) * 100
            print(f"      Label {label}: {count} ({percentage:.1f}%)")


def main():
    """Main function for BBQ evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate BBQ results with accuracy metrics")
    
    parser.add_argument("--model", type=str, default="gpt_4o_mini",
                        help="Model name for results directory (default: gpt_4o_mini)")
    parser.add_argument("--quantization", type=str, default=None, choices=["8bit", "4bit", "none"],
                        help="Quantization type for model directory (8bit, 4bit, or none)")
    parser.add_argument("--results_file", type=str,
                        help="Path to specific results CSV file (overrides model-based path)")
    parser.add_argument("--results_dir", type=str,
                        help="Directory containing results CSV files (overrides model-based path)")

    
    args = parser.parse_args()
    
    # Get the current script directory and build relative paths
    script_dir = Path(__file__).parent
    tasks_data_dir = script_dir.parent / "tasks_data"
    
    # Build model-specific paths
    model_dir_name = get_model_dir_name(args.model, args.quantization)
    model_results_dir = tasks_data_dir / "results" / "bbq" / model_dir_name
    
    # Find results files
    results_files = []
    
    if args.results_file:
        # User specified a custom file
        results_file_path = Path(args.results_file)
        if results_file_path.exists():
            results_files.append(results_file_path)
            print(f"🎯 Using specified results file: {results_file_path}")
        else:
            print(f"❌ Results file not found: {results_file_path}")
            return
    elif args.results_dir:
        # User specified a custom directory
        results_dir = Path(args.results_dir)
        if results_dir.exists():
            results_files = list(results_dir.glob("*.csv"))
            if not results_files:
                print(f"❌ No CSV files found in: {results_dir}")
                return
            print(f"🎯 Using specified results directory: {results_dir}")
        else:
            print(f"❌ Results directory not found: {results_dir}")
            return
    else:
        # Use model-based default paths
        if model_results_dir.exists():
            results_files = list(model_results_dir.glob("*.csv"))
            if not results_files:
                print(f"❌ No CSV files found in model directory: {model_results_dir}")
                return
            print(f"🎯 Using model directory '{args.model}': {model_results_dir}")
        else:
            print(f"❌ Model results directory not found: {model_results_dir}")
            print(f"💡 Available models in results directory:")
            bbq_results_dir = tasks_data_dir / "results" / "bbq"
            if bbq_results_dir.exists():
                for model_dir in bbq_results_dir.iterdir():
                    if model_dir.is_dir():
                        print(f"   - {model_dir.name}")
            return
    
    print(f"🔍 Found {len(results_files)} results files to evaluate")
    
    # Process each results file
    for results_file in results_files:
        print(f"\n📂 Processing: {results_file.name}")
        
        # Load results
        results = load_bbq_results(str(results_file))
        if not results:
            continue
        
        # Evaluate results
        evaluated_results = evaluate_bbq_results(results)
        
        # Calculate overall metrics
        overall_metrics = calculate_overall_metrics(evaluated_results)
        
        # Save results back to original files (overwrite)
        base_name = str(results_file).replace('.csv', '')
        
        # Save JSON (overwrite original JSON file)
        json_output_file = f"{base_name}.json"
        save_results_to_json(evaluated_results, json_output_file)
        
        # Save CSV (overwrite original CSV file)
        save_results_to_csv(evaluated_results, str(results_file))
        
        # Print summary
        print_evaluation_summary(overall_metrics)


if __name__ == "__main__":
    main() 