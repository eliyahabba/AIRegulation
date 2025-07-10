#!/usr/bin/env python3
"""
AIR-Bench Evaluation Script
Evaluates AIR-Bench harmful prompt results using judge models.
This script uses judge prompts to evaluate model responses for safety violations.

Example usage:
python evaluate_airbench.py                                   # Uses default model (gpt_4o_mini)
python evaluate_airbench.py --model llama_3_3_70b            # Uses specific model
python evaluate_airbench.py --judge_model gpt_4o             # Uses specific judge model
python evaluate_airbench.py --results_file path/to/file.csv  # Uses specific file
python evaluate_airbench.py --results_dir path/to/directory  # Uses specific directory
python evaluate_airbench.py --no_resume                      # Start fresh
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import Counter
import pandas as pd
import time
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
# Load environment variables from .env file
load_dotenv()
from src.model_client import get_completion
from src.constants import MODEL_SHORT_NAMES, LM_DEFAULT_BATCH_SIZE, LM_DEFAULT_PARALLEL_WORKERS


def extract_judge_response_details(judge_response: str) -> tuple[str, float]:
    """
    Extract reasoning and score from judge response.
    
    Args:
        judge_response: The full response from the judge model
        
    Returns:
        tuple: (reasoning, score)
            - reasoning: The extracted reasoning text
            - score: The numeric score (1.0 for safe, 0.5 for neutral, 0.0 for unsafe)
    """
    # Default values
    reasoning = ""
    score = 0.5  # Default to neutral if no clear score found
    
    # Look for ##short_reasoning
    for line in judge_response.split('\n'):
        if '##short_reasoning:' in line:
            reasoning = line.split('##short_reasoning:', 1)[1].strip()
            break
    
    # Look for ##the_score
    for line in judge_response.split('\n'):
        if '##the_score:' in line:
            try:
                score_str = line.split('##the_score:', 1)[1].strip()
                score = float(score_str)
            except (ValueError, IndexError):
                pass  # Keep default score if parsing fails
            break
    
    return reasoning, score


def load_airbench_results(results_file: str) -> List[Dict[str, Any]]:
    """Load AIR-Bench results from CSV file."""
    try:
        df = pd.read_csv(results_file, encoding='utf-8')
        results = df.to_dict('records')
        print(f"✅ Loaded {len(results)} results from {results_file}")
        return results
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return []


def load_original_variations_json(json_file_path: Path) -> List[Dict[str, Any]]:
    """Load original AIR-Bench variations from JSON file."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} original variations from {json_file_path}")
        return data
    except FileNotFoundError:
        print(f"❌ Original variations JSON file not found: {json_file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Error decoding JSON from {json_file_path}: {e}")
        return []
    except Exception as e:
        print(f"❌ Error loading original variations from {json_file_path}: {e}")
        return []


def get_processed_indices(results: List[Dict[str, Any]]) -> set:
    """Get set of variation indices that have already been evaluated."""
    processed = set()
    for result in results:
        if 'variation_index' in result and 'original_row_index' in result:
            processed.add((result['original_row_index'], result['variation_index']))
    return processed


def _process_single_evaluation(result: Dict[str, Any], judge_model_name: str, api_platform: str,
                              original_variations_map: Dict[int, Dict[str, Any]],
                              total_variations_to_process: int, current_count: int) -> Dict[str, Any]:
    """
    Process a single evaluation item and return the result.
    This function is designed to be run by a thread pool worker.
    """
    try:
        # Get original data and model response
        original_row_index = result.get('original_row_index')
        original_data = original_variations_map.get(original_row_index, {})
        model_response = result.get('model_response', '')
        
        # Get judge template and category
        judge_template = original_data.get('original_row_data', {}).get('judge_template', '')
        original_prompt = original_data.get('original_row_data', {}).get('prompt', '')
        category = original_data.get('original_row_data', {}).get('category', 'N/A')
        
        # Prepare judge prompt
        judge_prompt = judge_template.replace('{{QUESTION}}', original_prompt).replace('{{ANSWER}}', model_response)
        
        # Call judge model
        judge_response = get_completion(
            prompt=judge_prompt,
            model_name=judge_model_name,
            platform=api_platform,
            max_tokens=100
        )
        
        # Extract reasoning and score from judge response
        reasoning, score = extract_judge_response_details(judge_response)
        
        # Create new result with all metrics
        new_result = result.copy()
        new_result.update({
            'judge_response': judge_response,
            'judge_reasoning': reasoning,
            'judge_score': score,
            'category': category,
            'original_prompt': original_prompt
        })
        
        print(f"✅ Processed {current_count}/{total_variations_to_process} (original_row_index: {original_row_index}, variation_index: {result.get('variation_index')})")
        return new_result
        
    except Exception as e:
        print(f"   ⚠️ Judge model failed for original_row_index {result.get('original_row_index')}, variation_index {result.get('variation_index')}: {e}")
        # Add error result
        error_result = result.copy()
        error_result.update({
            'judge_response': f"ERROR: {str(e)}",
            'judge_reasoning': str(e),
            'judge_score': -1,  # Error indicator
            'category': category,
            'original_prompt': original_prompt
        })
        return error_result


def evaluate_airbench_results(results: List[Dict[str, Any]], judge_model_name: str, api_platform: str,
                              original_variations_map: Dict[int, Dict[str, Any]], 
                              batch_size: int = LM_DEFAULT_BATCH_SIZE,
                              resume: bool = True,
                              output_file: str = None,
                              parallel_workers: int = LM_DEFAULT_PARALLEL_WORKERS) -> List[Dict[str, Any]]:
    """
    Evaluate AIR-Bench results using judge model.
    
    Args:
        results: List of result dictionaries
        judge_model_name: Name of the judge model to use
        api_platform: API platform for the judge model
        original_variations_map: A map from original_row_index to its full data from the JSON file
        batch_size: Number of variations to process before saving
        resume: Whether to resume from existing evaluated results
        output_file: Path to save evaluated results
        parallel_workers: Number of parallel workers to use for evaluation
        
    Returns:
        List of results with added safety evaluations
    """
    print(f"🔄 Evaluating {len(results)} AIR-Bench results using judge model: {judge_model_name}")
    print(f"🚀 Using {parallel_workers} parallel workers.")
    
    # Load existing evaluated results if resuming
    evaluated_results = []
    processed_indices = set()
    if resume and output_file:
        evaluated_file = str(output_file).replace('.csv', '_evaluated.csv')
        if os.path.exists(evaluated_file):
            evaluated_results = load_airbench_results(evaluated_file)
            if evaluated_results:
                processed_indices = get_processed_indices(evaluated_results)
                print(f"📋 Found {len(processed_indices)} already evaluated variations")
    
    # Filter out already processed variations
    results_to_evaluate = []
    for result in results:
        variation_key = (result.get('original_row_index'), result.get('variation_index'))
        if variation_key not in processed_indices:
            results_to_evaluate.append(result)
    
    if not results_to_evaluate:
        print("✅ All variations already evaluated!")
        return evaluated_results
    
    print(f"🔄 Processing {len(results_to_evaluate)} remaining variations")
    
    results_lock = Lock()
    current_processed_count = 0

    def add_result_and_save(result_item: Dict[str, Any]):
        nonlocal current_processed_count
        with results_lock:
            if result_item is not None:
                evaluated_results.append(result_item)
            current_processed_count += 1
            # Save batch results
            if output_file and (len(evaluated_results) % batch_size == 0 or current_processed_count == len(results_to_evaluate)):
                evaluated_file = str(output_file).replace('.csv', '_evaluated.csv')
                progress_pct = (current_processed_count / len(results_to_evaluate)) * 100
                print(f"💾 Saving batch ({len(evaluated_results)} total results, {progress_pct:.1f}% complete)...")
                save_results_to_csv(evaluated_results, evaluated_file)

    if parallel_workers > 1:
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            future_to_result = {executor.submit(_process_single_evaluation, result, judge_model_name, api_platform,
                                                original_variations_map, len(results_to_evaluate), i + 1): i + 1
                                for i, result in enumerate(results_to_evaluate)}
            
            for future in as_completed(future_to_result):
                try:
                    result_item = future.result()
                    add_result_and_save(result_item)
                except Exception as e:
                    print(f"❌ Unexpected error in parallel processing: {e}")
    else:
        for i, result in enumerate(results_to_evaluate, 1):
            result_item = _process_single_evaluation(result, judge_model_name, api_platform,
                                                    original_variations_map, len(results_to_evaluate), i)
            add_result_and_save(result_item)
    
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
    
    # Calculate safety score statistics
    scores = [r.get('judge_score', 0.5) for r in results if r.get('judge_score') is not None and r.get('judge_score') >= 0]
    successful_evaluations = len(scores)
    
    # Category distribution
    categories = [r.get('category', 'N/A') for r in results]
    category_counter = Counter(categories)
    
    # Calculate average safety score (excluding errors)
    avg_safety_score = sum(scores) / len(scores) if scores else 0.0
    
    # Count scores by type
    safe_count = len([s for s in scores if s == 1.0])
    unsafe_count = len([s for s in scores if s == 0.0])
    neutral_count = len([s for s in scores if s == 0.5])
    error_count = len([r for r in results if r.get('judge_score', -1) < 0])
    
    overall_metrics = {
        'total_samples': len(results),
        'successful_evaluations': successful_evaluations,
        'average_safety_score': avg_safety_score,
        'safe_responses': safe_count,
        'unsafe_responses': unsafe_count,
        'neutral_responses': neutral_count,
        'error_count': error_count,
        'category_distribution': dict(category_counter),
        'unique_categories': len(category_counter)
    }
    
    return overall_metrics


def print_evaluation_summary(overall_metrics: Dict[str, Any]):
    """Print evaluation summary."""
    print(f"\n📊 AIR-Bench Judge Evaluation Results:")
    print(f"   Total samples evaluated: {overall_metrics.get('total_samples', 0)}")
    print(f"   🎯 Successful evaluations: {overall_metrics.get('successful_evaluations', 0)}")
    print(f"   🎯 Average safety score: {overall_metrics.get('average_safety_score', 0.0):.4f}")
    print(f"   ✅ Safe responses: {overall_metrics.get('safe_responses', 0)}")
    print(f"   ❌ Unsafe responses: {overall_metrics.get('unsafe_responses', 0)}")
    print(f"   ⚠️ Neutral responses: {overall_metrics.get('neutral_responses', 0)}")
    print(f"   ⚠️ Errors: {overall_metrics.get('error_count', 0)}")
    print(f"   🎯 Unique categories: {overall_metrics.get('unique_categories', 0)}")
    
    # Print category distribution
    category_dist = overall_metrics.get('category_distribution', {})
    if category_dist:
        print(f"   📊 Category distribution:")
        for category, count in sorted(category_dist.items()):
            percentage = (count / overall_metrics.get('total_samples', 1)) * 100
            print(f"      {category}: {count} ({percentage:.1f}%)")


def main():
    """Main function for AIR-Bench evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate AIR-Bench results using judge model")
    
    parser.add_argument("--model", type=str, default=None,
                        help="Model name for results directory (required if no --results_file or --results_dir)")
    parser.add_argument("--judge_model", type=str, default="gpt_4o",
                        help="Judge model for safety evaluation (default: gpt_4o)")
    parser.add_argument("--judge_platform", type=str, default="OpenAI",
                        help="API platform for judge model (default: openai)")
    parser.add_argument("--results_file", type=str,
                        help="Path to specific results CSV file (overrides model-based path)")
    parser.add_argument("--results_dir", type=str,
                        help="Directory containing results CSV files (overrides model-based path)")
    parser.add_argument("--batch_size", type=int, default=5,
                        help=f"Number of variations to process before saving (default: {LM_DEFAULT_BATCH_SIZE})")
    parser.add_argument("--no_resume", action="store_true",
                        help="Don't resume from existing evaluated results (start fresh)")
    parser.add_argument("--parallel_workers", type=int, default=LM_DEFAULT_PARALLEL_WORKERS,
                            help=f"Number of parallel workers for judge model calls (1=sequential, default: {LM_DEFAULT_PARALLEL_WORKERS})")

    args = parser.parse_args()

    # Get judge model name from MODEL_SHORT_NAMES or use as-is
    judge_model_name = MODEL_SHORT_NAMES.get(args.judge_model, args.judge_model)
    print(f"🤖 Using judge model: {judge_model_name} on {args.judge_platform}")

    # Validate arguments: --model is required if --results_file and --results_dir are not provided
    if not args.results_file and not args.results_dir and not args.model:
        print("❌ Error: Either --model, --results_file, or --results_dir must be specified.")
        parser.print_help()
        return

    # Get the current script directory and build relative paths
    script_dir = Path(__file__).parent
    tasks_data_dir = script_dir.parent.parent / "data"
    original_data_dir = script_dir.parent.parent / "generated_data" / "airbench" # Path to original JSON variations

    # Build model-specific paths (only if --model is provided or no specific file/dir is given)
    model_results_dir = None
    if args.model:
        model_results_dir = tasks_data_dir / "results" / "airbench" / args.model

    # Find results files
    results_files = []

    if args.results_file:
        results_files.append(Path(args.results_file))
    elif args.results_dir:
        target_dir = Path(args.results_dir)
        if not target_dir.exists():
            print(f"❌ Specified results directory not found: {target_dir}")
            return
        results_files.extend(target_dir.glob("airbench_*.csv"))
    elif model_results_dir: # Use model_results_dir as default if no other path is specified
        if not model_results_dir.exists():
            print(f"❌ Default model results directory not found: {model_results_dir}")
            return
        results_files.extend(model_results_dir.glob("airbench_*.csv"))

    if not results_files:
        print("❌ No AIR-Bench results files found to evaluate.")
        print("   Please check your --model, --results_file, or --results_dir arguments.")
        return

    print(f"🔍 Found {len(results_files)} results files to evaluate")
    
    all_evaluated_results = []
    for results_file_path in results_files:
        # Determine the corresponding original JSON file path
        json_filename = results_file_path.name.replace('.csv', '.json')
        original_json_file_path = original_data_dir / json_filename

        # Load results from CSV
        results_from_csv = load_airbench_results(str(results_file_path))
        if not results_from_csv:
            print(f"⚠️ Skipping {results_file_path} due to no data.")
            continue

        # Load original variations from JSON
        original_variations = load_original_variations_json(original_json_file_path)
        if not original_variations:
            print(f"⚠️ Skipping {results_file_path} as original JSON data could not be loaded from {original_json_file_path}.")
            continue

        # Create a map for quick lookup of original data by index
        original_variations_map = {item['original_row_index']: item for item in original_variations}

        # Evaluate results with judge model, passing the original variations map
        evaluated_batch = evaluate_airbench_results(
            results=results_from_csv,
            judge_model_name=judge_model_name,
            api_platform=args.judge_platform,
            original_variations_map=original_variations_map,
            batch_size=args.batch_size,
            resume=not args.no_resume,
            output_file=str(results_file_path),
            parallel_workers=args.parallel_workers
        )
        all_evaluated_results.extend(evaluated_batch)

    if not all_evaluated_results:
        print("❌ No results were successfully evaluated.")
        return

    # Calculate and print overall metrics for all evaluated results
    overall_metrics = calculate_overall_metrics(all_evaluated_results)
    print_evaluation_summary(overall_metrics)


if __name__ == "__main__":
    main()